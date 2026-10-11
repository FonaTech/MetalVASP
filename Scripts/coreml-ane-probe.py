#!/usr/bin/env python3
"""Independent Core ML FP16/FP32 matrix probe; never a VASP operator path.

Query MLComputePlan per-operation preferred/supported devices. A compute-unit
switch alone is not placement evidence, and a plan is not a hardware trace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time


def sha256_tree(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def is_ane_arithmetic(operation):
    return operation['operator'].rsplit('.', 1)[-1] in ('linear', 'matmul', 'conv') and \
           bool(operation.get('preferred')) and operation['preferred']['class'] == 'MLNeuralEngineComputeDevice'


def prepare(output, size, batch, seed, precisions, kinds):
    import numpy as np
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    if output.exists():
        raise ValueError('refusing to overwrite an existing probe directory')
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    inputs = rng.normal(0.0, 1.0, (3, 1, batch, size)).astype(np.float32)
    # Exact FP32 input values widened to FP64: error isolates the model path,
    # rather than conflating float64 -> float32 input quantization with it.
    reference = inputs.astype(np.float64) @ weight.astype(np.float64).T
    rounded_input_weight_reference = inputs.astype(np.float16).astype(np.float64) @ weight.astype(np.float16).astype(np.float64).T
    rounded_fp16_output_reference = rounded_input_weight_reference.astype(np.float16).astype(np.float64)
    rounded_fp32_output_reference = reference.astype(np.float32).astype(np.float64)
    np.savez(output / 'inputs-and-reference.npz', weight=weight, inputs=inputs, reference=reference,
             rounded_input_weight_reference=rounded_input_weight_reference,
             rounded_fp16_output_reference=rounded_fp16_output_reference,
             rounded_fp32_output_reference=rounded_fp32_output_reference)
    manifest = {'schema': 1, 'seed': seed, 'size': size, 'batch': batch,
                'host': platform.platform(), 'coremltools_version': ct.__version__,
                'numpy_version': np.__version__, 'models': [],
                'generation': 'seeded FP32 inputs/weights; exact-value FP64 matrix product reference',
                'model_source': 'Core ML MIL Builder; no external trained model or Torch conversion',
                'vasp_integration': False}
    for precision in precisions:
        for kind in kinds:
            if kind == 'static_linear':
                @mb.program(input_specs=[mb.TensorSpec(shape=(1, batch, size))], opset_version=ct.target.macOS13)
                def program(x):
                    return mb.linear(x=x, weight=weight, bias=np.zeros(size, dtype=np.float32), name='result')
            elif kind == 'dynamic_matmul':
                @mb.program(input_specs=[mb.TensorSpec(shape=(1, batch, size)),
                                        mb.TensorSpec(shape=(size, size))], opset_version=ct.target.macOS13)
                def program(x, w):
                    return mb.matmul(x=x, y=w, transpose_y=True, name='result')
            else:
                @mb.program(input_specs=[mb.TensorSpec(shape=(1, size, 1, batch))], opset_version=ct.target.macOS13)
                def program(x):
                    return mb.conv(x=x, weight=weight.reshape(size, size, 1, 1),
                                   bias=np.zeros(size, dtype=np.float32),
                                   strides=[1, 1], pad_type='valid', groups=1, name='result')
            model = ct.convert(program, convert_to='mlprogram', skip_model_load=True,
                               minimum_deployment_target=ct.target.macOS13,
                               compute_precision=ct.precision.FLOAT16 if precision == 'fp16' else ct.precision.FLOAT32)
            name = kind + '_' + precision
            package = output / (name + '.mlpackage')
            model.save(str(package))
            record = {'name': name, 'kind': kind, 'compute_precision': precision,
                      'package': package.name, 'output': model.get_spec().description.output[0].name,
                      'input_layout': 'NCHW 1x1 convolution' if kind == 'conv1x1' else 'row-major matrix',
                      'files_sha256': sha256_tree(package)}
            manifest['models'].append(record)
    manifest['input_sha256'] = hashlib.sha256((output / 'inputs-and-reference.npz').read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('prepared ' + str(output), flush=True)


def device_record(device):
    if device is None:
        return None
    result = {'class': type(device).__name__}
    count = getattr(device, 'total_core_count', None)
    if count is not None:
        result['total_core_count'] = count
    return result


def compute_plan_record(plan):
    operations = []
    if plan.model_structure.program is None:
        raise ValueError('probe requires an ML Program compute plan')
    def visit(block, function, depth=0):
        for operation in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
            record = {'function': function, 'depth': depth, 'operator': operation.operator_name,
                      'operator_basename': operation.operator_name.rsplit('.', 1)[-1],
                      'outputs': [output.name for output in operation.outputs],
                      'preferred': device_record(usage.preferred_compute_device) if usage else None,
                      'supported': [device_record(d) for d in usage.supported_compute_devices] if usage else [],
                      'estimated_cost_weight': cost.weight if cost else None}
            operations.append(record)
            for nested in operation.blocks:
                visit(nested, function, depth + 1)
    for name, function in plan.model_structure.program.functions.items():
        visit(function.block, name)
    return operations


def run_probe(output, repeats, warmups):
    import numpy as np
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.compute_device import MLComputeDevice
    if platform.system() != 'Darwin' or tuple(int(part) for part in platform.mac_ver()[0].split('.')[:2]) < (14, 4):
        raise ValueError('MLComputePlan placement requires an actual macOS14.4+ runtime')
    if (output / 'results.json').exists():
        raise ValueError('refusing to overwrite existing probe results')
    manifest = json.loads((output / 'manifest.json').read_text())
    if hashlib.sha256((output / 'inputs-and-reference.npz').read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('prepared probe inputs changed')
    data = np.load(output / 'inputs-and-reference.npz')
    # Older prepared packets predate component references. Recompute locally
    # without altering their immutable inputs or manifest.
    rounded_input_weight_reference = (data['rounded_input_weight_reference']
        if 'rounded_input_weight_reference' in data.files else
        data['inputs'].astype(np.float16).astype(np.float64) @ data['weight'].astype(np.float16).astype(np.float64).T)
    rounded_fp16_output_reference = (data['rounded_fp16_output_reference']
        if 'rounded_fp16_output_reference' in data.files else rounded_input_weight_reference.astype(np.float16).astype(np.float64))
    rounded_fp32_output_reference = (data['rounded_fp32_output_reference']
        if 'rounded_fp32_output_reference' in data.files else data['reference'].astype(np.float32).astype(np.float64))
    report = {'schema': 1, 'host': platform.platform(), 'models': [],
              'warmups_excluded_per_mode': warmups, 'measured_repeats_per_mode': repeats,
              'placement_evidence': 'MLComputePlan anticipated preferred/supported devices; no hardware execution trace',
              'observed_ane_execution_trace': None, 'vasp_integration': False,
              'performance_claim_accepted': False,
              'timing_environment': 'idle-host conditions are not audited by this script',
              'fp64_semantics_supported': False,
              'precision_gates': {'fp16_probe_absolute': 0.005, 'fp32_probe_absolute': 1e-5,
                                  'ds_operator_absolute': 1e-10},
              'available_devices': [device_record(d) for d in MLComputeDevice.get_all_compute_devices()]}
    for record in manifest['models']:
        model_record = dict(record, modes=[])
        report['models'].append(model_record)
        package = output / record['package']
        if sha256_tree(package) != record['files_sha256']:
            raise ValueError('prepared Core ML package changed: ' + record['name'])
        compiled_path = output / (record['name'] + '.mlmodelc')
        try:
            begin = time.perf_counter()
            ct.models.utils.compile_model(str(package), destination_path=str(compiled_path))
            model_record['compile_seconds'] = time.perf_counter() - begin
            for mode_name, units in (('cpu_only', ct.ComputeUnit.CPU_ONLY), ('cpu_and_ne', ct.ComputeUnit.CPU_AND_NE)):
                mode = {'compute_units': mode_name, 'status': 'pending'}
                model_record['modes'].append(mode)
                try:
                    plan = MLComputePlan.load_from_path(str(compiled_path), compute_units=units)
                    mode['operations'] = compute_plan_record(plan)
                    mode['ane_preferred_operations'] = sum(1 for op in mode['operations']
                        if op['preferred'] and op['preferred']['class'] == 'MLNeuralEngineComputeDevice')
                    mode['ane_preferred_arithmetic_operations'] = sum(1 for op in mode['operations']
                        if is_ane_arithmetic(op))
                    model = ct.models.CompiledMLModel(str(compiled_path), compute_units=units)
                    timings, errors, residual_errors = [], [], []
                    if record['compute_precision'] == 'fp16':
                        rounded_reference = rounded_input_weight_reference
                        output_rounded_reference = rounded_fp16_output_reference
                    else:
                        rounded_reference = data['reference']
                        output_rounded_reference = rounded_fp32_output_reference
                    mode['input_weight_quantization_max_absolute'] = float(np.max(np.abs(rounded_reference - data['reference'])))
                    mode['reference_output_rounding_max_absolute'] = float(np.max(np.abs(output_rounded_reference - rounded_reference)))
                    for iteration in range(warmups + repeats):
                        index = iteration % len(data['inputs'])
                        feed = {'x': data['inputs'][index]}
                        if record['kind'] == 'dynamic_matmul':
                            feed['w'] = data['weight']
                        elif record['kind'] == 'conv1x1':
                            feed['x'] = np.ascontiguousarray(feed['x'].transpose(0, 2, 1)[:, :, None, :])
                        begin = time.perf_counter()
                        result = model.predict(feed)
                        elapsed = time.perf_counter() - begin
                        actual = np.asarray(result[record['output']], dtype=np.float64)
                        if record['kind'] == 'conv1x1':
                            actual = actual[:, :, 0, :].transpose(0, 2, 1)
                        if actual.shape != data['reference'][index].shape or not np.isfinite(actual).all():
                            raise ValueError('nonfinite/wrong-shaped Core ML output')
                        errors.append(float(np.max(np.abs(actual - data['reference'][index]))))
                        residual_errors.append(float(np.max(np.abs(actual - output_rounded_reference[index]))))
                        if iteration >= warmups:
                            timings.append(elapsed)
                    mode.update(status='prediction-completed', prediction_seconds=timings,
                                median_seconds=statistics.median(timings),
                                max_absolute_error=max(errors),
                                execution_vs_rounded_input_weight_and_output_reference_max_absolute=max(residual_errors),
                                probe_precision_gate_passed=max(errors) <= report['precision_gates'][record['compute_precision'] + '_probe_absolute'],
                                ds_precision_gate_passed=max(errors) <= report['precision_gates']['ds_operator_absolute'],
                                strict_ane_placement_gate_passed=mode['ane_preferred_arithmetic_operations'] > 0)
                except Exception as error:
                    mode.update(status='failed', error=str(error))
                (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
                print(record['name'] + ' ' + mode_name + ' ' + mode['status'], flush=True)
            completed = {mode['compute_units']: mode for mode in model_record['modes']
                         if mode['status'] == 'prediction-completed'}
            if 'cpu_only' in completed and 'cpu_and_ne' in completed:
                model_record['cpu_to_ne_wall_ratio'] = (completed['cpu_only']['median_seconds'] /
                                                       completed['cpu_and_ne']['median_seconds'])
        except Exception as error:
            model_record.update(status='failed', error=str(error))
        (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = ('completed' if all(model.get('modes') and all(mode['status'] == 'prediction-completed'
                         for mode in model['modes']) for model in report['models']) else 'partially-failed')
    report['production_ane_path_accepted'] = False
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--precision', choices=('fp16', 'fp32', 'both'), default='both')
    parser.add_argument('--kinds', default='static_linear,dynamic_matmul',
                        help='Comma-separated static_linear,dynamic_matmul,conv1x1')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmups', type=int, default=3)
    args = parser.parse_args()
    if args.prepare_only and args.run_prepared:
        parser.error('--prepare-only and --run-prepared are exclusive')
    if min(args.size, args.batch, args.repeats) <= 0 or args.warmups < 0:
        parser.error('size/batch/repeats must be positive; warmups must be nonnegative')
    kinds = args.kinds.split(',')
    if len(set(kinds)) != len(kinds) or any(kind not in ('static_linear', 'dynamic_matmul', 'conv1x1') for kind in kinds):
        parser.error('--kinds must select distinct known model kinds')
    try:
        if not args.run_prepared:
            prepare(args.output, args.size, args.batch, args.seed,
                    ('fp16', 'fp32') if args.precision == 'both' else (args.precision,), kinds)
        if args.prepare_only:
            return 0
        return run_probe(args.output, args.repeats, args.warmups)
    except Exception as error:
        failure = {'status': 'failed', 'error': str(error), 'exception': type(error).__name__,
                   'vasp_integration': False, 'production_ane_path_accepted': False}
        failure_path = args.output / 'probe-failure.json'
        if args.output.is_dir() and not failure_path.exists():
            failure_path.write_text(json.dumps(failure, indent=2) + '\n')
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
