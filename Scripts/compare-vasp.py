#!/usr/bin/env python3
"""Compare completed CPU and Metal VASP outputs without redistributing inputs."""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def number(text):
    return float(text.replace('D', 'E').replace('d', 'e'))


def read(directory, log_name=None):
    d = Path(directory)
    root = ET.parse(d / 'vasprun.xml').getroot()
    calculations = root.findall('calculation')
    if not calculations:
        raise ValueError(f'{d}/vasprun.xml contains no completed calculation')
    calc = calculations[-1]
    out = (d / 'OUTCAR').read_text(errors='replace')
    iteration_headers = list(re.finditer(r'Iteration\s+\d+\s*\(\s*\d+\s*\)', out))
    final_iteration_out = out[iteration_headers[-1].start():] if iteration_headers else out
    log_path = d / log_name if log_name else next((d / name for name in ('run.log', 'vasp.log', 'metal.log', 'cpu.log', 'stdout.log') if (d / name).is_file()), None)
    # OSZICAR contains the same electronic iteration record when stdout was
    # redirected under an unknown name. OUTCAR remains the convergence source.
    if log_path is None and (d / 'OSZICAR').is_file():
        log_path = d / 'OSZICAR'
    log = log_path.read_text(errors='replace') if log_path else ''

    def arr(name):
        return [[number(x) for x in (v.text or '').split()]
                for v in calc.findall(f"varray[@name='{name}']/v")]

    energy = calc.find("energy/i[@name='e_fr_energy']")
    if energy is None or not energy.text:
        raise ValueError(f'{d}/vasprun.xml is missing final free energy')
    scf = re.findall(r'^\s*(?:DAV|RMM|CG|DMP)\s*:\s*\d+\s+([-+0-9.DEde]+)', log, re.M)
    scsteps = calc.findall('scstep')
    # XML provides full precision and does not depend on the stdout filename.
    last_scf = scsteps[-1].find("energy/i[@name='e_fr_energy']") if scsteps else None
    scf_energy = number(last_scf.text) if last_scf is not None and last_scf.text else number(scf[-1]) if scf else None
    stats_path = d / 'VASP_METAL_STATS.json'
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if not isinstance(stats, dict):
        raise ValueError(f'{stats_path} must contain a JSON object')
    inputs = {name: hashlib.sha256((d / name).read_bytes()).hexdigest()
              for name in ('INCAR', 'KPOINTS', 'POSCAR', 'POTCAR') if (d / name).is_file()}
    atom_nodes = root.findall('atominfo/atoms')
    atom_count = int(atom_nodes[0].text) if atom_nodes and atom_nodes[0].text else None
    return dict(directory=str(d.resolve()), energy_ev=number(energy.text),
                scf_final_energy_ev=scf_energy, forces=arr('forces'), stress=arr('stress'),
                iterations=len(scsteps), ionic_steps=len(calculations), atom_count=atom_count,
                converged='aborting loop because EDIFF is reached' in final_iteration_out,
                completed='General timing and accounting' in out,
                hermitian_warnings=len(re.findall(r'not\s+hermitian', out + '\n' + log, re.I)),
                input_sha256=inputs, log_file=str(log_path) if log_path else None, stats=stats)


def compare(ref, candidate, tolerances=None, required_gpu_ops=()):
    tolerances = tolerances or dict(energy_ev=1e-5, force_ev_per_ang=1e-4, stress_kb=1e-3)

    def maxerr(a, b):
        if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
            return None
        values = [abs(x - y) for xx, yy in zip(a, b) for x, y in zip(xx, yy)]
        return max(values) if values and all(math.isfinite(v) for v in values) else None

    e = abs(candidate['energy_ev'] - ref['energy_ev'])
    e = e if math.isfinite(e) else None
    f = maxerr(candidate['forces'], ref['forces'])
    s = maxerr(candidate['stress'], ref['stress'])
    es = abs(candidate['scf_final_energy_ev'] - ref['scf_final_energy_ev']) if all(x['scf_final_energy_ev'] is not None for x in (ref, candidate)) else None
    es = es if es is not None and math.isfinite(es) else None
    arrays_valid = all(len(x['forces']) > 0 and all(len(row) == 3 for row in x['forces'])
                       and len(x['stress']) == 3 and all(len(row) == 3 for row in x['stress'])
                       and (x.get('atom_count') is None or len(x['forces']) == x['atom_count'])
                       for x in (ref, candidate))
    finite = all(math.isfinite(v) for x in (ref, candidate) for array in (x['forces'], x['stress']) for row in array for v in row)
    numerical = (arrays_valid and finite and e is not None and es is not None and f is not None and s is not None
                 and e <= tolerances['energy_ev'] and es <= tolerances['energy_ev']
                 and f <= tolerances['force_ev_per_ang'] and s <= tolerances['stress_kb']
                 and all(x['converged'] and x['completed'] and x['hermitian_warnings'] == 0 for x in (ref, candidate))
                 and candidate['iterations'] == ref['iterations']
                 and candidate.get('ionic_steps', 1) == ref.get('ionic_steps', 1))
    counters = candidate['stats']
    counter_errors = [key for key, value in counters.items() if (key.endswith(('_calls', '_success', '_eligible')) or key.startswith('adaptive_cpu_'))
                      and (isinstance(value, bool) or not isinstance(value, int) or value < 0)]

    def count(key):
        value = counters.get(key, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    success_keys = dict(gemm='gpu_success', fft='fft3d_success', batch_fft='fft3d_batch_success',
                        reduce='dsum_gpu_success', dot='zdotc_gpu_success', eigensolver='zhegv_gpu_success',
                        standard_eigensolver='zheev_gpu_success', selected_eigensolver='zheevx_gpu_success',
                        hamiltonian='hamiltonian_success', local_potential='vh_apply_gpu_success',
                        overlap='orth_overlap_success', full_orthogonalization='full_orth_success')
    operations = {name: count(key) for name, key in success_keys.items()}
    missing_ops = [name for name in required_gpu_ops if operations.get(name, 0) == 0]
    gpu = counters.get('gpu_device') == 1 and any(operations.values()) and not counter_errors and not missing_ops
    pairs = [('gemm', 'gpu_eligible', 'gpu_success'), ('reduce', 'dsum_gpu_eligible', 'dsum_gpu_success'),
             ('generalized_eigensolver', 'zhegv_gpu_eligible', 'zhegv_gpu_success'),
             ('standard_eigensolver', 'zheev_gpu_eligible', 'zheev_gpu_success'),
             ('selected_eigensolver', 'zheevx_gpu_eligible', 'zheevx_gpu_success'),
             ('dot', 'zdotc_gpu_eligible', 'zdotc_gpu_success'),
             ('hamiltonian', 'hamiltonian_calls', 'hamiltonian_success'),
             ('local_potential', 'vh_apply_calls', 'vh_apply_gpu_success'),
             ('overlap', 'orth_overlap_calls', 'orth_overlap_success'),
             ('full_orthogonalization', 'full_orth_calls', 'full_orth_success')]
    rejected = {name: count(a) - count(b) for name, a, b in pairs if count(a) != count(b)}
    # The shared adaptive counter includes both scalar and batched FFT calls.
    fft_calls = count('fft3d_calls') + count('fft3d_batch_calls')
    fft_success = count('fft3d_success') + count('fft3d_batch_success')
    fft_adaptive = count('adaptive_cpu_fft')
    if fft_calls != fft_success + fft_adaptive:
        rejected['fft'] = fft_calls - fft_success - fft_adaptive
    impossible = [name for name, value in rejected.items() if value < 0]
    coverage = gpu and not rejected and not impossible
    input_names = ('INCAR', 'KPOINTS', 'POSCAR', 'POTCAR')
    mismatched = [name for name in input_names if name in ref.get('input_sha256', {}) and name in candidate.get('input_sha256', {})
                  and ref['input_sha256'][name] != candidate['input_sha256'][name]]
    input_complete = all(name in x.get('input_sha256', {}) for x in (ref, candidate) for name in input_names)
    identity = not mismatched if input_complete else False if mismatched else None
    return dict(reference=ref, candidate=candidate, delta_energy_ev=e, delta_scf_final_energy_ev=es,
                max_force_delta_ev_per_ang=f, max_stress_delta_kb=s, numerical_gate=bool(numerical),
                numeric_arrays_present_and_valid=arrays_valid, tolerances=tolerances,
                input_identity_gate=identity, mismatched_input_files=mismatched,
                gpu_execution_gate=bool(gpu), gpu_operations=operations, required_gpu_ops=list(required_gpu_ops),
                missing_required_gpu_ops=missing_ops, gpu_all_eligible_completed_gate=bool(coverage),
                gpu_rejected_or_cpu_fallbacks=rejected, invalid_gpu_counters=counter_errors,
                adaptive_cpu_gemm=count('adaptive_cpu_gemm'), adaptive_cpu_fft=fft_adaptive,
                adaptive_cpu_eigh=count('adaptive_cpu_eigh'), lobpcg_success=count('lobpcg_gpu_success'),
                lobpcg_jacobi_rescue=count('lobpcg_jacobi_rescue'),
                gpu_gate_scope='At least one recorded GPU operator succeeded; this does not establish full VASP GPU execution.',
                numerical_gate_scope='Final ionic step only; this does not validate an entire relaxation or dynamics trajectory.',
                performance_gate='not evaluated: supply independently measured wall times')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--reference-log')
    ap.add_argument('--candidate-log')
    ap.add_argument('--output')
    ap.add_argument('--energy-tolerance', type=float, default=1e-5, help='eV, applied to free energy and final SCF energy')
    ap.add_argument('--force-tolerance', type=float, default=1e-4, help='eV/Angstrom')
    ap.add_argument('--stress-tolerance', type=float, default=1e-3, help='kbar')
    ap.add_argument('--require-gpu-ops', default='', help='Comma-separated operation names shown in gpu_operations')
    ap.add_argument('--require-all-eligible', action='store_true')
    ap.add_argument('--require-identical-inputs', action='store_true')
    ap.add_argument('--numerical-only', action='store_true', help='Compare CPU baselines without requiring GPU execution')
    a = ap.parse_args()
    tolerances = dict(energy_ev=a.energy_tolerance, force_ev_per_ang=a.force_tolerance, stress_kb=a.stress_tolerance)
    if any(not math.isfinite(value) or value < 0 for value in tolerances.values()):
        ap.error('tolerances must be finite and nonnegative')
    required = tuple(x.strip() for x in a.require_gpu_ops.split(',') if x.strip())
    try:
        r = compare(read(a.reference, a.reference_log), read(a.candidate, a.candidate_log), tolerances, required)
    except (OSError, ValueError, ET.ParseError) as error:
        print(f'VASP comparison failed: {error}', file=sys.stderr)
        return 2
    # Sanitise non-finite raw fields as well as deltas so the report is valid JSON.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    text = json.dumps(clean(r), indent=2, allow_nan=False)
    if a.output:
        Path(a.output).write_text(text + '\n')
    print(text)
    passed = r['numerical_gate'] and not r['mismatched_input_files'] and (a.numerical_only or r['gpu_execution_gate'])
    passed = passed and (not a.require_all_eligible or r['gpu_all_eligible_completed_gate'])
    passed = passed and (not a.require_identical_inputs or r['input_identity_gate'] is True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
