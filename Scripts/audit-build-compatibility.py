#!/usr/bin/env python3
"""Compile actual installed SDKs and audit a Mach-O dependency closure.

This does not emulate an older OS/GPU. SDK syntax/link checks and dependency
minimum versions are separate from target-machine runtime acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile


def version(value):
    parts = tuple(int(part) for part in value.split('.'))
    return parts + (0,) * max(0, 3 - len(parts))


def parse_load_commands(content):
    result = {'minimum_versions': [], 'rpaths': []}
    for block in re.split(r'Load command \d+\n', content):
        if re.search(r'\bcmd LC_BUILD_VERSION\b', block):
            minimum = re.search(r'^\s*minos (\S+)', block, re.M)
            if minimum:
                result['minimum_versions'].append(minimum.group(1))
        if re.search(r'\bcmd LC_VERSION_MIN_MACOSX\b', block):
            minimum = re.search(r'^\s*version (\S+)', block, re.M)
            if minimum:
                result['minimum_versions'].append(minimum.group(1))
        if re.search(r'\bcmd LC_RPATH\b', block):
            path = re.search(r'^\s*path (.+?) \(offset \d+\)', block, re.M)
            if path:
                result['rpaths'].append(path.group(1))
    return result


def dependencies(content):
    return [line.strip().split(' (compatibility version ', 1)[0]
            for line in content.splitlines() if line.startswith('\t') and ' (compatibility version ' in line]


def expand_loader(value, loader, executable):
    return value.replace('@loader_path', str(loader.parent)).replace('@executable_path', str(executable.parent))


def command(args, log=None):
    completed = subprocess.run(args, capture_output=True, text=True, timeout=90)
    if log:
        Path(log).write_text('$ ' + ' '.join(args) + '\n' + completed.stdout + completed.stderr)
    return completed


def audit_binary(binary, deployment):
    binary = Path(binary).resolve()
    queue = [(binary, [])]
    seen = set()
    records, unavailable = [], []
    while queue:
        path, inherited = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        load = command(['otool', '-l', str(path)])
        linked = command(['otool', '-L', str(path)])
        if load.returncode or linked.returncode:
            unavailable.append({'path': str(path), 'reason': 'otool failed'})
            continue
        info = parse_load_commands(load.stdout)
        rpaths = [expand_loader(p, path, binary) for p in info['rpaths']] + inherited
        record = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'minimum_versions': info['minimum_versions'], 'rpaths': info['rpaths'],
                  'exceeds_target': any(version(n) > version(deployment) for n in info['minimum_versions'])}
        if not info['minimum_versions']:
            record['minimum_version_unverified'] = True
        records.append(record)
        for name in dependencies(linked.stdout):
            # macOS ships these through the dyld shared cache. No file means
            # "not inspected", never a fabricated old-system compatibility.
            if name.startswith(('/usr/lib/', '/System/Library/')):
                continue
            if name.startswith('@rpath/'):
                candidates = [Path(p) / name[len('@rpath/'):] for p in rpaths]
            else:
                candidates = [Path(expand_loader(name, path, binary))]
            resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if resolved:
                queue.append((resolved, rpaths))
            else:
                unavailable.append({'loader': str(path), 'dependency': name, 'reason': 'unresolved non-system library'})
    violations = [record for record in records if record['exceeds_target']]
    unknown = [record for record in records if record.get('minimum_version_unverified')]
    return {'binary': str(binary), 'requested_deployment': deployment, 'libraries': records,
            'violations': violations, 'unverified': unavailable + unknown,
            'non_system_dependency_minimum_gate': not violations and not unavailable and not unknown,
            'system_shared_cache_audited': False, 'old_os_runtime_tested': False}


def sdk_inventory(extra=()):
    paths = [Path(p) for p in extra]
    for root in ('/Library/Developer/CommandLineTools/SDKs',
                 '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs',
                 '/Applications/Swift Playground.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs'):
        paths.extend(Path(root).glob('MacOSX*.sdk'))
    records = []
    seen = set()
    for path in paths:
        if not (path / 'SDKSettings.json').is_file():
            continue
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        settings = json.loads((path / 'SDKSettings.json').read_text())
        cpp = path / 'usr/include/c++/v1'
        if not (cpp / 'cstdio').is_file() and str(path).startswith('/Library/Developer/CommandLineTools/'):
            cpp = Path('/Library/Developer/CommandLineTools/usr/include/c++/v1')
        records.append({'path': str(path), 'version': settings.get('Version', ''),
                        'display_name': settings.get('DisplayName', ''),
                        'cxx_headers': str(cpp) if (cpp / 'cstdio').is_file() else None})
    return sorted(records, key=lambda record: version(record['version']))


def compile_sdk(record, source, output, deployment, openblas):
    build = output / ('sdk-' + record['version'])
    # SDKs with the same version can come from different tool installations.
    if build.exists():
        build = output / (build.name + '-' + hashlib.sha256(record['path'].encode()).hexdigest()[:8])
    build.mkdir()
    result = dict(record, requested_deployment=deployment, build_directory=str(build),
                  old_os_runtime_tested=False, gpu_runtime_tested=False)
    if version(record['version']) < (11, 0):
        result.update(status='not-supported', reason='Apple Silicon source builds require SDK11 or newer; deployment remains macOS13+')
        return result
    if not record['cxx_headers']:
        result.update(status='unavailable', reason='installed SDK has no usable C++ standard headers')
        return result
    base = ['xcrun', 'clang++', '-std=c++17', '-fobjc-arc', '-fno-fast-math', '-ffp-contract=off',
            '-isysroot', record['path'], '-isystem', record['cxx_headers'],
            '-mmacosx-version-min=' + deployment, '-Werror=unguarded-availability-new',
            '-I' + str(source / 'Sources'), '-I' + str(Path(openblas) / 'include')]
    inputs = sorted((source / 'Sources').glob('*.mm'))
    syntax = command(base + ['-fsyntax-only'] + [str(p) for p in inputs], build / 'syntax.log')
    result['syntax_returncode'] = syntax.returncode
    if syntax.returncode:
        result.update(status='failed', reason='actual SDK source syntax check failed')
        return result
    # The backend may gain separate guarded owner/encoder implementation
    # translation units. Link all library sources, excluding CLI/BLAS wrappers.
    library_sources = [path for path in inputs if path.name not in
                       ('main.mm', 'metal_runner.mm', 'metal_device_probe.mm', 'vasp_metal_abi.mm')]
    targets = [('metal-device-probe', source / 'Sources/metal_device_probe.mm'),
               ('metal-backend-tests', source / 'Tests/metal_backend_tests.cpp')]
    if (source / 'Tests/compile_policy_tests.mm').exists():
        targets.append(('compile-policy-tests', source / 'Tests/compile_policy_tests.mm'))
    for name, test_source in targets:
        linked = command(base + ['-UNDEBUG'] + [str(path) for path in library_sources] + [str(test_source),
                                '-framework', 'Foundation', '-framework', 'Metal', '-o', str(build / name)],
                         build / (name + '-link.log'))
        result[name + '_link_returncode'] = linked.returncode
        if linked.returncode:
            result.update(status='failed', reason='actual SDK object/framework linking failed')
            return result
        result[name + '_macho'] = audit_binary(build / name, deployment)
        if name == 'compile-policy-tests':
            tested = command([str(build / name)], build / (name + '-run.log'))
            result[name + '_cpu_test_returncode'] = tested.returncode
            if tested.returncode:
                result.update(status='failed', reason='CPU-only compile policy regression failed')
                return result
    result.update(status='source-syntax-and-link-passed',
                  baseline_msl='2.4' if version(record['version']) >= (12, 0) else '2.3')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New evidence directory')
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--deployment-target', default='13.0')
    parser.add_argument('--openblas-root', default=os.environ.get('OPENBLAS_ROOT', '/opt/homebrew/opt/openblas'))
    parser.add_argument('--sdk', action='append', default=[])
    parser.add_argument('--binary', action='append', default=[])
    parser.add_argument('--skip-compile', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite an existing evidence directory')
    args.output.mkdir(parents=True)
    # Compile a single immutable adapter snapshot, even if another agent edits
    # the active source during this audit. It contains no licensed VASP input.
    snapshot = args.output.resolve() / 'source-snapshot'
    for directory in ('Sources', 'Tests'):
        shutil.copytree(args.project / directory, snapshot / directory)
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in snapshot.rglob('*') if path.is_file()}
    report = {'schema': 1, 'host': platform.platform(), 'architecture': platform.machine(),
              'deployment_target': args.deployment_target, 'source_sha256': hashes,
              'compiler': command(['xcrun', 'clang++', '--version']).stdout,
              'old_os_runtime_tested': False, 'other_m_series_tested': False,
              'installed_sdks': sdk_inventory(args.sdk)}
    report['compile_matrix'] = ([] if args.skip_compile else
        [compile_sdk(record, snapshot, args.output.resolve(), args.deployment_target, args.openblas_root)
         for record in report['installed_sdks']])
    report['binary_deployment_audits'] = [audit_binary(binary, args.deployment_target) for binary in args.binary]
    (args.output / 'compatibility.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('compile_matrix', 'binary_deployment_audits')}, indent=2))
    # Non-system dependency violations are evidence, not an audit crash.
    return 1 if any(record['status'] == 'failed' for record in report['compile_matrix']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
