#!/usr/bin/env python3
"""Local-only diagnostics and transactional, source-fingerprint-gated builds.

SPDX-License-Identifier: MIT
No network access, package installation, or licensed-source uploads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
MAX_FILES = 100000
MAX_BYTES = 8 * 1024**3


class SetupError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(args, timeout=20):
    """Read-only probes; absence and failure stay explicit."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {'ok': p.returncode == 0, 'exit_code': p.returncode,
                'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()[:1500]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'ok': False, 'error': str(e)}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def environment():
    is_mac = platform.system() == 'Darwin'
    d = {'schema': 'metalvasp-diagnostics-v1', 'os': platform.system(),
         'architecture': platform.machine(), 'python': platform.python_version(),
         'cpu_count': os.cpu_count(), 'device': {}, 'tools': {},
         'metal_capabilities': 'Not probed yet; runtime checks actual MTLDevice families and limits.',
         'privacy': 'Local only. Serial numbers, UUIDs, user accounts and source contents are excluded.'}
    if is_mac:
        d['macos'] = run(['sw_vers', '-productVersion']).get('stdout')
        for key, name in [('model', 'hw.model'), ('chip', 'machdep.cpu.brand_string'),
                          ('memory_bytes', 'hw.memsize'), ('performance_cores', 'hw.perflevel0.physicalcpu')]:
            v = run(['sysctl', '-n', name])
            d['device'][key] = v.get('stdout') if v['ok'] else None
        levels = run(['sysctl', '-n', 'hw.nperflevels'])
        core_groups = []
        try:
            for level in range(min(int(levels.get('stdout', '0')), 8)):
                label = run(['sysctl', '-n', 'hw.perflevel' + str(level) + '.name'])
                cores = run(['sysctl', '-n', 'hw.perflevel' + str(level) + '.physicalcpu'])
                if label['ok'] and cores['ok']:
                    core_groups.append({'name': label['stdout'], 'physical_cores': int(cores['stdout'])})
            # Newer chips may expose separate Super and Performance groups.
            # Do not equate perflevel0 with all performance cores.
            perf = sum(g['physical_cores'] for g in core_groups
                       if 'performance' in g['name'].lower() or 'super' in g['name'].lower())
            if perf:
                d['device']['performance_cores'] = perf
        except (ValueError, TypeError):
            pass
        d['device']['core_groups'] = core_groups
        info = run(['system_profiler', 'SPDisplaysDataType', '-json'], timeout=30)
        try:
            # Never retain the raw hardware inventory, display serial or UUID.
            cards = json.loads(info.get('stdout', '{}')).get('SPDisplaysDataType', [])
            d['device']['gpus'] = [{k: c[k] for k in ['sppci_model', 'spdisplays_metal',
                                                       'spdisplays_vendor'] if k in c} for c in cards]
        except (ValueError, TypeError):
            d['device']['gpus'] = None
        d['sdk'] = run(['xcrun', '--sdk', 'macosx', '--show-sdk-path']).get('stdout')
        d['sdk_version'] = run(['xcrun', '--sdk', 'macosx', '--show-sdk-version']).get('stdout')
    for tool in ['make', 'cmake', 'clang++', 'gfortran', 'mpif90', 'python3', 'xcrun']:
        exe = shutil.which(tool)
        record = {'available': bool(exe)}
        if exe:
            v = run([exe, '--version'])
            record['version'] = v.get('stdout', '').splitlines()[:1]
            record['path'] = exe
        d['tools'][tool] = record
    for var, pkg in [('OPENBLAS_ROOT', 'openblas'), ('FFTW_ROOT', 'fftw'),
                     ('SCALAPACK_ROOT', 'scalapack')]:
        roots = [os.environ.get(var), '/opt/homebrew/opt/' + pkg, '/usr/local/opt/' + pkg]
        d[var] = next((str(Path(r).expanduser()) for r in roots if r and Path(r).expanduser().is_dir()), None)
    d['build_platform_supported'] = is_mac and platform.machine() == 'arm64'
    # A missing offline Metal compiler is not a runtime-source-compile failure.
    d['offline_metal'] = run(['xcrun', '--find', 'metal']) if is_mac else {'ok': False}
    missing = [t for t in ['make', 'clang++', 'gfortran', 'mpif90', 'cmake'] if not d['tools'][t]['available']]
    d['missing_tools'] = missing
    d['dependency_install_hint'] = 'brew install cmake gcc open-mpi openblas fftw scalapack (run yourself; verify MPI/Fortran and LP64 ABI).'
    d['recommended_jobs'] = choose_jobs(d)
    return d


def choose_jobs(d):
    cores = d.get('device', {}).get('performance_cores') or d.get('cpu_count') or 1
    try:
        memory = int(d.get('device', {}).get('memory_bytes') or 4 * 1024**3)
        return max(1, min(int(cores), max(1, memory // (2 * 1024**3)), 16))
    except (ValueError, TypeError):
        return 1


def member_path(name):
    # Backslash is rejected so the same rule also works on Windows archive tools.
    if '\\' in name or '\x00' in name:
        raise SetupError('Archive contains a forbidden path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or (p.parts and ':' in p.parts[0]):
        raise SetupError('Archive path escapes the extraction root')
    return Path(*p.parts)


def extract(source, dest):
    """Validate every entry before writing. Never follow links or special files."""
    source, dest = Path(source), Path(dest)
    if source.is_dir():
        files = list(source.rglob('*'))
        if len(files) > MAX_FILES:
            raise SetupError('Source tree exceeds file-count limit')
        if any(p.is_symlink() or (not p.is_file() and not p.is_dir()) for p in files):
            raise SetupError('Source tree contains a link or special file; use a clean official archive')
        if sum(p.stat().st_size for p in files if p.is_file()) > MAX_BYTES:
            raise SetupError('Source tree exceeds extraction budget')
        shutil.copytree(source, dest)
        return
    if not source.is_file():
        raise SetupError('VASP source path does not exist')
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            entries = z.infolist()
            total = 0
            names = set()
            for e in entries:
                key = str(member_path(e.filename))
                mode = (e.external_attr >> 16) & 0o170000
                if mode not in (0, 0o100000, 0o040000):
                    raise SetupError('ZIP link or special file rejected')
                if key in names:
                    raise SetupError('Duplicate archive member')
                names.add(key); total += e.file_size
            if len(entries) > MAX_FILES or total > MAX_BYTES:
                raise SetupError('Archive exceeds extraction budget')
            dest.mkdir()
            for e in entries:
                p = dest / member_path(e.filename)
                if e.is_dir():
                    p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(e) as f, p.open('wb') as g:
                        shutil.copyfileobj(f, g)
        return
    try:
        with tarfile.open(source, 'r:*') as t:
            entries = t.getmembers()
            total = 0; names = set()
            for e in entries:
                key = str(member_path(e.name))
                if not (e.isfile() or e.isdir()):
                    raise SetupError('TAR link or special file rejected')
                if key in names:
                    raise SetupError('Duplicate archive member')
                names.add(key); total += e.size
            if len(entries) > MAX_FILES or total > MAX_BYTES:
                raise SetupError('Archive exceeds extraction budget')
            dest.mkdir()
            for e in entries:
                p = dest / member_path(e.name)
                if e.isdir():
                    p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(e) as f, p.open('wb') as g:
                        shutil.copyfileobj(f, g)
                    p.chmod(e.mode & 0o777)
    except tarfile.TarError as e:
        raise SetupError('Unrecognized or damaged source archive') from e


def locate_source(path):
    matches = [p.parent.parent for p in Path(path).rglob('main.F') if p.parent.name == 'src']
    if len(matches) != 1:
        raise SetupError('Expected exactly one VASP src/main.F tree')
    return matches[0]


def source_version(root):
    candidates = []
    metadata = root / 'src/version.F'
    if metadata.is_file():
        text = metadata.read_text(errors='replace')
        parts = [re.findall(r'(?im)^\s*integer\s*,\s*parameter\s*::\s*' + n + r'\s*=\s*(\d+)\b', text)
                 for n in ['major', 'minor', 'patch']]
        if all(len(v) == 1 for v in parts):
            candidates.append('.'.join(v[0] for v in parts))
    for rel in ['src/version.F', 'src/main.F']:
        p = root / rel
        if p.is_file():
            text = p.read_text(errors='replace')
            candidates += re.findall(r'(?i)vasp[.\s]+(\d+\.\d+\.\d+)', text)
    unique = set(candidates)
    if len(unique) != 1:
        raise SetupError('Cannot identify one VASP version from source metadata; filename alone is insufficient')
    return unique.pop()


def reconstruct(base, rule):
    if digest(base) != rule['input_sha256']:
        raise SetupError('Source fingerprint mismatch: ' + rule['path'])
    result = bytearray()
    for op in rule['operations']:
        if set(op) == {'copy'}:
            at, size = op['copy']
            if at < 0 or size < 0 or at + size > len(base):
                raise SetupError('Invalid recipe range')
            result.extend(base[at:at+size])
        elif set(op) == {'insert'}:
            result.extend(op['insert'].encode('utf-8'))
        else:
            raise SetupError('Invalid recipe operation')
    if digest(result) != rule['output_sha256']:
        raise SetupError('Reconstructed output hash mismatch')
    return bytes(result)


def prepare_patches(root, allow_unvalidated=False):
    adapter = json.loads((PROJECT / 'adapters/vasp-6.4.1.json').read_text())
    version = source_version(root)
    if version != adapter['tested_vasp_version'] and not (allow_unvalidated and version.startswith('6.')):
        raise SetupError('VASP ' + version + ': inspection only. Native adapter is validated for 6.4.1; see docs/COMPATIBILITY.md.')
    outputs = []
    for rule in adapter['files']:
        source = root / member_path(rule.get('base_path', rule['path']))
        if not source.is_file():
            raise SetupError('Required source layout missing: ' + rule['path'])
        outputs.append((rule, reconstruct(source.read_bytes(), rule)))
    return version, outputs


def make_value(value):
    # Make expands variables, comments and newlines before the shell executes.
    # Dependency paths therefore have tighter rules than the source archive path.
    if not value or any(c.isspace() or c in "$#`\\\"';&|<>!()" for c in value):
        raise SetupError('Dependency/work/SDK path contains unsupported Make or shell characters: ' + value)
    return value


def parameterize(payload, dependencies, env, args):
    profile = args.profile
    cpu = '-march=armv8-a' if profile == 'portable' else '-march=native'
    flags = ("-DMETAL_VASP_DEFAULT_PROFILE='\"" + profile + "\"' "
             "-DMETAL_VASP_DEFAULT_LANGUAGE='\"auto\"' "
             '-DMETAL_VASP_FP32_FAST_MATH=0 -mmacosx-version-min=' + args.deployment_target)
    values = {n: make_value(dependencies[n]) for n in ['OPENBLAS_ROOT', 'SCALAPACK_ROOT', 'FFTW_ROOT', 'CLANG_RUNTIME_FILE']}
    values.update(SDKROOT=make_value(env['sdk']), VASP_TARGET_CPU=cpu,
                  METAL_VASP_CXX_FLAGS=flags, DEPLOYMENT=args.deployment_target)
    text = payload.decode()
    for name, value in values.items():
        text = text.replace('@' + name + '@', value)
    if re.search(r'@[A-Z_]+@', text):
        raise SetupError('Unresolved build parameter')
    return text.encode()


def command(args, cwd=None, env=None):
    print('+', ' '.join(str(x) for x in args), flush=True)
    p = subprocess.run([str(x) for x in args], cwd=cwd, env=env)
    if p.returncode:
        raise SetupError('Command failed with exit ' + str(p.returncode) + '; logs and private work copy retained')


def inspect_input(source, allow=False):
    if Path(source).is_dir():
        return inspect_tree(Path(source), source, allow)
    with tempfile.TemporaryDirectory(prefix='metalvasp-inspect-') as temp:
        staged = Path(temp) / 'input'
        extract(source, staged)
        root = locate_source(staged)
        return inspect_tree(root, source, allow)


def inspect_tree(root, source, allow=False):
    version = source_version(root)
    result = {'version': version, 'native_validated_version': '6.4.1',
              'source_fingerprint_gate': False, 'native_adapter_files': 16,
              'source_path': str(source), 'licensed_source_uploaded': False}
    try:
        prepare_patches(root, allow)
        result['source_fingerprint_gate'] = True
        result['status'] = 'validated_source_layout' if version == '6.4.1' else 'unvalidated_version_compatible_fingerprints'
    except SetupError as e:
        result['status'] = 'inspection_only'; result['reason'] = str(e)
    return result


def backend_build(args, env, work):
    flags = ['-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_Fortran_COMPILER=' + (env['tools']['gfortran'].get('path') or 'NOTFOUND'),
             '-DMETAL_VASP_PROFILE=' + args.profile,
             '-DCMAKE_OSX_DEPLOYMENT_TARGET=' + args.deployment_target,
             '-DMETAL_VASP_FAST_MATH=OFF']
    # Preserve custom LP64 installations in the backend ABI/lifecycle test targets.
    depfile = work / 'dependencies.json'
    if depfile.is_file():
        deps = json.loads(depfile.read_text())
        library = Path(deps['OPENBLAS_ROOT']) / 'lib/libopenblas.dylib'
        if not library.is_file():
            library = Path(deps['OPENBLAS_ROOT']) / 'lib/libopenblas.a'
        if not library.is_file():
            raise SetupError('OpenBLAS link artifact cannot be located')
        flags += ['-DOPENBLAS_INCLUDE_DIR=' + deps['OPENBLAS_ROOT'] + '/include',
                  '-DOPENBLAS_LIBRARY=' + str(library)]
    command(['cmake', '-S', PROJECT, '-B', work / 'backend', *flags])
    command(['cmake', '--build', work / 'backend', '--parallel', args.jobs])
    testenv = os.environ.copy(); testenv['METAL_REQUIRE_GPU'] = '1'
    command(['ctest', '--test-dir', work / 'backend', '--output-on-failure'], env=testenv)
    probe = run([str(work / 'backend/metal-device-probe')], timeout=30)
    if not probe['ok']:
        raise SetupError('Actual Metal device probe failed')
    write_json(work / 'metal-device-probe.json', probe)
    if args.offline:
        if not env['offline_metal']['ok']:
            raise SetupError('Offline Metal Toolchain missing; omit --offline to use runtime source compilation')
        # Portable 2.3 is an explicit baseline; auto uses compatible 3.0 here.
        # Runtime code may use newer guarded MSL; no device model guess creates 4.0 bytecode.
        lang = '2.3' if args.profile == 'portable' else '3.0'
        command([sys.executable, PROJECT / 'Scripts/build-metallib.py', '--project', PROJECT,
                 '--output', work / 'offline', '--language', lang, '--metal40', '0',
                 '--device-tuned', '0' if args.profile == 'portable' else '1',
                 '--deployment-target', args.deployment_target])


def build(args, env):
    if not env['build_platform_supported']:
        raise SetupError('Native build requires an arm64 Mac; diagnostics and installer tests work elsewhere')
    if env['missing_tools']:
        raise SetupError('Missing tools: ' + ', '.join(env['missing_tools']) + '. ' + env['dependency_install_hint'])
    work = Path(args.work).expanduser().absolute()
    make_value(str(work))
    if work.exists():
        raise SetupError('Choose a new work directory; existing builds are never overwritten')
    if args.source:
        source = Path(args.source).expanduser().resolve()
        if source.is_dir() and (source == work or source in work.parents):
            raise SetupError('Work directory must be outside the source tree')
    work.mkdir(parents=True)
    write_json(work / 'diagnostics.json', env)
    result = {'status': 'preparing', 'release': '0.15.0-rc2', 'profile': args.profile,
              'deployment_target': args.deployment_target, 'jobs': args.jobs,
              'wholeprogram_validated_here': False, 'source_uploaded': False,
              'scope': 'Local capability selection; global optimality is not claimed.'}
    write_json(work / 'build-status.json', result)
    try:
        if args.command == 'backend':
            backend_build(args, env, work)
        else:
            staged = work / 'licensed-input'
            extract(source, staged)
            root = locate_source(staged)
            version, outputs = prepare_patches(root, args.allow_unvalidated_version)
            # Resolve actual LP64/Fortran widths before native link. No manual shell interpolation.
            command([sys.executable, PROJECT / 'Scripts/resolve-dependencies.py',
                     '--output', work / 'dependencies.json'])
            deps = json.loads((work / 'dependencies.json').read_text())
            for rule, data in outputs:
                if rule.get('parameters'):
                    data = parameterize(data, deps, env, args)
                (root / rule['path']).write_bytes(data)
            metal = root / 'metal'
            if metal.exists():
                raise SetupError('Input already contains a metal/ tree; use an unmodified official source')
            shutil.copytree(PROJECT / 'Sources', metal)
            shutil.copy2(metal / 'vasp_metal_abi.mm', metal / 'metal_abi.mm')
            # Never mix user-provided object modules or original executables into this build.
            for name in ['build', 'bin']:
                p = root / name
                if p.exists():
                    p.rename(root / ('input-' + name))
            (root / 'bin').mkdir()
            for f in (root / 'testsuite').rglob('*') if (root / 'testsuite').exists() else []:
                if f.is_file() and f.name == 'runtest':
                    s = f.read_text()
                    for original, new in [('vasp_std', 'vasp_mps'), ('vasp_gam', 'vasp_mps_gam'), ('vasp_ncl', 'vasp_mps_ncl')]:
                        s = s.replace('/bin/' + original, '/bin/' + new)
                    f.write_text(s)
            command([sys.executable, PROJECT / 'Scripts/audit-integer-abi.py', '--openblas-root',
                     deps['OPENBLAS_ROOT'], '--makefile', root / 'makefile.include', '--output', work / 'integer-abi.json'])
            result.update(vasp_version=version, source_fingerprints_checked=True,
                          adapter_validation='reviewed_6.4.1' if version == '6.4.1' else 'experimental_unvalidated_version',
                          licensed_work_copy=str(root), backend_source_sha256=json.loads((PROJECT / 'adapters/backend-provenance.json').read_text())['backend_sha256'])
            write_json(work / 'build-status.json', result)
            if not args.configure_only:
                backend_build(args, env, work)
                names = {'std': 'vasp_mps', 'gam': 'vasp_mps_gam', 'ncl': 'vasp_mps_ncl'}
                for target in args.targets:
                    command(['make', '-C', root, 'DEPS=1', '-j' + str(args.jobs), target])
                    if not (root / 'bin' / names[target]).is_file():
                        raise SetupError('Build did not produce expected ' + names[target])
                result['binaries'] = {t: str(root / 'bin' / names[t]) for t in args.targets}
                # Dependency closure and deployment compatibility are separate from compilation.
                import importlib.util
                spec = importlib.util.spec_from_file_location('compat', PROJECT / 'Scripts/audit-build-compatibility.py')
                audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
                result['deployment_audits'] = {t: audit.audit_binary(Path(p), args.deployment_target) for t, p in result['binaries'].items()}
                result['dependency_minimum_gate'] = all(r['non_system_dependency_minimum_gate'] for r in result['deployment_audits'].values())
                if not result['dependency_minimum_gate']:
                    result['deployment_notice'] = 'Compiled; at least one dependency exceeds target or is unresolved. Do not redistribute as compatible with that target.'
                launcher = work / 'run-metal.sh'
                import shlex
                p = result['binaries'].get('std') or next(iter(result['binaries'].values()))
                text = '#!/bin/sh\nset -eu\nexport VASP_METAL=1\nexport VASP_METAL_PROFILE=' + shlex.quote(args.profile) + '\n'
                text += 'export VASP_METAL_WAVE_OWNER=1 VASP_METAL_DAVIDSON_OWNER=1 VASP_METAL_NONLOCAL=1\n'
                if args.offline:
                    text += 'export VASP_METAL_LIBRARY=' + shlex.quote(str(work / 'offline/backend.metallib')) + '\n'
                text += 'exec ' + shlex.quote(p) + ' "$@"\n'
                launcher.write_text(text); launcher.chmod(0o755)
                result['launcher'] = str(launcher)
        result['status'] = 'configured_not_compiled' if args.configure_only else 'built_requires_local_scientific_validation'
        write_json(work / 'build-status.json', result)
        print(json.dumps(result, indent=2))
    except Exception as e:
        result.update(status='failed', reason=str(e))
        write_json(work / 'build-status.json', result)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', nargs='?', choices=['doctor', 'inspect', 'plan', 'build', 'backend'], default=None)
    p.add_argument('--source', help='Local licensed VASP archive or clean source directory; never uploaded')
    p.add_argument('--work', default=str(Path.home() / 'metalvasp-build'))
    p.add_argument('--profile', choices=['auto', 'portable', 'advanced'], default='auto')
    p.add_argument('--jobs', type=int)
    p.add_argument('--deployment-target', default='13.0')
    p.add_argument('--targets', nargs='+', choices=['std', 'gam', 'ncl'], default=['std', 'gam', 'ncl'])
    p.add_argument('--configure-only', action='store_true')
    p.add_argument('--allow-unvalidated-version', action='store_true', help='Allow other VASP 6.x metadata ONLY when all source fingerprints match; results remain unvalidated')
    p.add_argument('--offline', action='store_true', help='Optional matched offline metallib; runtime compilation is the default')
    p.add_argument('--json', type=Path)
    args = p.parse_args()
    if not re.fullmatch(r'\d+(?:\.\d+){0,2}', args.deployment_target) or int(args.deployment_target.split('.')[0]) < 13:
        p.error('Deployment target must be macOS 13 or newer')
    if args.command is None:
        if not sys.stdin.isatty():
            args.command = 'doctor'
        else:
            args.command = 'build'
            print('MetalVASP uses your licensed source locally. No VASP download or upload occurs.\n金属加速构建仅使用本机合法授权的 VASP 源码，不上传源码。')
            args.source = input('Local VASP source/archive path / 本机源码或安装包路径: ').strip()
            args.work = input('New work directory / 新工作目录 [~/metalvasp-build]: ').strip() or args.work
    if args.command in ['inspect', 'plan', 'build'] and not args.source:
        p.error('--source is required')
    if args.jobs is not None and (args.jobs < 1 or args.jobs > 256):
        p.error('--jobs must be 1..256')
    if args.command == 'inspect':
        result = inspect_input(Path(args.source).expanduser().resolve(), args.allow_unvalidated_version)
    else:
        env = environment()
        args.jobs = args.jobs or choose_jobs(env)
        if args.command in ['build', 'backend']:
            build(args, env); return
        result = env
        if args.command == 'plan':
            result['source'] = inspect_input(Path(args.source).expanduser().resolve(), args.allow_unvalidated_version)
            result['plan'] = {'profile': args.profile, 'jobs': args.jobs, 'deployment_target': args.deployment_target,
                              'targets': args.targets, 'integer_abi': 'LP64', 'fp64_math': 'DS safe/precise; no fast-math',
                              'advanced_selection': 'Actual SDK/OS/MTLDevice capability guards; model name is informational',
                              'compile_started': False, 'new_nonlocal_enabled_in_launcher': True}
    if args.json:
        write_json(args.json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (SetupError, OSError) as e:
        print('MetalVASP: ' + str(e), file=sys.stderr)
        sys.exit(2)
