#!/usr/bin/env python3
"""Resolve local LP64 VASP dependencies without machine-specific usernames."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

_abi_spec = importlib.util.spec_from_file_location('metal_vasp_integer_audit',
    Path(__file__).with_name('audit-integer-abi.py'))
_abi = importlib.util.module_from_spec(_abi_spec)
_abi_spec.loader.exec_module(_abi)


def prefixes(package):
    native = ('/opt/homebrew', '/usr/local') if platform.machine() == 'arm64' else ('/usr/local', '/opt/homebrew')
    yield from (Path(base) / 'opt' / package for base in native)
    brew = shutil.which('brew')
    if brew:
        result = subprocess.run([brew, '--prefix', package], capture_output=True, text=True)
        if result.returncode == 0:
            yield Path(result.stdout.strip())


def resolve(variable, package, headers, library, extra=()):
    explicit = os.environ.get(variable)
    candidates = [Path(explicit).expanduser()] if explicit else [*extra, *prefixes(package)]
    for candidate in candidates:
        # Make expands whitespace and '$' in dependency paths. Reject these
        # explicitly rather than emit a build file that silently links elsewhere.
        path = str(candidate.absolute())
        if any(c.isspace() or c in '$#' for c in path):
            if explicit:
                raise ValueError(f'{variable} contains unsupported Make path characters: {path}')
            continue
        if all((candidate / 'include' / header).is_file() for header in headers) and any(
                path.is_file() for path in (candidate / 'lib').glob(f'lib{library}.*')):
            return path
    source = f'explicit {variable}={explicit}' if explicit else 'local install discovery'
    raise ValueError(f'cannot resolve {variable} from {source}; set {variable} to an installation with include/lib')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--makefile', type=Path, help='Also reject ILP64 flags in this makefile.include')
    args = parser.parse_args()
    vasp_roots = []
    if os.environ.get('VASP_HOME'):
        vasp_roots.append(Path(os.environ['VASP_HOME']).expanduser())
    vasp_roots.append(Path.home() / '.local' / 'vasp' / '6.4.1')
    vasp_roots.extend((Path.home() / '.local' / 'vasp').glob('*/'))
    required = ('python3', 'make', 'clang++', 'gfortran', 'mpif90')
    tools = {tool: shutil.which(tool) for tool in required}
    missing = [tool for tool, path in tools.items() if path is None]
    if missing:
        raise ValueError('missing build tools: ' + ', '.join(missing))
    abi_errors = _abi.requested_abi_errors(os.environ, args.makefile.read_text() if args.makefile else '')
    if abi_errors:
        raise ValueError('; '.join(abi_errors))
    runtime = os.environ.get('CLANG_RUNTIME_FILE')
    if not runtime:
        result = subprocess.run([tools['clang++'], '-print-file-name=libclang_rt.osx.a'], capture_output=True, text=True)
        if result.returncode:
            raise ValueError('clang++ could not locate the macOS availability runtime')
        runtime = result.stdout.strip()
    runtime_path = Path(runtime).expanduser().absolute()
    if not runtime_path.is_file():
        raise ValueError(f'CLANG_RUNTIME_FILE is not an existing libclang_rt.osx.a: {runtime_path}')
    # The Make variable is quoted when passed to the linker, so spaces work.
    # Avoid characters with separate Make/shell expansion semantics.
    if any(c in str(runtime_path) for c in '$#"`\n\r'):
        raise ValueError(f'CLANG_RUNTIME_FILE contains unsupported Make/shell characters: {runtime_path}')
    result = {
        'OPENBLAS_ROOT': resolve('OPENBLAS_ROOT', 'openblas', ('cblas.h',), 'openblas'),
        'FFTW_ROOT': resolve('FFTW_ROOT', 'fftw', ('fftw3.h',), 'fftw3'),
        'SCALAPACK_ROOT': resolve('SCALAPACK_ROOT', 'scalapack', (), 'scalapack', vasp_roots),
        'CLANG_RUNTIME_FILE': str(runtime_path),
        'tools': tools,
        'platform': platform.platform(),
        'architecture': platform.machine(),
        'integer_abi': 'LP64',
        'ilp64_supported': False,
    }
    result['integer_abi_checks'] = {
        'openblas': _abi.audit_openblas(result['OPENBLAS_ROOT'], tools['clang++']),
        'fortran': _abi.audit_fortran(os.environ.get('FC', tools['mpif90'])),
        'scalapack': 'ABI width requires library build provenance and MPI integration checks',
    }
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        print(f'dependency preflight: {error}', file=sys.stderr)
        sys.exit(2)
