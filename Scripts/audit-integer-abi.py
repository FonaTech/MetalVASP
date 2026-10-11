#!/usr/bin/env python3
"""Fail closed on known ILP64 requests; verify the selected OpenBLAS LP64 ABI.

The adapter's interposed BLAS/LAPACK symbols use 32-bit Fortran INTEGER.
An ILP64 library must not be substituted under those identical symbol names.
ScaLAPACK has no portable integer-width query: it requires build provenance
and integration checks in addition to this OpenBLAS/header/compiler audit.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tempfile


class AbiError(ValueError):
    pass


FLAG_PATTERN = re.compile(
    r"(?i)(?:-fdefault-integer-8\b|-finteger-4-integer-8\b|-i8\b|"
    r"[-/]integer[-_]size(?:\s+|[=:])(?:64|8)\b|-qintsize(?:\s+|=)8\b|"
    r"(?:-D|\b)MKL_ILP64\b|(?:-D|\b)(?:OPENBLAS_)?USE64BITINT(?:\s*[=:]\s*1)?\b|"
    r"\bINTERFACE64\s*[=:]\s*1\b|\bBLA_SIZEOF_INTEGER\s*[=:]\s*8\b)")
ENV_FLAGS = ('FFLAGS', 'FCFLAGS', 'FCLFLAGS', 'CPP_OPTIONS', 'CPPFLAGS',
             'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'MAKEFLAGS', 'FC', 'FCL')


def requested_abi_errors(environment, makefile_text=''):
    errors = []
    for key in ENV_FLAGS:
        value = environment.get(key, '')
        if FLAG_PATTERN.search(value):
            errors.append(f'{key} requests an ILP64/64-bit INTEGER ABI')
    # Inspect logical Make lines, excluding comments. Never evaluate arbitrary
    # Make shell expressions or the caller's configuration while auditing it.
    active = '\n'.join(line.split('#', 1)[0] for line in makefile_text.splitlines())
    if FLAG_PATTERN.search(active):
        errors.append('makefile.include requests an ILP64/64-bit INTEGER ABI')
    for key in ('METAL_VASP_INTEGER_ABI', 'BLAS_INTEGER_ABI', 'SCALAPACK_INTEGER_ABI'):
        value = environment.get(key, '').strip().upper()
        if value and value != 'LP64':
            errors.append(f'{key}={value}: only LP64 is implemented')
    for key in ('BLA_SIZEOF_INTEGER', 'BLAS_INTEGER_SIZE'):
        value = environment.get(key, '').strip()
        if value and value not in ('4', '32'):
            errors.append(f'{key}={value}: the adapter requires 4-byte INTEGER')
    if 'ILP64' in environment.get('MKL_INTERFACE_LAYER', '').upper():
        errors.append('MKL_INTERFACE_LAYER requests ILP64')
    for key in ('OPENBLAS_USE64BITINT', 'USE64BITINT', 'INTERFACE64'):
        value = environment.get(key, '').strip().upper()
        if value and value not in ('0', 'OFF', 'NO', 'FALSE'):
            errors.append(f'{key} requests an ILP64 library')
    return errors


def require_lp64_config(config):
    if not config.startswith('OpenBLAS '):
        raise AbiError('unrecognized OpenBLAS runtime configuration; integer width is unverified')
    if 'USE64BITINT' in config.upper() or 'INTERFACE64' in config.upper():
        raise AbiError('selected OpenBLAS runtime reports 64-bit INTEGER; LP64 adapter cannot link it')


def run(command):
    environment = os.environ.copy()
    if platform.system() == 'Darwin' and not environment.get('SDKROOT'):
        sdk = subprocess.run(['xcrun', '--sdk', 'macosx', '--show-sdk-path'],
                             capture_output=True, text=True, timeout=20)
        if sdk.returncode == 0:
            environment['SDKROOT'] = sdk.stdout.strip()
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment)
    if result.returncode:
        raise AbiError(f'ABI probe failed ({command[0]}): {result.stderr.strip() or result.stdout.strip()}')
    return result.stdout.strip()


def audit_openblas(root, clang=None):
    root = Path(root).resolve()
    clang = clang or shutil.which('clang++')
    if not clang:
        raise AbiError('clang++ is required to verify the OpenBLAS header integer width')
    libraries = sorted((root / 'lib').glob('libopenblas*.dylib'))
    if not libraries:
        libraries = sorted((root / 'lib').glob('libopenblas*.so*'))
    if not libraries:
        raise AbiError('OpenBLAS audit requires a dynamic library with openblas_get_config')
    exact = next((p for p in libraries if p.name in ('libopenblas.dylib', 'libopenblas.so')), libraries[0])
    # The configuration query takes no integer arguments, so querying an ILP64
    # library is safe. Do not call numerical routines before this width gate.
    library = ctypes.CDLL(str(exact))
    try:
        library.openblas_get_config.restype = ctypes.c_char_p
        config_bytes = library.openblas_get_config()
    except AttributeError as error:
        raise AbiError('OpenBLAS runtime lacks openblas_get_config; integer width is unverified') from error
    if not config_bytes:
        raise AbiError('OpenBLAS runtime returned no ABI configuration')
    config = config_bytes.decode('utf-8', errors='replace')
    require_lp64_config(config)
    with tempfile.TemporaryDirectory(prefix='metal-vasp-abi-') as temp:
        source = Path(temp) / 'header.cpp'
        binary = Path(temp) / 'header-probe'
        source.write_text('#include <cblas.h>\n#include <cstdio>\n'
                          'static_assert(sizeof(blasint)==4,"MetalVASP requires LP64 OpenBLAS headers");\n'
                          'int main(){std::printf("%zu",sizeof(blasint));}\n')
        run([clang, '-std=c++17', '-I' + str(root / 'include'), str(source), '-o', str(binary)])
        header_bytes = int(run([str(binary)]))
    return {'library': str(exact.resolve()), 'runtime_config': config,
            'header_integer_bytes': header_bytes, 'runtime_integer_bytes': 4, 'passed': True}


def audit_fortran(compiler=None):
    command = shlex.split(compiler or os.environ.get('FC', 'gfortran'))
    if not command or not shutil.which(command[0]):
        raise AbiError('a working Fortran compiler is required for INTEGER/c_int ABI verification')
    with tempfile.TemporaryDirectory(prefix='metal-vasp-fortran-abi-') as temp:
        source = Path(temp) / 'integer.f90'
        binary = Path(temp) / 'integer-probe'
        source.write_text('program integer_probe\n use iso_c_binding\n implicit none\n'
                          ' integer :: n\n integer(c_int) :: c\n'
                          ' print *,storage_size(n)/8,storage_size(c)/8\nend program\n')
        run(command + [str(source), '-o', str(binary)])
        widths = [int(n) for n in run([str(binary)]).split()]
        if widths != [4, 4]:
            raise AbiError(f'Fortran default INTEGER/c_int widths {widths}; LP64 requires [4, 4]')
    return {'compiler': command, 'default_integer_bytes': widths[0], 'c_int_bytes': widths[1], 'passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openblas-root', required=True)
    parser.add_argument('--makefile', type=Path)
    parser.add_argument('--fortran-compiler')
    parser.add_argument('--clang')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {'schema': 1, 'implemented_integer_abi': 'LP64', 'ilp64_supported': False,
              'scalapack_integer_abi': 'requires library build provenance and MPI integration checks',
              'passed': False}
    try:
        errors = requested_abi_errors(os.environ, args.makefile.read_text() if args.makefile else '')
        if errors:
            raise AbiError('; '.join(errors))
        result['openblas'] = audit_openblas(args.openblas_root, args.clang)
        result['fortran'] = audit_fortran(args.fortran_compiler)
        result['passed'] = True
    except (AbiError, OSError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
    content = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(content)
    print(content, end='')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
