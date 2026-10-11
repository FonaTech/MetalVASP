#!/usr/bin/env python3
"""Regression of ABI rejection rules; these are not GPU/VASP evidence."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('abi_audit', Path(__file__).resolve().parents[1] / 'Scripts' / 'audit-integer-abi.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class IntegerAbiAuditTests(unittest.TestCase):
    def test_lp64(self):
        self.assertEqual(audit.requested_abi_errors({'BLA_SIZEOF_INTEGER': '4', 'METAL_VASP_INTEGER_ABI': 'LP64'},
                                                 '# -fdefault-integer-8\nFFLAGS=-O3 -fdefault-real-8'), [])

    def test_default_integer_widening_flags(self):
        for flag in ('-fdefault-integer-8', '-finteger-4-integer-8', '-i8', '-integer-size 64', '-integer-size=64', '-qintsize=8'):
            with self.subTest(flag=flag):
                self.assertTrue(audit.requested_abi_errors({'FFLAGS': '-O3 ' + flag}))
                self.assertTrue(audit.requested_abi_errors({}, 'FFLAGS = ' + flag))

    def test_ilp64_library_requests(self):
        for environment in ({'METAL_VASP_INTEGER_ABI': 'ILP64'}, {'SCALAPACK_INTEGER_ABI': 'ILP64'},
                            {'BLAS_INTEGER_SIZE': '8'}, {'BLA_SIZEOF_INTEGER': '8'},
                            {'MKL_INTERFACE_LAYER': 'ILP64,GNU'}, {'OPENBLAS_USE64BITINT': '1'},
                            {'CPP_OPTIONS': '-DMKL_ILP64'}, {'MAKEFLAGS': 'INTERFACE64=1'}):
            with self.subTest(environment=environment):
                self.assertTrue(audit.requested_abi_errors(environment))

    def test_runtime_width(self):
        audit.require_lp64_config('OpenBLAS 0.3.32 DYNAMIC_ARCH NO_AFFINITY VORTEX MAX_THREADS=64')
        with self.assertRaises(audit.AbiError):
            audit.require_lp64_config('OpenBLAS 0.3.32 USE64BITINT DYNAMIC_ARCH')
        with self.assertRaises(audit.AbiError):
            audit.require_lp64_config('unknown runtime')


if __name__ == '__main__':
    unittest.main()
