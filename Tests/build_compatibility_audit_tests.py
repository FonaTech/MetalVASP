#!/usr/bin/env python3
"""Mach-O parsing regression only; never substitutes for actual SDK builds."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('build_audit', Path(__file__).resolve().parents[1] / 'Scripts' / 'audit-build-compatibility.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class BuildCompatibilityAuditTests(unittest.TestCase):
    def test_equivalent_versions(self):
        self.assertEqual(audit.version('13.0'), audit.version('13.0.0'))

    def test_current_macho(self):
        record = audit.parse_load_commands('Load command 0\n cmd LC_BUILD_VERSION\n platform 1\n minos 26.0\n sdk 26.5\n'
                                           'Load command 1\n cmd LC_RPATH\n path @loader_path/../lib (offset 12)\n')
        self.assertEqual(record['minimum_versions'], ['26.0'])
        self.assertEqual(record['rpaths'], ['@loader_path/../lib'])

    def test_legacy_macho(self):
        record = audit.parse_load_commands('Load command 0\n cmd LC_VERSION_MIN_MACOSX\n version 13.0\n sdk 14.0\n')
        self.assertEqual(record['minimum_versions'], ['13.0'])

    def test_library_names_and_context(self):
        self.assertEqual(audit.dependencies('a:\n\t@rpath/libfoo.dylib (compatibility version 1.0.0, current version 1.0.0)\n'),
                         ['@rpath/libfoo.dylib'])
        self.assertEqual(audit.expand_loader('@loader_path/lib', Path('/a/bin/load'), Path('/b/bin/exe')), '/a/bin/lib')


if __name__ == '__main__':
    unittest.main()
