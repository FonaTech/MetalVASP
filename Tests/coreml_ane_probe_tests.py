#!/usr/bin/env python3
"""Check the placement evidence gate without importing Core ML or emulating it."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('ane_probe', Path(__file__).resolve().parents[1] / 'Scripts' / 'coreml-ane-probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PlacementGateTests(unittest.TestCase):
    def test_namespaced_arithmetic(self):
        for operator in ('linear', 'ios16.linear', 'ios18.conv', 'ios16.matmul'):
            self.assertTrue(probe.is_ane_arithmetic({'operator': operator, 'preferred': {'class': 'MLNeuralEngineComputeDevice'}}))

    def test_supported_alone_is_insufficient(self):
        self.assertFalse(probe.is_ane_arithmetic({'operator': 'ios16.linear', 'preferred': {'class': 'MLCPUComputeDevice'},
                                                'supported': [{'class': 'MLNeuralEngineComputeDevice'}]}))

    def test_cast_is_insufficient(self):
        self.assertFalse(probe.is_ane_arithmetic({'operator': 'ios16.cast', 'preferred': {'class': 'MLNeuralEngineComputeDevice'}}))


if __name__ == '__main__':
    unittest.main()
