#!/usr/bin/env python3
"""Regression tests for the audit tool using explicitly synthetic outputs."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

script = Path(__file__).resolve().parents[1] / 'Scripts' / 'compare-vasp.py'
spec = importlib.util.spec_from_file_location('compare_vasp', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(energy_ev=-1., scf_final_energy_ev=-1.,
                              forces=[[0., 0., 0.]], stress=[[0., 0., 0.] for _ in range(3)],
                              iterations=4, ionic_steps=1, atom_count=1, converged=True,
                              completed=True, hermitian_warnings=0, stats={})
        self.candidate = copy.deepcopy(self.reference)
        self.candidate['stats'] = dict(gpu_device=1, gpu_eligible=2, gpu_success=2)

    def compare(self):
        return module.compare(self.reference, self.candidate)

    def test_valid_small_case_needs_no_eigensolver_or_fft(self):
        result = self.compare()
        self.assertTrue(result['numerical_gate'])
        self.assertTrue(result['gpu_execution_gate'])
        self.assertTrue(result['gpu_all_eligible_completed_gate'])

    def test_empty_arrays_cannot_pass(self):
        for data in (self.reference, self.candidate):
            data['forces'] = []
            data['stress'] = []
        self.assertFalse(self.compare()['numerical_gate'])

    def test_nonfinite_energy_cannot_pass(self):
        self.candidate['energy_ev'] = math.nan
        result = self.compare()
        self.assertFalse(result['numerical_gate'])
        self.assertIsNone(result['delta_energy_ev'])

    def test_atom_count_mismatch_cannot_pass(self):
        self.candidate['atom_count'] = 2
        self.assertFalse(self.compare()['numerical_gate'])

    def test_scalar_and_batch_fft_share_adaptive_counter(self):
        self.candidate['stats'].update(fft3d_calls=4, fft3d_success=4,
                                       fft3d_batch_calls=3, fft3d_batch_success=1,
                                       adaptive_cpu_fft=2)
        self.assertTrue(self.compare()['gpu_all_eligible_completed_gate'])

    def test_fft_failure_is_not_adaptive_routing(self):
        self.candidate['stats'].update(fft3d_calls=4, fft3d_success=3, adaptive_cpu_fft=0)
        result = self.compare()
        self.assertFalse(result['gpu_all_eligible_completed_gate'])
        self.assertEqual(result['gpu_rejected_or_cpu_fallbacks']['fft'], 1)

    def test_negative_and_fractional_counters_fail_gpu_gate(self):
        for value in (-1, 0.5, True):
            self.candidate['stats']['gpu_success'] = value
            self.assertFalse(self.compare()['gpu_execution_gate'])

    def test_explicit_operation_requirement(self):
        result = module.compare(self.reference, self.candidate, required_gpu_ops=('fft',))
        self.assertFalse(result['gpu_execution_gate'])
        self.assertEqual(result['missing_required_gpu_ops'], ['fft'])

    def test_full_orth_fallback_is_audited(self):
        self.candidate['stats'].update(full_orth_calls=2, full_orth_success=1)
        self.assertEqual(self.compare()['gpu_rejected_or_cpu_fallbacks']['full_orthogonalization'], 1)

    def test_final_iteration_convergence_and_xml_energy(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            p.joinpath('vasprun.xml').write_text('''<modeling><atominfo><atoms>1</atoms></atominfo><calculation>
              <scstep><energy><i name="e_fr_energy">-1.123456789D0</i></energy></scstep>
              <energy><i name="e_fr_energy">-1.123456789D0</i></energy>
              <varray name="forces"><v>0 0 0</v></varray>
              <varray name="stress"><v>0 0 0</v><v>0 0 0</v><v>0 0 0</v></varray>
              </calculation></modeling>''')
            p.joinpath('OUTCAR').write_text('Iteration 1(1)\naborting loop because EDIFF is reached\nIteration 2(1)\nGeneral timing and accounting\n')
            p.joinpath('cpu.log').write_text('DAV: 1 -1.123E0\n')
            data = module.read(directory)
            self.assertEqual(data['scf_final_energy_ev'], -1.123456789)
            self.assertFalse(data['converged'])
            self.assertTrue(data['completed'])


if __name__ == '__main__':
    unittest.main()
