#!/usr/bin/env python3
"""CPU-only parser fixtures; these are not runtime ANE execution evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'Scripts' / 'audit-coreml-trace.py'
SPEC = importlib.util.spec_from_file_location('coreml_trace_audit', SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TraceAuditTests(unittest.TestCase):
    identifier = 'MetalVASP.ANEProbe.Conv512.FP16.Native.PID23'
    target = {'prediction_intervals': [{'start_unix_seconds': 100.2, 'end_unix_seconds': 100.4}]}

    def row(self, **changes):
        row = {'start-time-ns': 250000000, 'duration-ns': 100000000,
               'ane-event-name': 'Apple Neural Engine', 'gpu-state': 'Active Active',
               'formatted-label': self.identifier + '_main__Op0_AneInference Prediction'}
        row.update(changes)
        return row

    def select(self, row):
        return AUDIT.own_prediction_events([row], self.identifier, 100.0, self.target, hardware=True)

    def test_explicit_own_active_prediction(self):
        self.assertEqual(len(self.select(self.row())), 1)

    def test_global_wrong_process_load_idle_or_wrong_time_fail(self):
        for change in ({'formatted-label': 'Neural Engine Prediction'},
                       {'formatted-label': self.identifier + '0 Prediction'},
                       {'formatted-label': self.identifier + ' Prediction'},
                       {'formatted-label': self.identifier + ' Load: cached'},
                       {'gpu-state': 'Idle'}, {'ane-event-name': 'GPU'},
                       {'start-time-ns': 450000000}):
            with self.subTest(change=change):
                self.assertEqual(self.select(self.row(**change)), [])

    def test_instruments_references_and_nested_model_name(self):
        xml = '''<trace-query-result><row><start-time id="1">250000000</start-time>
          <duration id="2">100000000</duration><ane-event-name id="3">Apple Neural Engine</ane-event-name>
          <formatted-label id="4"><string>{name}_main__Op0_AneInference</string><narrative-text> Prediction</narrative-text></formatted-label>
          <gpu-state id="5">Active</gpu-state></row><row><start-time ref="1"/><duration ref="2"/>
          <ane-event-name ref="3"/><formatted-label ref="4"/><gpu-state ref="5"/></row></trace-query-result>'''.format(name=self.identifier)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'table.xml'
            path.write_text(xml)
            rows = AUDIT.resolved_rows(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(AUDIT.own_prediction_events(rows, self.identifier, 100, self.target, hardware=True)), 2)


if __name__ == '__main__':
    unittest.main()
