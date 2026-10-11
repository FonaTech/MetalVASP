#!/usr/bin/env python3
"""Export official Core ML trace tables; fail closed on unknown event schemas.

Preferred-device plans and supported-device inventories never constitute the
actual runtime execution gate. Retain raw XML for schema-specific inspection.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


def normalized(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())


def export(trace, output, arguments):
    completed = subprocess.run(['xcrun', 'xctrace', 'export', '--input', str(trace),
                                '--output', str(output)] + arguments,
                               capture_output=True, text=True, timeout=60)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


def inspect_table(path):
    root = ET.parse(path).getroot()
    rows = root.findall('.//row')
    fields = []
    for element in root.iter():
        if element.tag in ('col', 'column'):
            fields.append(dict(element.attrib, **{child.tag: child.text for child in element if child.text}))
    # Do not promote a generic text match to execution evidence. A new Apple
    # schema must be inspected and explicitly supported before the strict gate.
    matches = []
    for index, row in enumerate(rows):
        value = ET.tostring(row, encoding='unicode')
        if 'neural' in value.lower() or re.search(r'\bANE\b', value):
            matches.append({'row_index': index, 'text_excerpt': value[:4000]})
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'row_count': len(rows), 'columns': fields, 'neural_engine_text_matches': matches[:25]}


def resolved_rows(path):
    """Resolve Instruments XML id/ref cells without treating arbitrary text as proof."""
    root = ET.parse(path).getroot()
    definitions = {element.get('id'): element for element in root.iter() if element.get('id')}

    def resolve(element, visited=()):
        if element.get('ref'):
            identifier = element.get('ref')
            if identifier in visited or identifier not in definitions:
                raise ValueError('cyclic or missing Instruments XML cell reference')
            return resolve(definitions[identifier], visited + (identifier,))
        # Nested strings can hold model identity while a parent has an empty
        # fmt attribute. Preserve both human-readable and raw leaf values.
        return ' '.join(filter(None, [element.get('fmt'), element.text and element.text.strip()] +
                               [resolve(child, visited) for child in element]))

    rows = []
    for row in root.findall('.//row'):
        cells = {element.tag: resolve(element) for element in row}
        for tag in ('start-time', 'duration'):
            element = row.find(tag)
            if element is None:
                continue
            while element.get('ref'):
                element = definitions[element.get('ref')]
            if element.text:
                cells[tag + '-ns'] = int(element.text)
        rows.append(cells)
    return rows


def own_prediction_events(rows, identifier, recording_start, target, hardware=False):
    """Require exact process-specific model identity and completed prediction overlap.

    Global ANE rows lacking this unique displayName are never attributed to the
    target, even when their timestamps overlap. Load/cache events also fail.
    """
    intervals = target.get('prediction_intervals', [])
    selected = []
    for row in rows:
        label = row.get('formatted-label', '') if hardware else ' '.join(
            row.get(key, '') for key in ('coreml-model-name', 'coreml-model-event', 'formatted-label'))
        if not re.search(re.escape(identifier) + r'(?![A-Za-z0-9])', label) or not re.search(
                r'\bprediction\b', ' '.join(str(value) for value in row.values()), re.I):
            continue
        if hardware and ('neural engine' not in row.get('ane-event-name', '').lower() or
                         set(row.get('gpu-state', '').lower().split()) != {'active'}):
            continue
        if hardware and not re.search(re.escape(identifier) + r'_main__Op\d+_AneInference\b', label):
            continue
        if 'start-time-ns' not in row or 'duration-ns' not in row:
            continue
        begin = recording_start + row['start-time-ns'] / 1e9
        end = begin + row['duration-ns'] / 1e9
        if not any(begin < interval.get('end_unix_seconds', 0) and
                   end > interval.get('start_unix_seconds', float('inf')) for interval in intervals):
            continue
        selected.append(dict(row, start_unix_seconds=begin, end_unix_seconds=end))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--target-result', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite trace audit evidence')
    args.output.mkdir(parents=True)
    target = json.loads(args.target_result.read_text()) if args.target_result.is_file() else {}
    report = {'schema': 2, 'trace': str(args.trace.resolve()), 'target_pid': target.get('pid'),
              'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'target_result_present': args.target_result.is_file(),
              'target_completed': target.get('status') == 'prediction-completed',
              'anticipated_plan_is_execution_evidence': False, 'strict_ane_execution_trace_gate_passed': False,
              'performance_claim_accepted': False, 'tables': []}
    try:
        toc = args.output / 'toc.xml'
        export(args.trace, toc, ['--toc'])
        root = ET.parse(toc).getroot()
        report['recorded_targets'] = [dict(process.attrib) for process in root.findall('.//target/process')]
        report['target_pid_matches_recorded'] = any(
            str(target.get('pid')) == process.get('pid') for process in root.findall('.//target/process'))
        report['recorded_target_exited_successfully'] = any(
            str(target.get('pid')) == process.get('pid') and process.get('return-exit-status') == '0'
            for process in root.findall('.//target/process'))
        start_date = root.findtext('.//summary/start-date')
        if start_date:
            report['recording_start_unix_seconds'] = datetime.fromisoformat(start_date).timestamp()
        schemas = sorted({table.get('schema') for table in root.findall('.//table') if table.get('schema')})
        report['available_schemas'] = schemas
        selected = [schema for schema in schemas if 'coreml' in normalized(schema) or 'neuralengine' in normalized(schema)
                    or schema == 'ane-hw-intervals-internal']
        for index, schema in enumerate(selected):
            path = args.output / ('table-' + str(index) + '.xml')
            # Schema names originate from local Instruments, but preserve XPath
            # quoting rather than interpolating unexpected apostrophes.
            if "'" in schema:
                raise ValueError('unsupported quote in Instruments schema')
            query = f"/trace-toc/run[@number='1']/data/table[@schema='{schema}']"
            export(args.trace, path, ['--xpath', query])
            report['tables'].append(dict(schema=schema, **inspect_table(path)))
        report['status'] = 'exported'
        identifier = target.get('correlation_identifier', '')
        expected_identifier = 'MetalVASP.ANEProbe.Conv512.FP16.Native.PID' + str(target.get('pid'))
        report['own_coreml_prediction_events'] = []
        report['own_ane_hardware_prediction_events'] = []
        # This parser accepts only explicitly named native target events. A
        # coincident global ANE interval or a preferred-device plan fails.
        if (report['target_completed'] and report['target_pid_matches_recorded'] and
                report['recorded_target_exited_successfully'] and
                target.get('completed_prediction_count', 0) > 0 and identifier == expected_identifier and
                'recording_start_unix_seconds' in report):
            scope = any(table.get('schema') == 'coreml-os-signpost' and
                        table.get('target-pid') in ('SINGLE', str(target.get('pid')))
                        for table in root.findall('.//table'))
            for table in report['tables']:
                if table['schema'] not in ('coreml-os-signpost', 'ane-hw-intervals-internal'):
                    continue
                events = own_prediction_events(resolved_rows(Path(table['file'])), identifier,
                                               report['recording_start_unix_seconds'], target,
                                               hardware=table['schema'] == 'ane-hw-intervals-internal')
                if table['schema'] == 'coreml-os-signpost' and scope:
                    report['own_coreml_prediction_events'] = events
                elif table['schema'] == 'ane-hw-intervals-internal':
                    report['own_ane_hardware_prediction_events'] = events
            # The hardware schema supplies an explicit process-specific model
            # operation name. This identity + TOC PID + completed prediction
            # overlap establishes attribution even when Core ML signposts are
            # absent. Do not invent those missing software signpost rows.
            report['strict_ane_execution_trace_gate_passed'] = bool(report['own_ane_hardware_prediction_events'])
        report['own_coreml_prediction_event_count'] = len(report['own_coreml_prediction_events'])
        report['own_ane_hardware_prediction_event_count'] = len(report['own_ane_hardware_prediction_events'])
        report['coreml_prediction_signposts_present'] = bool(report['own_coreml_prediction_events'])
        report['gate_reason'] = ('target prediction did not complete in the recorded process' if
                                 not report['target_completed'] or not report['target_pid_matches_recorded'] else
                                 'raw runtime schemas require explicit device-event inspection; generic ANE text/plan is insufficient')
        if report['strict_ane_execution_trace_gate_passed']:
            report['gate_reason'] = 'completed exit0 target with matching TOC PID and PID-specific model Op/AneInference name in active ANE Prediction hardware events overlapping recorded prediction intervals'
    except Exception as error:
        report.update(status='failed', error=str(error), gate_reason='trace export or inspection failed')
    (args.output / 'trace-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report.get(key) for key in
                      ('schema', 'status', 'target_pid', 'target_completed', 'target_pid_matches_recorded',
                       'recorded_target_exited_successfully', 'own_coreml_prediction_event_count',
                       'own_ane_hardware_prediction_event_count', 'strict_ane_execution_trace_gate_passed',
                       'gate_reason', 'error')}, indent=2))
    return 0 if report['status'] == 'exported' else 1


if __name__ == '__main__':
    raise SystemExit(main())
