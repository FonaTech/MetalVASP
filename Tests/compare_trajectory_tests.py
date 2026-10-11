#!/usr/bin/env python3
"""Negative gates: detect corruption before the final trajectory frame."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

path=Path(__file__).parents[1]/'Scripts/compare-trajectory.py'
spec=importlib.util.spec_from_file_location('trajectory',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    step=dict(energy={'e_fr_energy':-2.0},last_scf_energy=-2.0,scf_iterations=4,
              forces=[[0.,0.,0.],[0.,0.,0.]],stress=[[0.,0.,0.]]*3,
              cell=[[4.,0.,0.],[0.,4.,0.],[0.,0.,4.]],positions=[[0.,0.,0.],[.25,.25,.25]])
    return dict(atom_count=2,steps=[copy.deepcopy(step) for _ in range(3)],ediff_completed_steps=3,
                completed=True,hermitian_warnings=0,input_sha256={'INCAR':'same'})


def md_fixture():
    run=fixture()
    run['md']=dict(required=True,ibrion=0,nsw=3,potim_fs=.5)
    run['final_velocity']=dict(values=[[.001,.002,.003],[-.001,-.002,-.003]],
                               coordinate_mode='cartesian',units='angstrom/fs',error=None)
    for i,step in enumerate(run['steps']):
        step['energy'].update(kinetic=.02+i*.001,total=-1.98+i*.002)
    return run


def contcar(vasp5=True,selective=False,mode=None,velocities=None,tail='',lattice=False,scale='1'):
    """Synthetic, two-atom format fixture; values are not simulation evidence."""
    lines=['synthetic Si2',scale,'4 0 0','0 4 0','0 0 4']
    if vasp5:lines.append('Si')
    lines.append('2')
    if selective:lines.append('Selective dynamics')
    lines.extend(['Direct','0 0 0'+(' T F T' if selective else ''),
                  '.25 .25 .25'+(' F T F' if selective else '')])
    if lattice:lines.extend(['Lattice velocities and vectors','1','100 101 102','103 104 105',
                             '106 107 108','4 0 0','0 4 0','0 0 4'])
    lines.append('' if mode is None else mode)
    if velocities is not False:
        lines.extend(velocities or ['.001 .002 .003','-.001 -.002 -.003'])
    return '\n'.join(lines)+'\n'+tail


def write_fixture(directory,run=None,velocity_text=None,include_parameters=True):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    run=run or md_fixture();root=ET.Element('modeling')
    ET.SubElement(ET.SubElement(root,'atominfo'),'atoms').text='2'
    md=run.get('md',dict(ibrion=-1,nsw=0,potim_fs=.5))
    if include_parameters:
        params=ET.SubElement(root,'parameters');separator=ET.SubElement(params,'separator',name='ionic')
        for name,key in [('IBRION','ibrion'),('NSW','nsw'),('POTIM','potim_fs')]:
            ET.SubElement(separator,'i',name=name).text=str(md[key])
    for step in run['steps']:
        calc=ET.SubElement(root,'calculation');energy=ET.SubElement(calc,'energy')
        for name,value in step['energy'].items():ET.SubElement(energy,'i',name=name).text=str(value)
        for _ in range(step['scf_iterations']):
            scf=ET.SubElement(calc,'scstep');en=ET.SubElement(scf,'energy')
            ET.SubElement(en,'i',name='e_fr_energy').text=str(step['last_scf_energy'])
        def array(parent,name,rows):
            node=ET.SubElement(parent,'varray',name=name)
            for row in rows:ET.SubElement(node,'v').text=' '.join(map(str,row))
        array(calc,'forces',step['forces']);array(calc,'stress',step['stress'])
        structure=ET.SubElement(calc,'structure');array(structure,'positions',step['positions'])
        array(ET.SubElement(structure,'crystal'),'basis',step['cell'])
    ET.ElementTree(root).write(directory/'vasprun.xml')
    (directory/'OUTCAR').write_text('aborting loop because EDIFF is reached\n'*run['ediff_completed_steps']+'General timing and accounting\n')
    (directory/'INCAR').write_text(f"IBRION = {md['ibrion']}; NSW = {md['nsw']} ! fixture\nPOTIM = {md['potim_fs']} # timestep\n")
    for name in ('POSCAR','POTCAR','KPOINTS'):(directory/name).write_text('synthetic fixture only\n')
    if velocity_text is not False:(directory/'CONTCAR').write_text(velocity_text or contcar())


class Gates(unittest.TestCase):
    def test_equal(self):
        a=fixture();self.assertTrue(m.compare(a,copy.deepcopy(a),3)['numerical_trajectory_gate'])
    def test_early_corruption(self):
        a=fixture();b=copy.deepcopy(a);b['steps'][0]['forces'][0][0]=.2
        r=m.compare(a,b,3);self.assertFalse(r['numerical_trajectory_gate']);self.assertEqual(r['failed_steps'],[1])
    def test_missing_step(self):
        a=fixture();b=copy.deepcopy(a);b['steps'].pop();self.assertFalse(m.compare(a,b,3)['numerical_trajectory_gate'])
    def test_unconverged(self):
        a=fixture();b=copy.deepcopy(a);b['ediff_completed_steps']=2;self.assertFalse(m.compare(a,b,3)['numerical_trajectory_gate'])
    def test_nonfinite(self):
        a=fixture();b=copy.deepcopy(a);b['steps'][1]['positions'][0][0]=float('nan')
        report=m.compare(a,b,3)
        self.assertFalse(report['numerical_trajectory_gate'])
        json.dumps(report,allow_nan=False)
    def test_iteration_mismatch(self):
        a=fixture();b=copy.deepcopy(a);b['steps'][0]['scf_iterations']=5;self.assertFalse(m.compare(a,b,3)['numerical_trajectory_gate'])
    def test_periodic_position(self):
        a=fixture();b=copy.deepcopy(a);b['steps'][0]['positions'][0][0]=1.;self.assertTrue(m.compare(a,b,3)['numerical_trajectory_gate'])
    def test_changed_input(self):
        a=fixture();b=copy.deepcopy(a);b['input_sha256']['INCAR']='changed';self.assertFalse(m.compare(a,b,3)['numerical_trajectory_gate'])


class MolecularDynamicsGates(unittest.TestCase):
    def assert_failed_json_safe(self,a,b):
        report=m.compare(a,b,3)
        self.assertFalse(report['numerical_trajectory_gate'])
        json.dumps(report,allow_nan=False)
        return report

    def test_equal_and_accounting(self):
        a=md_fixture();report=m.compare(a,copy.deepcopy(a),3)
        self.assertTrue(report['numerical_trajectory_gate'])
        self.assertTrue(report['md_energy_gate']);self.assertTrue(report['final_velocity_gate'])
        self.assertEqual(report['tolerances']['energy_ev'],1e-5)
        self.assertEqual(report['tolerances']['velocity_ang_per_fs'],1e-6)
        summary=report['reference_md']
        self.assertEqual(summary['nominal_nsw_times_potim_fs'],1.5)
        self.assertEqual(summary['recorded_frame_span_fs'],1.)
        self.assertEqual(summary['recorded_frame_intervals'],2)
        self.assertAlmostEqual(summary['total_energy']['first_to_last_drift_ev'],.004)
        self.assertAlmostEqual(summary['total_energy']['range_ev'],.004)

    def test_every_md_energy_missing(self):
        for field in ('kinetic','total'):
            for side in ('reference','candidate','both'):
                with self.subTest(field=field,side=side):
                    a=md_fixture();b=copy.deepcopy(a)
                    if side in ('reference','both'):del a['steps'][1]['energy'][field]
                    if side in ('candidate','both'):del b['steps'][1]['energy'][field]
                    report=self.assert_failed_json_safe(a,b)
                    self.assertFalse(report['md_energy_gate']);self.assertEqual(report['failed_steps'],[2])
                    self.assertIsNone(report['steps'][1][f'delta_{field}_energy_ev'])

    def test_every_md_energy_nonfinite(self):
        for field in ('kinetic','total'):
            for value in (float('nan'),float('inf'),-float('inf')):
                with self.subTest(field=field,value=value):
                    a=md_fixture();b=copy.deepcopy(a);a['steps'][0]['energy'][field]=value;b['steps'][0]['energy'][field]=value
                    report=self.assert_failed_json_safe(a,b)
                    self.assertEqual(report['failed_steps'],[1])
                    if field=='total':self.assertIsNone(report['reference_md']['total_energy'])

    def test_early_kinetic_and_total_delta(self):
        for field in ('kinetic','total'):
            with self.subTest(field=field):
                a=md_fixture();b=copy.deepcopy(a);b['steps'][0]['energy'][field]+=1.1e-5
                report=self.assert_failed_json_safe(a,b);self.assertEqual(report['failed_steps'],[1])
                b['steps'][0]['energy'][field]=a['steps'][0]['energy'][field]+.5e-5
                self.assertTrue(m.compare(a,b,3)['numerical_trajectory_gate'])

    def test_missing_final_velocities(self):
        for side in ('reference','candidate','both'):
            with self.subTest(side=side):
                a=md_fixture();b=copy.deepcopy(a)
                if side in ('reference','both'):a.pop('final_velocity')
                if side in ('candidate','both'):b.pop('final_velocity')
                self.assertFalse(self.assert_failed_json_safe(a,b)['final_velocity_gate'])

    def test_final_velocity_nonfinite_shape_and_error(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.subTest(value=value):
                a=md_fixture();b=copy.deepcopy(a);b['final_velocity']['values'][0][0]=value
                self.assertFalse(self.assert_failed_json_safe(a,b)['final_velocity_gate'])
        for change in ('atoms','components','parse_error'):
            with self.subTest(change=change):
                a=md_fixture();b=copy.deepcopy(a)
                if change=='atoms':b['final_velocity']['values'].pop()
                if change=='components':b['final_velocity']['values'][0].pop()
                if change=='parse_error':b['final_velocity']['error']='missing velocity block'
                self.assertFalse(self.assert_failed_json_safe(a,b)['final_velocity_gate'])

    def test_final_velocity_delta(self):
        a=md_fixture();b=copy.deepcopy(a);b['final_velocity']['values'][1][2]+=1.1e-6
        self.assertFalse(self.assert_failed_json_safe(a,b)['final_velocity_gate'])
        b['final_velocity']['values'][1][2]=a['final_velocity']['values'][1][2]+.5e-6
        self.assertTrue(m.compare(a,b,3)['numerical_trajectory_gate'])

    def test_md_parameters_required_and_identical(self):
        for change in ('missing','zero_potim','nan_potim','changed_nsw','wrong_mode'):
            with self.subTest(change=change):
                a=md_fixture();b=copy.deepcopy(a)
                if change=='missing':b.pop('md')
                if change=='zero_potim':b['md']['potim_fs']=0.
                if change=='nan_potim':b['md']['potim_fs']=float('nan')
                if change=='changed_nsw':b['md']['nsw']=4
                if change=='wrong_mode':b['md']['ibrion']=-1
                self.assertFalse(self.assert_failed_json_safe(a,b)['md_parameters_gate'])

    def test_static_does_not_require_md_data(self):
        a=fixture();report=m.compare(a,copy.deepcopy(a),3)
        self.assertTrue(report['numerical_trajectory_gate'])
        self.assertFalse(report['md_required']);self.assertIsNone(report['final_velocity_gate'])

    def test_large_identical_drift_has_no_physical_threshold(self):
        a=md_fixture();a['steps'][1]['energy']['total']=100.
        report=m.compare(a,copy.deepcopy(a),3)
        self.assertTrue(report['numerical_trajectory_gate'])
        self.assertAlmostEqual(report['reference_md']['total_energy']['range_ev'],101.98)


class VelocityParser(unittest.TestCase):
    def test_vasp4_vasp5_selective_bare_and_labelled(self):
        for vasp5 in (False,True):
            for selective in (False,True):
                for mode in (None,'Cartesian','C','K'):
                    with self.subTest(vasp5=vasp5,selective=selective,mode=mode):
                        result=m.contcar_velocities(contcar(vasp5=vasp5,selective=selective,mode=mode),2,.5)
                        self.assertEqual(result['coordinate_mode'],'cartesian')
                        self.assertEqual(result['values'],md_fixture()['final_velocity']['values'])

    def test_predictor_follows_velocities(self):
        predictor='\n1\n.5\n1 0 0 0\n999 998 997\n996 995 994\n'
        result=m.contcar_velocities(contcar(tail=predictor),2,.5)
        self.assertEqual(result['values'],md_fixture()['final_velocity']['values'])
        with self.assertRaises(ValueError):m.contcar_velocities(contcar(velocities=False,tail=predictor),2,.5)

    def test_lattice_velocities_are_not_ionic(self):
        result=m.contcar_velocities(contcar(lattice=True),2,.5)
        self.assertEqual(result['values'],md_fixture()['final_velocity']['values'])
        with self.assertRaises(ValueError):m.contcar_velocities(contcar(lattice=True,velocities=False),2,.5)

    def test_direct_conversion_uses_scaled_cell_and_potim(self):
        for scale in ('2','2 3 4','-512'):
            with self.subTest(scale=scale):
                result=m.contcar_velocities(contcar(mode='Direct',scale=scale),2,.5)
                factors=[8,12,16] if scale=='2 3 4' else [8,8,8]
                self.assertEqual(result['values'][0],[.001*factors[0]/.5,.002*factors[1]/.5,.003*factors[2]/.5])
        for dt in (None,0.,float('nan')):
            with self.subTest(dt=dt):
                with self.assertRaises(ValueError):m.contcar_velocities(contcar(mode='Direct'),2,dt)

    def test_nonorthogonal_direct_conversion(self):
        text=contcar(mode='Direct').replace('4 0 0\n0 4 0\n0 0 4','0 2 2\n2 0 2\n2 2 0')
        values=m.contcar_velocities(text,2,.5)['values'][0]
        for actual,expected in zip(values,[.02,.016,.012]):self.assertAlmostEqual(actual,expected)

    def test_fortran_exponents(self):
        result=m.contcar_velocities(contcar(velocities=['1d-3 2D-3 3E-3','-1D-3 -2d-3 -3E-3']),2,.5)
        self.assertEqual(result['values'],md_fixture()['final_velocity']['values'])

    def test_incomplete_nonfinite_wrong_count_and_empty_spheres(self):
        cases=[contcar(velocities=False),contcar(velocities=['.001 .002 .003']),
               contcar(velocities=['nan .002 .003','-.001 -.002 -.003']),
               contcar(velocities=['inf .002 .003','-.001 -.002 -.003']),
               contcar(velocities=['.001 .002 .003 junk','-.001 -.002 -.003']),
               contcar().replace('\n2\nDirect','\n3\nDirect'),
               contcar().replace('\n\n.001','\nEmpty spheres\n1\n0 0 0\n\n.001')]
        for index,text in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):m.contcar_velocities(text,2,.5)


class FileAndCommandGates(unittest.TestCase):
    def test_actual_read_with_incar_fallback_and_xml_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            write_fixture(directory,include_parameters=False)
            run=m.read(directory);self.assertEqual(run['md'],md_fixture()['md'])
            write_fixture(directory)
            (directory/'INCAR').write_text('IBRION = -1; NSW = 0; POTIM = 9\n')
            self.assertEqual(m.read(directory)['md'],md_fixture()['md'])

    def test_missing_and_malformed_contcar_fail_md_but_not_static(self):
        for text in (False,contcar(velocities=False),contcar(velocities=['nan 0 0','0 0 0'])):
            with self.subTest(text=text):
                with tempfile.TemporaryDirectory() as temporary:
                    reference=Path(temporary)/'cpu';candidate=Path(temporary)/'gpu'
                    write_fixture(reference);write_fixture(candidate,velocity_text=text)
                    report=m.compare(m.read(reference),m.read(candidate),3)
                    self.assertFalse(report['numerical_trajectory_gate'])
                    self.assertFalse(report['final_velocity_gate'])
                    self.assertTrue(report['candidate_md']['final_velocity_error'])
                    json.dumps(report,allow_nan=False)
        with tempfile.TemporaryDirectory() as temporary:
            reference=Path(temporary)/'cpu';candidate=Path(temporary)/'gpu'
            write_fixture(reference,run=fixture(),velocity_text=False);write_fixture(candidate,run=fixture(),velocity_text=False)
            self.assertTrue(m.compare(m.read(reference),m.read(candidate),3)['numerical_trajectory_gate'])

    def test_cli_missing_energy_and_velocity_have_failure_exit_and_report(self):
        for field in ('kinetic','total','velocity'):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root=Path(temporary);a=root/'cpu';b=root/'gpu';report_path=root/'report.json'
                    write_fixture(a);run=md_fixture()
                    if field!='velocity':del run['steps'][0]['energy'][field]
                    write_fixture(b,run=run,velocity_text=False if field=='velocity' else None)
                    completed=subprocess.run([sys.executable,str(path),'--reference',str(a),'--candidate',str(b),
                                              '--expected-steps','3','--output',str(report_path)],capture_output=True,text=True)
                    self.assertEqual(completed.returncode,1,completed.stderr)
                    report=json.loads(report_path.read_text());self.assertFalse(report['numerical_trajectory_gate'])


if __name__=='__main__':unittest.main()
