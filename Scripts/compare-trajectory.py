#!/usr/bin/env python3
"""Compare every VASP ionic step; final-step parity alone is insufficient."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def values(node):
    if node is None:
        return []
    return [[number(v) for v in (row.text or '').split()] for row in node.findall('v')]


def number(value):
    return float(value.replace('D','E').replace('d','e'))


def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def dynamics_parameters(root,incar,steps):
    # The XML parameters describe the executed run, including VASP defaults.
    requested={}
    for line in incar.splitlines():
        for item in re.split(r'[;]',re.split(r'[!#]',line,1)[0]):
            match=re.match(r'\s*(IBRION|NSW|POTIM)\s*=\s*(\S+)',item,re.I)
            if match:requested[match.group(1).upper()]=match.group(2)
    actual={}
    for name in ('IBRION','NSW','POTIM'):
        value=root.findtext(f"parameters/.//i[@name='{name}']")
        if value is None:value=root.findtext(f"incar/i[@name='{name}']")
        if value is None:value=requested.get(name)
        parsed=number(value.strip()) if value is not None else None
        if name!='POTIM' and parsed is not None:
            if not finite_number(parsed) or not parsed.is_integer():raise ValueError(f'Invalid {name}')
            parsed=int(parsed)
        actual[name]=parsed
    signal=any('kinetic' in step['energy'] or 'total' in step['energy'] for step in steps)
    return dict(required=actual['IBRION']==0 or signal,ibrion=actual['IBRION'],
                nsw=actual['NSW'],potim_fs=actual['POTIM'])


def contcar_velocities(text,expected_atoms,potim_fs=None):
    """Read VASP4/5 ionic velocities, excluding lattice and predictor blocks.

    VASP writes a blank velocity-mode line followed by Cartesian Angstrom/fs.
    Its explicit Direct input stores fractional displacement per POTIM, so that
    form needs the scaled cell and POTIM to produce the same Cartesian units.
    """
    lines=text.splitlines()
    if len(lines)<8:raise ValueError('CONTCAR structure is incomplete')
    scale=[number(v) for v in lines[1].split()]
    cell=[[number(v) for v in line.split()] for line in lines[2:5]]
    if len(scale) not in (1,3) or any(len(row)!=3 for row in cell):
        raise ValueError('CONTCAR lattice dimensions are invalid')
    if not all(finite_number(v) for v in scale+sum(cell,[])):
        raise ValueError('CONTCAR lattice is nonfinite')
    if len(scale)==1:
        factor=scale[0]
        if factor<0:
            volume=abs(sum(cell[0][i]*(cell[1][(i+1)%3]*cell[2][(i+2)%3]-cell[1][(i+2)%3]*cell[2][(i+1)%3]) for i in range(3)))
            if not finite_number(volume) or volume==0:raise ValueError('CONTCAR cell volume is invalid')
            factor=(-factor/volume)**(1/3)
        if not finite_number(factor) or factor<=0:raise ValueError('CONTCAR scale is invalid')
        scale=[factor]*3
    elif any(v<=0 for v in scale):raise ValueError('CONTCAR scales must be positive')
    cell=[[v*scale[j] for j,v in enumerate(row)] for row in cell]
    if not all(finite_number(v) for row in cell for v in row):raise ValueError('CONTCAR scaled lattice is nonfinite')
    index=5
    tokens=lines[index].split()
    if not tokens or not all(re.fullmatch(r'\d+',v) for v in tokens):
        symbols=tokens;index+=1;tokens=lines[index].split()
        if len(symbols)!=len(tokens):raise ValueError('CONTCAR element/count dimensions differ')
    if not tokens or not all(re.fullmatch(r'\d+',v) for v in tokens):
        raise ValueError('CONTCAR atom counts are invalid')
    atoms=sum(int(v) for v in tokens)
    if atoms!=expected_atoms or atoms<=0:raise ValueError('CONTCAR atom count differs from XML')
    index+=1
    if index<len(lines) and lines[index].strip().lower().startswith('s'):index+=1
    if index>=len(lines) or not lines[index].strip() or lines[index].strip()[0].lower() not in 'dck':
        raise ValueError('CONTCAR position coordinate mode is missing')
    index+=1
    for _ in range(atoms):
        if index>=len(lines) or len(lines[index].split())<3:raise ValueError('CONTCAR positions are incomplete')
        if not all(finite_number(number(v)) for v in lines[index].split()[:3]):raise ValueError('CONTCAR positions are nonfinite')
        index+=1
    while index<len(lines) and not lines[index].strip():index+=1
    if index<len(lines) and lines[index].strip().lower().startswith('lattice velocities'):
        index+=1
        if index>=len(lines) or not re.fullmatch(r'\s*\d+\s*',lines[index]):
            raise ValueError('CONTCAR lattice-velocity initialization is invalid')
        index+=1
        for _ in range(6):
            if index>=len(lines) or len(lines[index].split())!=3:raise ValueError('CONTCAR lattice-velocity block is incomplete')
            if not all(finite_number(number(v)) for v in lines[index].split()):raise ValueError('CONTCAR lattice-velocity block is nonfinite')
            index+=1
        while index<len(lines) and not lines[index].strip():index+=1
    if index>=len(lines):raise ValueError('CONTCAR ionic velocities are missing')
    mode='cartesian'
    label=lines[index].strip().lower()
    if label and label[0] in 'cdk':
        mode='direct' if label.startswith('d') else 'cartesian';index+=1
    velocities=[]
    for _ in range(atoms):
        if index>=len(lines) or len(lines[index].split())!=3:raise ValueError('CONTCAR ionic velocities are incomplete')
        row=[number(v) for v in lines[index].split()]
        if not all(finite_number(v) for v in row):raise ValueError('CONTCAR ionic velocities are nonfinite')
        velocities.append(row);index+=1
    if mode=='direct':
        if not finite_number(potim_fs) or potim_fs<=0:raise ValueError('Direct velocities require a positive finite POTIM')
        velocities=[[sum(row[k]*cell[k][j] for k in range(3))/potim_fs for j in range(3)] for row in velocities]
    if not all(finite_number(v) for row in velocities for v in row):raise ValueError('CONTCAR converted velocities are nonfinite')
    return dict(values=velocities,coordinate_mode=mode,units='angstrom/fs',error=None)


def read(directory):
    d=Path(directory)
    root=ET.parse(d/'vasprun.xml').getroot()
    atoms=int(root.findtext('atominfo/atoms'))
    steps=[]
    for calc in root.findall('calculation'):
        energy={i.attrib['name']:number(i.text) for i in calc.findall('energy/i')}
        scf=calc.findall('scstep')
        last=scf[-1].findtext("energy/i[@name='e_fr_energy']") if scf else None
        steps.append(dict(energy=energy,last_scf_energy=number(last) if last else None,scf_iterations=len(scf),
            forces=values(calc.find("varray[@name='forces']")),
            stress=values(calc.find("varray[@name='stress']")),
            positions=values(calc.find("structure/varray[@name='positions']")),
            cell=values(calc.find("structure/crystal/varray[@name='basis']"))))
    out=(d/'OUTCAR').read_text(errors='replace')
    logs=next((d/n for n in ('run.log','vasp.log','stdout.log') if (d/n).exists()),None)
    if logs:out+='\n'+logs.read_text(errors='replace')
    # Only OUTCAR records convergence. Stdout is included only for warnings.
    ediff=(d/'OUTCAR').read_text(errors='replace').count('aborting loop because EDIFF is reached')
    md=dynamics_parameters(root,(d/'INCAR').read_text(),steps)
    try:velocity=contcar_velocities((d/'CONTCAR').read_text(),atoms,md['potim_fs'])
    except (OSError,ValueError) as error:
        velocity=dict(values=[],coordinate_mode=None,units='angstrom/fs',error=str(error))
    return dict(atom_count=atoms,steps=steps,ediff_completed_steps=ediff,md=md,final_velocity=velocity,
        completed='General timing and accounting' in out,
        hermitian_warnings=len(re.findall(r'not\s+hermitian',out,re.I)),
        input_sha256={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ('INCAR','POSCAR','POTCAR','KPOINTS')})


def maxdiff(a,b):
    if len(a)!=len(b) or not a or any(len(x)!=len(y) for x,y in zip(a,b)):
        return None
    v=[abs(x-y) for aa,bb in zip(a,b) for x,y in zip(aa,bb)]
    return max(v) if v and all(math.isfinite(x) for x in v) else None


def md_required(run):
    return run.get('md',{}).get('required',False) or any('kinetic' in s['energy'] or 'total' in s['energy'] for s in run['steps'])


def md_parameters_valid(run):
    md=run.get('md',{})
    return (md.get('ibrion')==0 and isinstance(md.get('nsw'),int) and md['nsw']>=0
            and finite_number(md.get('potim_fs')) and md['potim_fs']>0)


def md_summary(run):
    md=run.get('md',{});dt=md.get('potim_fs');nsw=md.get('nsw');count=len(run['steps'])
    dt=dt if finite_number(dt) and dt>0 else None
    nominal=nsw*dt if isinstance(nsw,int) and nsw>=0 and dt is not None else None
    span=max(0,count-1)*dt if dt is not None else None
    missing=[i for i,s in enumerate(run['steps'],1) if not finite_number(s['energy'].get('total'))]
    total=[s['energy'].get('total') for s in run['steps']]
    energy=None
    if total and not missing:
        energy=dict(first_ev=total[0],last_ev=total[-1],first_to_last_drift_ev=total[-1]-total[0],
                    minimum_ev=min(total),maximum_ev=max(total),range_ev=max(total)-min(total))
        if not all(finite_number(v) for v in energy.values()):energy=None
    velocity=run.get('final_velocity',{})
    return dict(ibrion=md.get('ibrion'),configured_nsw=nsw,potim_fs=dt,
                nominal_nsw_times_potim_fs=nominal if finite_number(nominal) else None,
                recorded_frames=count,recorded_frame_intervals=max(0,count-1),
                recorded_frame_span_fs=span if finite_number(span) else None,
                time_accounting='Nominal NSW*POTIM; first-to-last XML frame span is (frames-1)*POTIM for constant timestep. No absolute frame timestamps are inferred.',
                total_energy=energy,total_energy_missing_or_nonfinite_steps=missing,
                total_energy_definition='VASP XML total (ETOTAL), including ionic/lattice kinetic and Nose thermostat energy terms when present. Drift/range are descriptive, with no stability threshold.',
                final_velocity_coordinate_mode=velocity.get('coordinate_mode'),
                final_velocity_error=velocity.get('error'))


def compare(a,b,expected_steps=None,tolerance=None):
    t=dict(energy_ev=1e-5,force_ev_per_ang=1e-4,stress_kbar=1e-3,position_ang=1e-4,cell_ang=1e-8,velocity_ang_per_fs=1e-6)
    if tolerance:t.update(tolerance)
    md=md_required(a) or md_required(b)
    md_parameter_gate=(all(md_required(z) and md_parameters_valid(z) for z in (a,b))
                       and a.get('md')==b.get('md')) if md else None
    identical=a['input_sha256']==b['input_sha256']
    equal=len(a['steps'])==len(b['steps']) and a['atom_count']==b['atom_count']
    expected=expected_steps is None or all(len(x['steps'])==expected_steps for x in (a,b))
    valid=len(a['steps'])>0 and equal and expected and identical and all(x['completed'] and not x['hermitian_warnings'] and x['ediff_completed_steps']==len(x['steps']) for x in (a,b))
    rows=[]
    for i,(x,y) in enumerate(zip(a['steps'],b['steps']),1):
        e=abs(x['energy'].get('e_fr_energy',float('nan'))-y['energy'].get('e_fr_energy',float('nan')))
        se=abs(x['last_scf_energy']-y['last_scf_energy']) if x['last_scf_energy'] is not None and y['last_scf_energy'] is not None else float('nan')
        f,s,cell=[maxdiff(x[k],y[k]) for k in ('forces','stress','cell')]
        position=None
        dimensions=all(len(z['forces'])==a['atom_count'] and all(len(row)==3 for row in z['forces']) and len(z['stress'])==3 and all(len(row)==3 for row in z['stress']) and len(z['cell'])==3 and all(len(row)==3 for row in z['cell']) and len(z['positions'])==a['atom_count'] and all(len(row)==3 for row in z['positions']) for z in (x,y))
        if dimensions:
            distances=[]
            for pa,pb in zip(x['positions'],y['positions']):
                df=[u-v for u,v in zip(pa,pb)]
                df=[v-round(v) if math.isfinite(v) else float('nan') for v in df]
                cart=[sum(df[k]*x['cell'][k][j] for k in range(3)) for j in range(3)]
                distances.append(math.sqrt(sum(v*v for v in cart)))
            position=max(distances)
        finite=dimensions and all(math.isfinite(v) for z in (x,y) for key in ('forces','stress','cell','positions') for row in z[key] for v in row) and all(math.isfinite(v) for v in (e,se)) and all(v is not None and math.isfinite(v) for v in (f,s,cell,position))
        passed=finite and e<=t['energy_ev'] and se<=t['energy_ev'] and f<=t['force_ev_per_ang'] and s<=t['stress_kbar'] and cell<=t['cell_ang'] and position<=t['position_ang'] and x['scf_iterations']==y['scf_iterations']
        md_deltas={}
        md_energy_passed=True
        if md:
            for name in ('kinetic','total'):
                u=x['energy'].get(name);v=y['energy'].get(name)
                delta=abs(u-v) if finite_number(u) and finite_number(v) else None
                md_deltas[f'delta_{name}_energy_ev']=delta if finite_number(delta) else None
                md_energy_passed=md_energy_passed and finite_number(delta) and delta<=t['energy_ev']
            passed=passed and md_energy_passed
        row=dict(step=i,passed=bool(passed),delta_energy_ev=e if math.isfinite(e) else None,
            delta_scf_energy_ev=se if math.isfinite(se) else None,max_force_delta_ev_per_ang=f,
            max_stress_delta_kbar=s,max_cell_delta_ang=cell,
            max_position_delta_ang=position if position is not None and math.isfinite(position) else None,
            reference_scf_iterations=x['scf_iterations'],candidate_scf_iterations=y['scf_iterations'],
            md_energy_gate=bool(md_energy_passed) if md else None,**md_deltas)
        rows.append(row);valid=valid and passed
    velocity_delta=None;velocity_gate=None
    if md:
        velocity_dimensions=all(z.get('final_velocity',{}).get('error') is None
            and len(z.get('final_velocity',{}).get('values',[]))==a['atom_count']
            and all(len(row)==3 and all(finite_number(v) for v in row) for row in z['final_velocity']['values']) for z in (a,b))
        if velocity_dimensions:velocity_delta=maxdiff(a['final_velocity']['values'],b['final_velocity']['values'])
        velocity_gate=finite_number(velocity_delta) and velocity_delta<=t['velocity_ang_per_fs']
        valid=valid and md_parameter_gate and velocity_gate
    md_energy_gate=all(r['md_energy_gate'] for r in rows) and equal and expected and bool(rows) if md else None
    md_maxima={}
    if md:
        for name in ('kinetic','total'):
            deltas=[r[f'delta_{name}_energy_ev'] for r in rows]
            md_maxima[f'max_{name}_energy_delta_ev']=max(deltas) if deltas and all(finite_number(v) for v in deltas) else None
    return dict(numerical_trajectory_gate=bool(valid),input_identity_gate=identical,equal_step_count_gate=equal,
                expected_step_count_gate=expected,reference_steps=len(a['steps']),candidate_steps=len(b['steps']),
                reference_ediff_steps=a['ediff_completed_steps'],candidate_ediff_steps=b['ediff_completed_steps'],
                md_required=bool(md),md_parameters_gate=md_parameter_gate,md_energy_gate=md_energy_gate,
                final_velocity_gate=bool(velocity_gate) if md else None,max_final_velocity_delta_ang_per_fs=velocity_delta,
                **md_maxima,
                reference_md=md_summary(a) if md else None,candidate_md=md_summary(b) if md else None,
                tolerances=t,failed_steps=[r['step'] for r in rows if not r['passed']],steps=rows,
                scope='Engineering pointwise regression: every completed ionic step, and MD kinetic/total energies plus final CONTCAR ionic velocities. Energy/velocity bounds are numerical parity tolerances, not physical convergence or stability criteria. Per-step velocities are not checked; descriptive energy drift/range impose no stability threshold.')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--reference',required=True);ap.add_argument('--candidate',required=True)
    ap.add_argument('--expected-steps',type=int);ap.add_argument('--output',required=True)
    a=ap.parse_args()
    try:r=compare(read(a.reference),read(a.candidate),a.expected_steps)
    except (OSError,ValueError,TypeError,ET.ParseError) as e:ap.exit(2,f'Trajectory comparison failed: {e}\n')
    Path(a.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='steps'},indent=2))
    return 0 if r['numerical_trajectory_gate'] else 1


if __name__=='__main__':raise SystemExit(main())
