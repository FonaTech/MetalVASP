#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reproduce CPU/GPU figures from public reviewed data, without VASP or a GPU."""
import os,tempfile,json,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='metalvasp-mpl-'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1];DATA=json.loads((ROOT/'benchmarks/data/measured.json').read_text())
OUT=ROOT/'docs/assets';OUT.mkdir(exist_ok=True)
CASES=DATA['cases'];N=len(CASES);Y=np.arange(N);NAMES=[c['system'] for c in CASES]
CPU='#7A8288';GPU='#007F86';BAD='#D55E00';BLUE='#0072B2';PURPLE='#AA6B9E'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':400})
FIGURES=[]
def save(fig,name,purpose,locators,evidence='measured'):
 fig.get_layout_engine().set(rect=(0,.065,1,.79 if name=='06-api-timers' else .91))
 footer='VASP 6.4.1 · Apple M5 Max · matched CPU/GPU · 3 formal repeats per route\n'+('Simulated electronic-output agreement; exact printed zeros retained' if evidence=='simulated' else f'Measured snapshot: {N} reviewed structures; {DATA["campaigns"]["B"]["completed"]}/20 background cases complete')
 fig.text(.01,.01,footer,fontsize=6.7,color='#555555',va='bottom')
 for ext in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',facecolor='white',metadata={'Creator':'MetalVASP Matplotlib'} if ext=='pdf' else {})
 FIGURES.append({'figure':name,'purpose':purpose,'evidence_class':evidence,'source_locators':locators,'files':{ext:hashlib.sha256((OUT/(name+'.'+ext)).read_bytes()).hexdigest() for ext in ['png','pdf','svg']}});plt.close(fig)
def categories(ax):ax.set(yticks=Y,yticklabels=NAMES,ylim=(N-.5,-.5));ax.grid(axis='x',alpha=.16)
def median(v):return float(np.median(v)) if v else np.nan
fig,axs=plt.subplots(1,2,figsize=(7.1,6.2),layout='constrained',gridspec_kw={'width_ratios':[1.4,1]})
values=[c['cpu_over_gpu'] for c in CASES]
axs[0].barh(Y,values,color=[GPU if v>=1 else BAD for v in values],height=.62)
for i,c in enumerate(CASES):
 axs[0].scatter([r['cpu_over_gpu'] for r in c['paired']],np.linspace(i-.12,i+.12,3),s=10,facecolors='white',edgecolors='#333333',lw=.5,zorder=3)
 axs[0].text(max([values[i]]+[r['cpu_over_gpu'] for r in c['paired']])+.10,i,f'{values[i]:.2f}×',fontsize=7,va='center')
axs[0].axvline(1,c='#333333',ls='--',lw=.8);categories(axs[0]);axs[0].set(xlabel='CPU time / GPU time (×)',title='a  Whole-program speedup',xlim=(0,max(values)*1.25))
for campaign,marker,label in [('A','o','Completed seven-case dataset'),('B','s','Completed background subset')]:
 subset=[c for c in CASES if c['campaign']==campaign]
 axs[1].scatter([c['atoms'] for c in subset],[c['cpu_over_gpu'] for c in subset],s=30,marker=marker,c=GPU if campaign=='A' else BLUE,label=label,zorder=3)
for name in ['C-2','Au-48','Si64','Si128']:
 c=next(x for x in CASES if x['system']==name);axs[1].annotate(name,(c['atoms'],c['cpu_over_gpu']),xytext=(5,-12) if name=='C-2' else (4,6),textcoords='offset points',fontsize=7)
axs[1].axhline(1,c='#333333',ls='--',lw=.8);axs[1].set(xlabel='Atoms',ylabel='CPU time / GPU time (×)',title='b  Chemistry and size coverage',xscale='log',ylim=(.5,max(values)*1.2));axs[1].set_xticks([2,4,8,16,32,64,128],labels=['2','4','8','16','32','64','128']);axs[1].grid(alpha=.15);axs[1].legend(frameon=False,fontsize=6.5,loc='upper left')
save(fig,'01-speed','CPU/GPU wall-time benefits retain small-case overhead and every paired repetition.',['measured.json#/cases'])
fig,axs=plt.subplots(2,3,figsize=(7.1,4.8),layout='constrained')
metrics=[('delta_energy_ev','Energy difference (eV)'),('delta_energy_per_atom_ev','Energy / atom difference (eV)'),('force_vector_rmse_ev_A','Force-vector RMSE (eV/Å)'),('force_component_max_ev_A','Max force-component difference (eV/Å)'),('stress_component_max_kbar','Max stress-component difference (kbar)'),('scf_sequence_max_delta_ev','Max full-SCF energy difference (eV)')]
formal=[p for p in DATA['precision'] if not p['warmup']]
for panel,(ax,(key,label)) in enumerate(zip(axs.flat,metrics)):
 for campaign,marker,color in [('A','o',GPU),('B','s',BLUE)]:
  p=[p for p in formal if p['campaign']==campaign];ax.scatter([r['atoms'] for r in p],[r[key] for r in p],s=18,marker=marker,facecolors='none',edgecolors=color,lw=.7,alpha=.8)
 limit=max(p[key] for p in formal);ax.set(xlabel='Atoms',ylabel=label,title=chr(97+panel)+'  CPU/GPU agreement',xscale='log');ax.set_yscale('symlog',linthresh=1e-10);ax.set_ylim(-1e-11,max(limit*2.3,1e-10));ax.grid(alpha=.12);ax.set_xticks([2,8,32,128],labels=['2','8','32','128'])
save(fig,'02-precision','All reviewed CPU/GPU electronic outputs pass the original numerical gates.',['measured.json#/precision'],'simulated')
fig,axs=plt.subplots(1,2,figsize=(7.1,6.2),layout='constrained',sharey=True)
for ax,key,title in [(axs[0],'timing','a  Whole-program elapsed time'),(axs[1],'scf_timing','b  SCF LOOP elapsed time')]:
 for mode,offset,color,label in [('cpu',-.12,CPU,'CPU'),('gpu',.12,GPU,'GPU')]:
  med=np.array([c[key][mode]['median_seconds'] for c in CASES]);lo=np.array([c[key][mode]['min_seconds'] for c in CASES]);hi=np.array([c[key][mode]['max_seconds'] for c in CASES]);ax.errorbar(med,Y+offset,xerr=[med-lo,hi-med],fmt='o',ms=3,color=color,capsize=2,lw=.7,label=label)
 categories(ax);ax.set(xscale='log',xlabel='Median and observed min–max (s)',title=title)
axs[0].legend(frameon=False,fontsize=7)
save(fig,'03-times','CPU/GPU benefits appear in absolute total and SCF elapsed times.',['measured.json#/cases/*/timing','measured.json#/cases/*/scf_timing'])
fig,axs=plt.subplots(1,2,figsize=(7.1,3.6),layout='constrained')
def resource(ax,key,mode,color,label,marker='o'):
 for c in CASES:
  v=[r[key]/1024**3 for r in DATA['memory'] if r['system']==c['system'] and r['mode']==mode and r[key] is not None and r[key]>0]
  if v:
   med=median(v);ax.errorbar(c['atoms'],med,yerr=[[med-min(v)],[max(v)-med]],fmt=marker,ms=3,color=color,lw=.7,capsize=2)
 ax.plot([],[],marker,color=color,label=label)
resource(axs[0],'rss_observed_peak_bytes','cpu',CPU,'CPU process RSS');resource(axs[0],'rss_observed_peak_bytes','gpu',GPU,'GPU process RSS');resource(axs[1],'metal_observed_peak_bytes','gpu',GPU,'Observed Metal allocations');resource(axs[1],'planned_nonlocal_device_workspace_peak_bytes','gpu',PURPLE,'Planned nonlocal workspace','s')
for ax,title in zip(axs,['a  Sampled process resident memory','b  Separate device resource scopes']):
 ax.set(xscale='log',yscale='log',xlabel='Atoms',ylabel='Median peak observation (GiB)',title=title);ax.set_xticks([2,8,32,128],labels=['2','8','32','128']);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=6.5)
save(fig,'04-memory','GPU resource observations remain distinct from process RSS and operation plans.',['measured.json#/memory'])
STAGES=['EDDAV','FORNL','STRESS','POTLOK','LOOP'];acc={}
for c in CASES:
 for mode in ['cpu','gpu']:
  for stage in STAGES:acc[c['system'],mode,stage]=median([r['wall_seconds'] for r in DATA['source_stages'] if r['system']==c['system'] and r['mode']==mode and r['stage']==stage])
matrix=np.array([[acc[c['system'],'cpu',s]/acc[c['system'],'gpu',s] if acc[c['system'],'gpu',s]>0 else np.nan for s in STAGES] for c in CASES])
fig,ax=plt.subplots(figsize=(7.1,6.1),layout='constrained');cmap=plt.get_cmap('BrBG').copy();cmap.set_bad('#D8DDE0');im=ax.imshow(np.log2(matrix),aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=5))
for i in range(N):
 for j in range(len(STAGES)):
  v=matrix[i,j];ax.text(j,i,f'{v:.2f}×' if np.isfinite(v) else 'NA',ha='center',va='center',fontsize=7,color='white' if np.isfinite(v) and (v>6 or v<.4) else '#111111')
ax.set(yticks=Y,yticklabels=NAMES,xticks=np.arange(len(STAGES)),xticklabels=STAGES,title='CPU / GPU source-stage wall-time ratios');bar=fig.colorbar(im,ax=ax,shrink=.75,pad=.03);bar.set_ticks([-2,-1,0,1,2,3,4,5],labels=['0.25×','0.5×','1×','2×','4×','8×','16×','32×']);bar.set_label('CPU / GPU median stage time')
save(fig,'05-stages','CPU/GPU acceleration varies across calculation stages and eligible layouts.',['measured.json#/source_stages'])
fig,axs=plt.subplots(1,2,figsize=(7.1,3.9),layout='constrained');timers=[('api_wall_seconds','#333333','API wall'),('process_cpu_seconds',PURPLE,'Process CPU'),('gpu_command_seconds',GPU,'GPU command'),('host_wait_seconds','#E69F00','Host wait'),('conversion_instrumented_seconds',BLUE,'Conversion')];ops=['gemm','fft','hamiltonian','orth_overlap','nonlocal_projection']
for ax,system in zip(axs,['Si64','Si128']):
 for i,(key,col,label) in enumerate(timers):
  v=[median([r[key] for r in DATA['operators'] if r['system']==system and r['mode']=='gpu' and r['operator']==op]) for op in ops];ax.bar(np.arange(len(ops))+(i-2)*.15,v,.14,color=col,label=label)
 ax.set(xticks=np.arange(len(ops)),xticklabels=['GEMM','FFT','Hamiltonian','Overlap','Nonlocal'],yscale='log',ylabel='Distinct timer median (s)',title=system+' GPU-route timers');ax.tick_params(axis='x',rotation=35)
fig.legend(*axs[0].get_legend_handles_labels(),frameon=False,fontsize=6.5,ncol=5,loc='upper center',bbox_to_anchor=(.5,.995))
save(fig,'06-api-timers','GPU-route costs include arithmetic, host conversions and waits.',['measured.json#/operators'])
fig,axs=plt.subplots(1,2,figsize=(7.1,6.2),layout='constrained',gridspec_kw={'width_ratios':[1,1.3]})
for campaign,marker,color,label in [('A','o',GPU,'Dataset A'),('B','s',BLUE,'Dataset B')]:
 chosen=[c for c in CASES if c['campaign']==campaign];vals=[median([r['wa_import_bytes']/1e9 for r in DATA['memory'] if r['system']==c['system'] and r['mode']=='gpu' and r['wa_import_bytes'] is not None]) for c in chosen];axs[0].scatter([c['atoms'] for c in chosen],vals,s=25,c=color,marker=marker,label=label)
axs[0].set(xscale='log',yscale='log',xlabel='Atoms',ylabel='Cumulative wave-basis ABI imports (GB)',title='a  GPU-route import counters');axs[0].set_xticks([2,8,32,128],labels=['2','8','32','128']);axs[0].grid(alpha=.15);axs[0].legend(frameon=False,fontsize=7)
for i,c in enumerate(CASES):
 rows=[r for r in DATA['memory'] if r['system']==c['system'] and r['mode']=='gpu'];success=median([r['nonlocal_success'] for r in rows if r['nonlocal_success'] is not None]);calls=median([r['nonlocal_calls'] for r in rows if r['nonlocal_calls'] is not None]);axs[1].barh(i,calls if np.isfinite(calls) else 0,height=.6,color='#C7CDD1');axs[1].barh(i,success if np.isfinite(success) else 0,height=.6,color=GPU);axs[1].text(10.5,i,'CPU fallback' if c['spin']==2 else (f'{int(success)}/{int(calls)}' if np.isfinite(success) else 'NA'),fontsize=6.8,va='center',color=CPU if c['spin']==2 else '#333333')
categories(axs[1]);axs[1].set(xlim=(0,19),xlabel='Completed / entered nonlocal GPU calls',title='b  Explicit projection eligibility')
save(fig,'07-reuse-dispatch','The current GPU release records communication and preserves explicit CPU fallback.',['measured.json#/memory'])
captions={c['figure']:c for c in json.loads((ROOT/'benchmarks/figure-captions.json').read_text())}
for f in FIGURES:f.update({k:v for k,v in captions[f['figure']].items() if k!='figure'})
manifest={'schema':'metalvasp-figures-v2','comparison':'CPU versus released GPU path','data_sha256':hashlib.sha256((ROOT/'benchmarks/data/measured.json').read_bytes()).hexdigest(),'dataset_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'benchmarks/data').iterdir() if p.is_file()},'figures':FIGURES}
(OUT/'figure-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(OUT/'figure-specs.json').write_text(json.dumps({'print_width_mm':180,'data_rendering':'Matplotlib, no image model','figures':FIGURES},ensure_ascii=False,indent=2)+'\n')
print('Generated',len(FIGURES),'CPU/GPU figure families in PNG/PDF/SVG')
