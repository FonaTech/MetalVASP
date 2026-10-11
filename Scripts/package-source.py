#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Produce deterministic source-only ZIP/tar.gz archives after a boundary audit."""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

def package(root, output):
    root=root.resolve(); output=output.resolve()
    if root==output or root in output.parents:
        raise ValueError('Archive output must be outside the source tree')
    spec=importlib.util.spec_from_file_location('public_audit',root/'Scripts/audit-public-package.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    result=audit.audit(root)
    if result['status']!='passed':raise ValueError(json.dumps(result))
    output.mkdir(parents=True,exist_ok=True)
    files=[p for p in sorted(root.rglob('*'))if p.is_file()and not any(n in ['.git','__pycache__','.DS_Store']for n in p.relative_to(root).parts)]
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files if p.name!='MANIFEST.sha256.json'}
    (root/'MANIFEST.sha256.json').write_text(json.dumps({'schema':'metalvasp-package-sha256-v1','files':manifest},indent=2)+'\n')
    if root/'MANIFEST.sha256.json' not in files:files.append(root/'MANIFEST.sha256.json')
    files.sort(); prefix=root.name
    stamp=1791676800
    paths=[output/(prefix+'.zip'),output/(prefix+'.tar.gz')]
    with zipfile.ZipFile(paths[0],'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9)as z:
        for p in files:
            info=zipfile.ZipInfo(prefix+'/'+str(p.relative_to(root)),date_time=(2026,10,11,0,0,0))
            info.create_system=3;info.compress_type=zipfile.ZIP_DEFLATED
            mode=0o755 if p.stat().st_mode&0o111 else 0o644
            info.external_attr=(0o100000|mode)<<16
            z.writestr(info,p.read_bytes(),compresslevel=9)
    with paths[1].open('wb')as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=stamp,compresslevel=9)as gz,tarfile.open(fileobj=gz,mode='w')as t:
        for p in files:
            info=tarfile.TarInfo(prefix+'/'+str(p.relative_to(root)));info.size=p.stat().st_size
            info.mode=0o755 if p.stat().st_mode&0o111 else 0o644
            info.mtime=stamp;info.uid=info.gid=0;info.uname=info.gname=''
            t.addfile(info,io.BytesIO(p.read_bytes()))
    summary={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in paths}
    (output/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+n+'\n'for n,v in summary.items()))
    (output/'package-review.json').write_text(json.dumps({'status':'packaged','files':len(files),'archives':summary,'boundary_audit':result},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return paths

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();package(Path(__file__).resolve().parents[1],a.output)
