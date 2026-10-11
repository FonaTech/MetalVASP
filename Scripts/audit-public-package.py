#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit source-only release boundaries and local Markdown links; no network."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shlex
import sys

def audit_discovery_command(command, entries):
    """Check exact archive spelling even on a case-insensitive filesystem."""
    args=shlex.split(command,comments=True)
    start='.';pattern='test*.py'
    for flags,default in [(('-s','--start-directory'),start),(('-p','--pattern'),pattern)]:
        value=default
        for i,arg in enumerate(args):
            if arg in flags:
                if i+1==len(args):return ['unittest discovery option is missing its value']
                value=args[i+1]
            elif any(arg.startswith(flag+'=') for flag in flags):value=arg.split('=',1)[1]
        if flags[0]=='-s':start=posixpath.normpath(value)
        else:pattern=value
    if start!='.' and entries.get(start)!='directory':
        return ['unittest start directory missing or incorrectly capitalized: '+start]
    prefix='' if start=='.' else start+'/'
    matches=[name for name,kind in entries.items() if kind=='file' and name.startswith(prefix)
             and '/' not in name[len(prefix):] and fnmatch.fnmatchcase(name[len(prefix):],pattern)]
    return [] if matches else ['unittest discovery selects no test files: '+start+'/'+pattern]

def audit(root):
    errors=[];count=0
    entries={p.relative_to(root).as_posix():('directory' if p.is_dir() else 'file')
             for p in root.rglob('*') if not any(n in {'.git','__pycache__'} for n in p.relative_to(root).parts)}
    denied_dirs={'licensed-input','input-build','input-bin','original-binaries','original-build'}
    denied_names={'POTCAR','OUTCAR','WAVECAR','CHGCAR','main.F','makefile.include'}
    denied_suffixes={'.o','.a','.dylib','.so','.metallib','.mod','.tgz','.gz','.zip'}
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        if '.git' in rel.parts or '__pycache__' in rel.parts:continue
        if p.is_symlink():errors.append(str(rel)+': symlink');continue
        if not p.is_file():continue
        count+=1
        if any(d in denied_dirs or d.startswith('build-') for d in rel.parts[:-1]) or p.name in denied_names or p.suffix in denied_suffixes:
            errors.append(str(rel)+': prohibited private/build payload')
        b=p.read_bytes()
        if b[:4] in [b'\x7fELF',b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xca\xfe\xba\xbe'] or b.startswith(b'!<arch>'):
            errors.append(str(rel)+': native binary')
        if p.suffix not in {'.png','.pdf'}:
            t=b.decode('utf-8',errors='replace')
            if re.search(r'/(?:Users/[A-Za-z0-9_.-]+|private/var/folders)/',t):errors.append(str(rel)+': private absolute path')
            if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{40,}',t):errors.append(str(rel)+': possible credential')
            if p.suffix=='.md':
                for link in re.findall(r'\]\(([^)]+)\)',t):
                    target=link.split('#')[0].split(' ')[0]
                    if not target or '://'in target or target.startswith('mailto:'):continue
                    if target.startswith('/') or not (p.parent/target).exists():errors.append(str(rel)+': missing/absolute local link '+target)
    required=['LICENSE','NOTICE.md','README.md','README.zh-CN.md','release-validation.json','Scripts/metalvasp.py',
              'adapters/vasp-6.4.1.json','Tests/test_installer.py','.github/workflows/installer.yml']
    for n in required:
        if entries.get(n)!='file':errors.append(n+': required file missing or incorrectly capitalized')
    workflow=root/'.github/workflows/installer.yml'
    if workflow.is_file():
        commands=re.findall(r'^\s*-\s+run:\s*(python\S*\s+-m\s+unittest\s+discover[^\n]*)',workflow.read_text(),re.M)
        if not commands:errors.append('installer workflow: missing unittest discovery command')
        for command in commands:errors.extend(audit_discovery_command(command,entries))
    # Match the unchanged reviewed backend, independently of the packaging list.
    provenance=root/'adapters/backend-provenance.json'
    if provenance.is_file():
        for n,h in json.loads(provenance.read_text())['backend_sha256'].items():
            if not (root/n).is_file() or hashlib.sha256((root/n).read_bytes()).hexdigest()!=h:errors.append(n+': backend provenance mismatch')
    return {'status':'passed'if not errors else'failed','files_checked':count,'errors':errors,
            'scope':'Mechanical source-only/privacy/link/hash audit; not a legal ownership opinion or native validation.'}

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('root',nargs='?',default='.');a.add_argument('--json')
    x=a.parse_args();result=audit(Path(x.root).resolve())
    if x.json:Path(x.json).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));sys.exit(0 if result['status']=='passed'else 1)
