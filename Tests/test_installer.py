# SPDX-License-Identifier: MIT
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

spec = importlib.util.spec_from_file_location('installer', Path(__file__).resolve().parents[1] / 'Scripts/metalvasp.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class InstallerSecurity(unittest.TestCase):
    def archive(self, folder, name, entries):
        p = folder / name
        with tarfile.open(p, 'w:gz') as t:
            for path, payload, kind in entries:
                info = tarfile.TarInfo(path)
                if kind == 'file':
                    info.size = len(payload); t.addfile(info, io.BytesIO(payload))
                else:
                    info.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
                    info.linkname = payload; t.addfile(info)
        return p

    def test_tar_traversal_links_absolute_and_duplicates_write_nothing(self):
        for entries in [[('../escape', b'x', 'file')], [('/tmp/escape', b'x', 'file')],
                        [('safe', '/tmp', 'symlink')], [('safe', 'other', 'hardlink')],
                        [('file', b'a', 'file'), ('file', b'b', 'file')]]:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as td:
                d = Path(td); p = self.archive(d, 'input.tgz', entries)
                with self.assertRaises(m.SetupError): m.extract(p, d/'output')
                self.assertFalse((d/'output').exists())

    def test_zip_symlink_and_traversal_rejected_before_write(self):
        for name, mode in [('../escape', 0o100644), ('link', 0o120777), ('C:/escape', 0o100644), ('a\\b', 0o100644)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                d=Path(td); p=d/'a.zip'
                with zipfile.ZipFile(p,'w') as z:
                    entry=zipfile.ZipInfo(name);entry.external_attr=mode<<16;z.writestr(entry,'secret')
                with self.assertRaises(m.SetupError):m.extract(p,d/'out')
                self.assertFalse((d/'out').exists())

    def test_valid_archive_with_spaces_and_shell_metacharacters_is_data(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);p=self.archive(d, 'source $(touch PWNED).tgz', [('vasp/src/main.F',b'placeholder\n','file')])
            m.extract(p,d/'private copy')
            self.assertEqual(m.locate_source(d/'private copy'),d/'private copy/vasp')
            self.assertFalse((d/'PWNED').exists())

    def test_size_budget_before_extraction(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(m,'MAX_BYTES',2):
            d=Path(td);p=self.archive(d,'s.tgz',[('file',b'123','file')])
            with self.assertRaises(m.SetupError):m.extract(p,d/'out')
            self.assertFalse((d/'out').exists())

    def test_directory_link_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);s=d/'src';s.mkdir();(s/'link').symlink_to('/tmp')
            with self.assertRaises(m.SetupError):m.extract(s,d/'out')
            self.assertFalse((d/'out').exists())

    def test_recipe_rejects_corrupt_input_output_and_range(self):
        base=b'original licensed text'; rule={'path':'fake','input_sha256':m.digest(base),'output_sha256':m.digest(base+b' own insertion'),'operations':[{'copy':[0,len(base)]},{'insert':' own insertion'}]}
        self.assertEqual(m.reconstruct(base,rule),base+b' own insertion')
        with self.assertRaises(m.SetupError):m.reconstruct(base+b'x',rule)
        wrong=dict(rule,output_sha256='0'*64)
        with self.assertRaises(m.SetupError):m.reconstruct(base,wrong)
        wrong=dict(rule,operations=[{'copy':[-1,2]}])
        with self.assertRaises(m.SetupError):m.reconstruct(base,wrong)

    def test_metadata_version_does_not_trust_archive_name(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'vasp.6.4.1';(r/'src').mkdir(parents=True)
            with self.assertRaises(m.SetupError):m.source_version(r)
            (r/'src/version.F').write_text('integer, parameter :: major = 6\ninteger, parameter :: minor = 6\ninteger, parameter :: patch = 0\n')
            self.assertEqual(m.source_version(r),'6.6.0')
            with self.assertRaisesRegex(m.SetupError,'inspection only'):m.prepare_patches(r)

    def test_make_expansion_and_shell_injection_rejected(self):
        for s in ['/tmp/a b','/tmp/$USER','/tmp/a#b','/tmp/`date`','/tmp/x;true','/tmp/a\nX=1','/tmp/x\\y']:
            with self.subTest(s=s), self.assertRaises(m.SetupError):m.make_value(s)
        self.assertEqual(m.make_value('/tmp/normal-path'),'/tmp/normal-path')

    def test_job_heuristic_obeys_memory_and_cores(self):
        d={'cpu_count':32,'device':{'performance_cores':'12','memory_bytes':str(8*1024**3)}}
        self.assertEqual(m.choose_jobs(d),4)
        d['device']['memory_bytes']=str(128*1024**3)
        self.assertEqual(m.choose_jobs(d),12)

    def test_existing_work_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            args=mock.Mock(work=td,source=None)
            env={'build_platform_supported':True,'missing_tools':[]}
            with self.assertRaisesRegex(m.SetupError,'never overwritten'):m.build(args,env)


if __name__ == '__main__':unittest.main()
