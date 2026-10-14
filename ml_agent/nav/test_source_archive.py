"""Deletion guards and byte preservation for historical source packing."""
import tempfile
from pathlib import Path
import unittest
import zipfile

from nav.source_archive import pack_snapshot, verify_archive, write_archive


class SourceArchiveTests(unittest.TestCase):
    def fixture(self, root):
        folder = root / 'ml_agent/dynamics_reports/example/source_snapshot'
        folder.mkdir(parents=True)
        source = folder / 'nested/script.py'
        source.parent.mkdir()
        source.write_bytes(b'print("past version")\r\n')
        (folder / 'native.cs').write_bytes(b'// preserve exact bytes\n')
        return folder, source

    def test_archive_is_complete_before_loose_copies_are_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, source = self.fixture(root)
            content = source.read_bytes()
            result = pack_snapshot(folder, root=root)
            self.assertEqual((result['files_removed'], result['python_files_removed']), (2, 1))
            self.assertFalse(folder.exists())
            archive = root / result['archive']
            self.assertEqual(len(verify_archive(archive)['files']), 2)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('nested/script.py'), content)

    def test_active_source_and_tracked_snapshot_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, source = self.fixture(root)
            with self.assertRaises(ValueError):
                pack_snapshot(root / 'ml_agent/nav', root=root)
            with self.assertRaises(ValueError):
                pack_snapshot(folder, root=root, tracked={source.relative_to(root).as_posix()})
            self.assertTrue(source.exists())
            self.assertFalse(folder.with_suffix('.zip').exists())

    def test_existing_archive_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, source = self.fixture(root)
            archive = folder.with_suffix('.zip')
            archive.write_bytes(b'existing evidence')
            with self.assertRaises(FileExistsError):
                pack_snapshot(folder, root=root)
            self.assertTrue(source.exists())
            self.assertEqual(archive.read_bytes(), b'existing evidence')

    def test_outside_source_aborts_without_committing_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, source = self.fixture(root)
            other = root / 'outside.py'
            other.write_bytes(b'important')
            with self.assertRaises(ValueError):
                write_archive(root / 'packed.zip', [source, other], folder)
            self.assertFalse((root / 'packed.zip').exists())
            self.assertFalse((root / 'packed.zip.tmp').exists())
            self.assertTrue(source.exists())


if __name__ == '__main__':
    unittest.main()
