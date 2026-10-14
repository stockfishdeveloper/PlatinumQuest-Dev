"""Verified source archives, avoiding thousands of loose experiment copies.

New experiments can call snapshot_sources instead of copying a source tree.
The cleanup CLI only packs untracked source_snapshot/source_snapshots directories
under the two generated-artifact roots. It never deletes active source or data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


MANIFEST = '__snapshot_manifest__.json'
ROOT = Path(__file__).resolve().parents[2]
SCOPES = ('ml_agent/dynamics_reports', 'ml_agent/datasets/marble_dynamics')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read(MANIFEST))
        if set(archive.namelist()) != set(manifest['files']) | {MANIFEST}:
            raise ValueError('Archive members disagree with manifest')
        for name, record in manifest['files'].items():
            content = archive.read(name)
            if len(content) != record['bytes'] or digest(content) != record['sha256']:
                raise ValueError('Source archive hash mismatch: ' + name)
    return manifest


def write_archive(destination, sources, base):
    """Commit and reread every byte before returning; refuse existing evidence."""
    destination, base = Path(destination), Path(base).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    sources = sorted({Path(p).resolve() for p in sources})
    manifest = dict(schema='verified_source_archive_v1', original_base=str(base), files={})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                name = path.relative_to(base).as_posix()
                if name == MANIFEST:
                    raise ValueError('Reserved source filename')
                content = path.read_bytes()
                manifest['files'][name] = dict(bytes=len(content), sha256=digest(content))
                archive.writestr(name, content)
            archive.writestr(MANIFEST, json.dumps(manifest, indent=2))
        verify_archive(temporary)
        # The archive is local historical evidence, never executable source.
        os.rename(temporary, destination)
        verify_archive(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def snapshot_sources(output, paths, root=ROOT):
    return write_archive(Path(output) / 'source_snapshot.zip', paths, root)


def pack_snapshot(folder, root=ROOT, tracked=()):
    """Delete only individually verified archived files within a checked scope."""
    root, folder = Path(root).resolve(), Path(folder).resolve()
    relative = folder.relative_to(root).as_posix()
    if folder.name not in ('source_snapshot', 'source_snapshots') or not any(
            relative.startswith(scope + '/') for scope in SCOPES):
        raise ValueError('Outside generated source-snapshot scopes: ' + str(folder))
    paths = sorted(p for p in folder.rglob('*') if p.is_file())
    for path in paths:
        path.resolve().relative_to(folder)
        if path.is_symlink() or path.relative_to(root).as_posix() in tracked:
            raise ValueError('Refusing linked/tracked snapshot: ' + str(path))
    archive = folder.with_suffix('.zip')
    manifest = write_archive(archive, paths, folder)
    # Verify the entire original set before deleting any member. A changed file
    # aborts, leaving the originals and archive available for inspection.
    for path in paths:
        if digest(path.read_bytes()) != manifest['files'][path.relative_to(folder).as_posix()]['sha256']:
            raise ValueError('Original changed during archive: ' + str(path))
    for path in paths:
        path.unlink()
    for directory in sorted((p for p in folder.rglob('*') if p.is_dir()), reverse=True):
        directory.resolve().relative_to(folder)
        directory.rmdir()
    folder.rmdir()
    return dict(original_directory=relative, archive=archive.relative_to(root).as_posix(),
                files_removed=len(paths), python_files_removed=sum(p.suffix == '.py' for p in paths),
                original_bytes=sum(r['bytes'] for r in manifest['files'].values()),
                archive_bytes=archive.stat().st_size, archive_sha256=digest(archive.read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Archive, verify, then remove generated loose copies')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    folders = sorted(p for scope in SCOPES for p in (ROOT / scope).rglob('*')
                     if p.is_dir() and p.name in ('source_snapshot', 'source_snapshots'))
    report = dict(schema='source_cleanup_v1', apply=args.apply,
                  planned=[p.relative_to(ROOT).as_posix() for p in folders], completed=[])
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    if args.apply:
        for folder in folders:
            report['completed'].append(pack_snapshot(folder, tracked=tracked))
            save()
    print(json.dumps(dict(planned=len(folders), completed=len(report['completed']),
                         python_files_removed=sum(r['python_files_removed'] for r in report['completed']),
                         report=str(args.report))))


if __name__ == '__main__':
    main()
