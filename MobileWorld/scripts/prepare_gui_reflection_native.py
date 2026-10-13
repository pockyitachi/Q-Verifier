#!/usr/bin/env python3
"""Fetch only official GUI-Reflection Action Verification screenshots.

Reads the central directory of the official multi-part GUIOdyssey ZIP with HTTP
Range requests, then downloads the requested members. Never fetches full parts.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import time
import zlib

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/datasets/gui_reflection_task_suite'


class Archive:
    def __init__(self, metadata):
        self.repo = metadata['repo']
        self.revision = metadata['revision']
        self.parts = sorted(metadata['parts'], key=lambda p: p['path'])
        self.client = httpx.Client(follow_redirects=True, timeout=90,
                                  limits=httpx.Limits(max_connections=12))
        self.urls = {}
        self.lock = threading.Lock()
        self.transferred = 0

    def read(self, disk, offset, length):
        data = []
        while length:
            part = self.parts[disk]
            amount = min(length, part['size'] - offset)
            if amount <= 0:
                raise ValueError('Invalid ZIP disk offset')
            base_url = f'https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{part["path"]}?download=true'
            expected = f'bytes {offset}-{offset + amount - 1}/{part["size"]}'
            for attempt in range(6):
                try:
                    with self.lock:
                        url = self.urls.get(disk, base_url)
                    with self.client.stream('GET', url, headers={'Range': f'bytes={offset}-{offset + amount - 1}'}) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('content-range') != expected:
                            raise RuntimeError('Server did not honor the exact Range; refusing full transfer')
                        content = response.read()
                        if len(content) != amount:
                            raise RuntimeError('Incomplete range response')
                        with self.lock:
                            self.urls[disk] = str(response.url)
                            self.transferred += len(content)
                        data.append(content)
                        break
                except Exception as error:
                    with self.lock:
                        self.urls.pop(disk, None)
                    if attempt == 5:
                        raise RuntimeError(f'Range read failed for disk {disk}, offset {offset}') from error
                    time.sleep(min(2**attempt, 16))
            length -= amount
            offset = 0
            disk += 1
        return b''.join(data)


def parse_central(data):
    entries = []
    position = 0
    while position < len(data):
        fields = struct.unpack_from('<4s6H3I5H2I', data, position)
        if fields[0] != b'PK\x01\x02':
            raise ValueError(f'Invalid ZIP directory signature at {position}')
        _, _, _, flags, method, _, _, crc, compressed, size, fn, ex, comment, disk, _, _, offset = fields
        name_bytes = data[position + 46:position + 46 + fn]
        name = name_bytes.decode('utf-8' if flags & 0x800 else 'cp437')
        extra = data[position + 46 + fn:position + 46 + fn + ex]
        cursor = 0
        while cursor < len(extra):
            kind, count = struct.unpack_from('<HH', extra, cursor)
            value = extra[cursor + 4:cursor + 4 + count]
            if kind == 1:
                j = 0
                if size == 0xffffffff:
                    size, = struct.unpack_from('<Q', value, j); j += 8
                if compressed == 0xffffffff:
                    compressed, = struct.unpack_from('<Q', value, j); j += 8
                if offset == 0xffffffff:
                    offset, = struct.unpack_from('<Q', value, j); j += 8
                if disk == 0xffff:
                    disk, = struct.unpack_from('<I', value, j)
            cursor += 4 + count
        entries.append({'name': name, 'disk': disk, 'offset': offset,
                        'compressed_size': compressed, 'size': size, 'crc32': crc,
                        'method': method, 'flags': flags,
                        'name_length': fn, 'central_extra_length': ex})
        position += 46 + fn + ex + comment
    return entries


def main():
    metadata = json.loads((DEST / 'archive_parts.json').read_text())
    archive = Archive(metadata)
    annotation = DEST / 'action_verification.jsonl'
    rows = [json.loads(line) for line in annotation.read_text().splitlines() if line.strip()]
    assert len(rows) == 603
    wanted = {name for row in rows for name in row['images']}
    target = DEST / 'images'
    target.mkdir(exist_ok=True)
    index_path = DEST / 'selected_zip_members.json'
    if index_path.exists():
        selected = json.loads(index_path.read_text())
    else:
        end = (DEST / 'zip_end.bin').read_bytes()
        position = end.rfind(b'PK\x06\x06')
        if position < 0:
            raise ValueError('Expected a ZIP64 end record')
        _, _, _, _, _, directory_disk, _, total, length, offset = struct.unpack_from('<4sQ2H2I4Q', end, position)
        print(f'Reading {length / 1e6:.1f} MB ZIP index for {total} members', flush=True)
        entries = parse_central(archive.read(directory_disk, offset, length))
        assert len(entries) == total
        selected = [entry for entry in entries if Path(entry['name']).name in wanted]
        if Counter(Path(e['name']).name for e in selected) != Counter(wanted):
            missing = wanted - {Path(e['name']).name for e in selected}
            raise RuntimeError(f'Missing or duplicate members: {sorted(missing)[:10]}')
        index_path.write_text(json.dumps(selected, indent=2) + '\n')
    print(f'Downloading {len(selected)} PNGs; compressed payload {sum(e["compressed_size"] for e in selected) / 1e6:.1f} MB', flush=True)
    old = json.loads((DEST / 'source_image_files.json').read_text())
    old_hashes = {Path(f['path']).name: f['sha256'] for f in old['files']}

    def get_image(entry):
        name = Path(entry['name']).name
        if name not in wanted or entry['flags'] & 1:
            raise RuntimeError('Unexpected or encrypted ZIP member')
        path = target / name
        if path.exists():
            image_bytes = path.read_bytes()
        else:
            request_size = entry['compressed_size'] + 30 + entry['name_length'] + entry['central_extra_length'] + 128
            raw = archive.read(entry['disk'], entry['offset'], request_size)
            header = struct.unpack_from('<4s5H3I2H', raw)
            if header[0] != b'PK\x03\x04' or header[3] != entry['method']:
                raise RuntimeError(f'ZIP member header mismatch: {name}')
            fn, ex = header[-2:]
            start = 30 + fn + ex
            if raw[30:30 + fn].decode('utf-8' if entry['flags'] & 0x800 else 'cp437') != entry['name']:
                raise RuntimeError(f'ZIP member filename mismatch: {name}')
            if start + entry['compressed_size'] > len(raw):
                raw = archive.read(entry['disk'], entry['offset'], start + entry['compressed_size'])
            payload = raw[start:start + entry['compressed_size']]
            if entry['method'] == 8:
                image_bytes = zlib.decompress(payload, -15)
            elif entry['method'] == 0:
                image_bytes = payload
            else:
                raise RuntimeError(f'Unsupported ZIP compression: {entry["method"]}')
        if len(image_bytes) != entry['size'] or zlib.crc32(image_bytes) != entry['crc32']:
            raise RuntimeError(f'ZIP integrity mismatch: {name}')
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            dimensions = list(image.size)
        digest = hashlib.sha256(image_bytes).hexdigest()
        expected = old_hashes.get(name)
        if expected is not None and expected != digest:
            raise RuntimeError(f'PNG differs from the original repository: {name}')
        if not path.exists():
            temp = path.with_suffix('.png.tmp')
            temp.write_bytes(image_bytes)
            temp.replace(path)
        return {'name': name, 'size': len(image_bytes), 'sha256': digest,
                'crc32': entry['crc32'], 'dimensions': dimensions,
                'original_repo_hash_match': True if expected else None}

    verified = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(get_image, entry) for entry in selected]
        for future in as_completed(futures):
            verified.append(future.result())
            if len(verified) % 50 == 0 or len(verified) == len(selected):
                print(f'Images verified: {len(verified)}/{len(selected)}; network {archive.transferred / 1e6:.1f} MB', flush=True)
                (DEST / 'download_status.json').write_text(json.dumps({'completed': len(verified), 'total': len(selected), 'network_bytes': archive.transferred}) + '\n')
    manifest = {'benchmark_repo': 'craigwu/GUI_Reflection_Task_Suite_Benchmark',
                'benchmark_revision': '4560c902c947bdd940dd9e6bc42f697faa73b85a',
                'annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
                'records': len(rows), 'judgments': len(rows) * 2, 'image_count': len(verified),
                'image_source_repo': archive.repo, 'image_source_revision': archive.revision,
                'original_repo_sha256_matches': sum(v['original_repo_hash_match'] is True for v in verified),
                'images': sorted(verified, key=lambda x: x['name'])}
    (DEST / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('ALL DATA READY AND VERIFIED', flush=True)


if __name__ == '__main__':
    main()
