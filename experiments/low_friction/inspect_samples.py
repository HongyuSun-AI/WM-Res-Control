import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_samples import construct
from inspect_recording import require, validate


def inspect(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    stats = manifest['normalization']
    ids = [s['episode_id'] for s in manifest['sources']]
    require(len(ids) == len(set(ids)), 'Episode reused across sources')
    for split in ('train', 'validation'):
        sources = [s for s in manifest['sources'] if s['split'] == split]
        if not sources:
            continue
        with np.load(directory/(split+'.npz'), allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
            offset = 0
            for source in sources:
                path = Path(source['path'])
                if not path.is_absolute(): path = directory/path
                require(hashlib.sha256((path/'transitions.jsonl').read_bytes()).hexdigest() == source['sha256'], 'Source changed')
                meta, rows = validate(path)
                require(meta['episode_id'] == source['episode_id'], 'Source episode changed')
                expected = construct(rows, manifest['history_length'])
                n = len(expected['action'])
                require(n == source['samples'], 'Sample count mismatch')
                for key, value in expected.items():
                    require(np.array_equal(data[key][offset:offset+n], value), 'Source alignment failed: '+key)
                require(np.all(data['episode_id'][offset:offset+n] == source['episode_id']), 'Episode id mismatch')
                offset += n
            require(offset == len(data['action']), 'Unexpected sample count')
            for name in ('history', 'action', 'target'):
                if name == 'history':
                    mean = stats['state']['mean']+stats['action']['mean']
                    scale = stats['state']['scale']+stats['action']['scale']
                else:
                    mean, scale = stats[name]['mean'], stats[name]['scale']
                normalized = data[name+'_normalized']
                require(np.isfinite(normalized).all(), 'Nonfinite normalized values')
                require(np.allclose(normalized*np.asarray(scale)+np.asarray(mean), data[name], atol=2e-6),
                        'Normalization round trip failed')
            print('PASS {}: {} aligned samples'.format(split, offset))
            for index in sorted(set((0, offset//2, offset-1))):
                print('sample={};step={};history_frames={};action_frame={} next_frame={} action={} target={}'.format(
                    index, data['step'][index], data['history_frames'][index].tolist(), *data['frames'][index],
                    data['action'][index].tolist(), data['target'][index].tolist()))
    print('validation_available={}'.format(manifest['validation_available']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    inspect(parser.parse_args().directory)
