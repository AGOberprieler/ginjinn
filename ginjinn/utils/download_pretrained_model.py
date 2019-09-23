import requests
import math
import os
from os import path
import tarfile
from tqdm import tqdm
import argparse
import sys
from shutil import rmtree

from ginjinn.core.tf_model import PRETRAINED_MODEL_URLS

AVAILALE_PRETRAINED_MODELS = list(filter(lambda x: not x.endswith('.config'), PRETRAINED_MODEL_URLS.keys()))


# TODO: (low priority) rework this to match new style
def download_and_extract_pretrained_model(url, out_dir, block_size=1024, rm=False, force=False):
    destination = path.abspath(path.join(out_dir, path.basename(url)))
    ckpt_dir = destination[:-7]

    if os.path.exists(ckpt_dir) and not force:
        print('Checkpoint already available at "{}"... download not necessary.'.format(ckpt_dir))
        return ckpt_dir

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    wrote = 0

    print('Downloading pretrained model from "{}"...'.format(url))
    with open(destination, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        msg = 'ERROR: Could not download pretrained model from "{}"'.format(url)
        raise Exception(msg)
    print('Finished downloading pretrained model from "{}"'.format(url))
    
    print('Extracting...')
    with tarfile.open(destination, 'r:gz') as f:
        f.extractall(out_dir)
    
    if rm:
        os.remove(destination)
    print('Done')

    return ckpt_dir

def download_pretrained_model(model, out_dir):
    pretrained_model_url = PRETRAINED_MODEL_URLS.get(model, None)
    if pretrained_model_url is None:
        msg = 'ERROR: No pretrained model available for config "{}".\nPretrained models are available for following configs:\n{}'.format(
            args.config_name,
            '\n'.join('\t"{}"'.format(c) for c in AVAILALE_PRETRAINED_MODELS),
        )
        print(msg, file=sys.stderr)
        sys.exit()

    ckpt_dir = download_and_extract_pretrained_model(
        pretrained_model_url,
        out_dir
    )

    print(f'Successfully downloaded pretrained model. Location: "{ckpt_dir}"')


def main():
    parser = argparse.ArgumentParser(
        description='Download pretrained model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'config_name',
        metavar='config_name',
        type=str,
        help='Name of the pipeline config file.\nPretrained models are available for following configs:\n{}'.format(
            '\n'.join('\t"{}"'.format(c) for c in AVAILALE_PRETRAINED_MODELS),
        )
    )
    parser.add_argument(
        '-d', '--out_dir',
        metavar='out_dir',
        type=str,
        help='Directory to which the pretrained model will be written.'
    )

    args = parser.parse_args()

    pretrained_model_url = PRETRAINED_MODEL_URLS.get(args.config_name, None)
    if pretrained_model_url is None:
        msg = 'ERROR: No pretrained model available for config "{}".\nPretrained models are available for following configs:\n{}'.format(
            args.config_name,
            '\n'.join('\t"{}"'.format(c) for c in AVAILALE_PRETRAINED_MODELS),
        )
        print(msg, file=sys.stderr)
        sys.exit()
    
    download_and_extract_pretrained_model(
        pretrained_model_url,
        args.out_dir
    )

if __name__ == '__main__':
    main()