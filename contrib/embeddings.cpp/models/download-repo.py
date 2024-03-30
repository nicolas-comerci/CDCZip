import argparse
import huggingface_hub as hh
import os

def download_model(args):
    assert '/' in args.repo_id, 'repo_id should be of the form <user>/<repo>'
    model_name = args.repo_id.split('/')[1]
    hh.snapshot_download(repo_id=args.repo_id, local_dir=f'./{model_name}', local_dir_use_symlinks=False)
    print(f'{args.repo_id} downloaded successfully')

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='Download original repo files')
    parser.add_argument('repo_id', help='Name of the repo')
    download_model(parser.parse_args())

