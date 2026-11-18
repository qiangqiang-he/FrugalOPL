"""CPU-only post-completion export and verified pull to this Windows workspace."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/helpers'))
from export_kl_rollouts import pack, verify_archive, digest

REMOTE_ROOT = '/home/ubuntu/heqiangqiang/FrugalOPL-aime24-avg16-20261108'
REMOTE_PYTHON = '/data/heqiangqiang/miniforge3/envs/rlvr/bin/python'
SSH_CONFIG = '/home/qqh/.ssh/codex-rtx5000-1/config'
SSH_OPTIONS = ['-F', SSH_CONFIG, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
HOST = 'rtx5000-1'
MODELS = ['Qwen3-4B', 'Qwen3-1.7B', 'Qwen3-8B']
LOCAL_OUTPUT = Path('/mnt/c/Users/qqian/Desktop/Projects/FrugalOPL/outputs/aime24_avg16_rtx5000_1_20261108')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def status(root):
    base = root/'outputs'
    completed, saved = [], {}
    for model in MODELS:
        folder = base/model
        if (folder/'completion_verified.json').is_file():
            record = json.loads((folder/'completion_verified.json').read_text())
            assert record['verified'] and record['response_count'] == 1440
            completed.append(model)
        saved[model] = len(list((folder/'rollouts').glob('batch_*.json')))
    result = base/'campaign_results.json'
    campaign = json.loads(result.read_text()) if result.is_file() else {}
    ready = len(completed) == 3 and campaign.get('pending') == []
    exit_status = base/'.logs/campaign.status'
    failed = exit_status.is_file() and exit_status.read_text().strip() != '0'
    return {'ready': ready, 'completed_models': completed, 'saved_batches': saved,
            'campaign_failed': failed, 'updated_at': int(time.time())}


def export_all(root):
    assert status(root)['ready'], 'All nine groups must complete before export'
    base = root/'outputs'
    cached = base/'local_transfer_manifest.json'
    if cached.is_file():
        manifest = json.loads(cached.read_text())
        assert all(digest(item['archive']) == item['archive_sha256'] for item in manifest['models'].values())
        return manifest
    records = {}
    for model in MODELS:
        folder = base/model
        details = json.loads((folder/'details.json').read_text())
        assert details['save_rollouts'] and len(details['datasets']) == 3
        for ds in details['datasets'].values():
            assert len(ds['questions']) == 30
            for q in ds['questions']:
                assert len(q['rollouts']) == 16
                assert all(isinstance(r.get('response'), str) and r['response'] for r in q['rollouts'])
        records[model] = pack(folder, folder/'all_rollouts_export.tar.gz')
    manifest = {'models': records, 'total_groups': 9, 'total_responses': 4320,
                'created_at': int(time.time())}
    write(cached,manifest)
    return manifest


def remote(action):
    command = shlex.join([REMOTE_PYTHON, REMOTE_ROOT+'/scripts/aime24_avg16_transfer.py',
                          action, '--root', REMOTE_ROOT])
    result = subprocess.run(['ssh', *SSH_OPTIONS, HOST, command], capture_output=True,
                            text=True, timeout=600, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def pull(manifest, output):
    verified = {}
    for model, item in manifest['models'].items():
        folder = output/model
        folder.mkdir(parents=True,exist_ok=True)
        archive = folder/'all_rollouts_export.tar.gz'
        if not archive.is_file() or digest(archive) != item['archive_sha256']:
            partial = archive.with_suffix(archive.suffix+'.partial')
            transport = shlex.join(['ssh', *SSH_OPTIONS])
            subprocess.run(['rsync', '--partial', '--append-verify', '--timeout=120',
                '-e', transport, HOST+':'+item['archive'], str(partial)], check=True, timeout=1800)
            assert digest(partial) == item['archive_sha256'], 'Archive checksum mismatch'
            partial.replace(archive)
        verified[model] = verify_archive(archive,folder,item['archive_sha256'])
        assert verified[model]['total_responses'] == 1440
        assert verified[model]['total_questions'] == 90
        assert len(verified[model]['datasets']) == 3
        assert all(v['responses'] == 480 and v['questions'] == 30
                   for v in verified[model]['datasets'].values())
    assert sum(v['total_responses'] for v in verified.values()) == 4320
    return {'state':'complete','total_groups':9,'total_responses':4320,
            'completed_at':int(time.time()),'local_output':str(output),'models':verified}


def watch(output, poll):
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'transfer.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous = None
    while True:
        try:
            state = remote('status')
            if state['ready']:
                write(output/'transfer_status.json',dict(state,state='transferring'))
                manifest = remote('pack')
                write(output/'local_transfer_manifest.json',manifest)
                verified = pull(manifest,output)
                write(output/'download_complete.json',verified)
                write(output/'transfer_status.json',verified)
                print(json.dumps(verified),flush=True)
                return
            if state['campaign_failed']:
                write(output/'transfer_status.json',dict(state,state='waiting_for_evaluation_recovery'))
                print(json.dumps(state),flush=True)
                time.sleep(poll)
                continue
            write(output/'transfer_status.json',dict(state,state='waiting_for_all_nine_groups'))
            key = (tuple(state['completed_models']),tuple(state['saved_batches'].values()))
            if key != previous:
                print(json.dumps(state),flush=True)
                previous = key
        except (subprocess.SubprocessError,OSError,ValueError,AssertionError) as error:
            message = str(error)
            if isinstance(error,subprocess.CalledProcessError) and error.stderr:
                message += ': '+error.stderr[-1600:]
            state = {'state':'retrying','error':message,'updated_at':int(time.time())}
            write(output/'transfer_status.json',state)
            print(json.dumps(state),flush=True)
        time.sleep(poll)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','pack','watch'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=LOCAL_OUTPUT)
    parser.add_argument('--poll',type=int,default=120)
    args = parser.parse_args()
    if args.action == 'watch':
        watch(args.output,args.poll)
    else:
        print(json.dumps(status(args.root) if args.action == 'status' else export_all(args.root)),flush=True)
