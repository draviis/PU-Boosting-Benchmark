"""Download six UCI tables and run the archived preparation functions (no fitting)."""
from pathlib import Path
from urllib.request import urlopen
import argparse,hashlib,importlib.util,json

HERE=Path(__file__).resolve().parent

def download(url,dest,digest):
    if dest.exists():raw=dest.read_bytes()
    else:
        with urlopen(url,timeout=120) as response:raw=response.read()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError(f'Source changed: {url}. Review against the archived recipe before proceeding.')
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(raw)
    assert hashlib.sha256(raw).hexdigest()==digest,dest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--dataset',required=True,choices=['rice','phishing','credit_default','dry_bean','optical_digits','online_shoppers'])
    a=p.parse_args();work=a.workspace.resolve()
    if not (work/'real_data_batch3/build_bundle.py').exists():
        raise ValueError('First materialize the release with reproduce.py --materialize-only.')
    spec=json.loads((HERE/'docs/real_data_download_sources.json').read_text())[a.dataset]
    for asset in spec['downloads']:
        download(asset['url'],work/asset['path'],asset['sha256'])
    path=work/spec['recipe'];loader=importlib.util.spec_from_file_location('frozen_preparation',path)
    mod=importlib.util.module_from_spec(loader);loader.loader.exec_module(mod)
    code=work/spec['prepared_parent']
    mod.prepare_dataset(a.dataset,mod.SPECS[a.dataset],code)
    print('Prepared data without fitting:',code/'data'/a.dataset)

if __name__=='__main__':main()
