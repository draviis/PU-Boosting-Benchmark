"""Obtain the separately licensed official KM2 source; never fit a model."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import argparse,hashlib,io

URL='https://web.eecs.umich.edu/~cscott/code/kernel_MPE.zip'
EXPECTED='12960c46c816b8118926458467b17013cc016aa5a89dd41b8dc267d6a8f7cb9f'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    a=p.parse_args()
    target=a.workspace/'mlj_revision/06_estimated_prior/sources/upstream_Kernel_MPE_grad_threshold.py'
    if target.exists():raise FileExistsError(target)
    with urlopen(URL,timeout=60) as response:raw=response.read()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:
        raise ValueError('Upstream archive changed; inspect it before use. No file was installed.')
    with ZipFile(io.BytesIO(raw)) as z:
        names=[n for n in z.namelist() if n.endswith('/Kernel_MPE_grad_threshold.py') or n=='Kernel_MPE_grad_threshold.py']
        assert len(names)==1,names
        source=z.read(names[0])
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(source)
    print('Verified author archive; source saved to',target)

if __name__=='__main__':main()
