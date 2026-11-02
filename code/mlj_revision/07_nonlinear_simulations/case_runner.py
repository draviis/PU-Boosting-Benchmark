"""Run the eight original-method blocks; retry only interrupted/incomplete blocks."""
from pathlib import Path
import argparse,json,subprocess,sys,time,os,hashlib
H=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);a=p.parse_args()
    manifest=json.loads((H/'execution_manifest.json').read_text())
    for file,key in [('worker.py','worker_sha256'),('main_cases.json','case_manifest_sha256'),('proposed_method_catalog.json','catalog_sha256')]:assert hashlib.sha256((H/file).read_bytes()).hexdigest()==manifest[key]
    case=json.loads((H/'main_cases.json').read_text())[a.case];root=H/'results'/case['key'];root.mkdir(parents=True,exist_ok=True)
    errors=[]
    for block in range(8):
        folder=root/('block'+str(block));folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():continue
        # Interrupted writes are archived, never overwritten.
        for z in folder.glob('M[0-9][0-9][0-9].npz'):
            meta=z.with_suffix('.json')
            if not meta.exists():
                archive=folder/'incomplete_attempts';archive.mkdir(exist_ok=True)
                target=archive/(str(time.time_ns())+'_'+z.name)
                assert folder.resolve() in target.resolve().parents
                z.rename(target)
        timeout=7200 if case['sn']==5 else 900
        cmd=[sys.executable,'-B',str(H/'worker.py'),'--case',str(a.case),'--block',str(block)]
        for attempt in range(2):
            t=time.monotonic()
            try:r=subprocess.run(cmd,timeout=timeout);code=r.returncode
            except subprocess.TimeoutExpired:code=124
            record=dict(case=case['key'],block=block,attempt=attempt,exit_code=code,seconds=time.monotonic()-t)
            with (root/('execution_'+str(time.time_ns())+'.json')).open('x') as f:json.dump(record,f)
            if code==0:break
            if code!=124:break
        if code:errors.append(dict(block=block,exit_code=code))
    if errors:raise RuntimeError(errors)
if __name__=='__main__':main()
