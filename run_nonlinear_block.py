"""Explicitly requested fitting entry: one frozen nonlinear method block.

Only the machine-specific compiled AdaPU binary hash assertion is replaced
by checks for a locally compiled extension and unchanged Cython/source files.
All scientific source, frozen parameter, catalog, and package-version checks
remain in the archived worker. This script is NOT used by reproduce.py.
"""
from pathlib import Path
import argparse,ast,hashlib,json,sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--case',type=int,required=True,help='Zero-based main case index, 0 through 1199')
    p.add_argument('--block',type=int,choices=range(8),required=True)
    a=p.parse_args();w=a.workspace.resolve()
    if not 0<=a.case<1200:p.error('--case must be 0 through 1199')
    dep=w/'phase3_confirmatory_S001_S040_100reps/Ada-PU/src'
    binaries=list(dep.glob('decisionstump*.so'))+list(dep.glob('decisionstump*.pyd'))
    if len(binaries)!=1:raise RuntimeError('Compile the frozen AdaPU extension first; expected exactly one local binary.')
    worker=w/'mlj_revision/07_nonlinear_simulations/worker.py'
    tree=ast.parse(worker.read_text(encoding='utf8'));removed=[]
    class BinaryPortability(ast.NodeTransformer):
        def visit_Assert(self,node):
            literals=[v.value for v in ast.walk(node) if isinstance(v,ast.Constant) and isinstance(v.value,str)]
            if 'decisionstump.cpython-310-x86_64-linux-gnu.so' in literals:
                removed.append(ast.unparse(node));return None
            return node
    tree=BinaryPortability().visit(tree)
    assert len(removed)==1,removed
    record=dict(case=a.case,block=a.block,archived_binary_check=removed[0],
                local_binary=binaries[0].name,local_binary_sha256=hashlib.sha256(binaries[0].read_bytes()).hexdigest(),
                scientific_source_changed=False)
    log=w/f'local_build_case{a.case}_block{a.block}.json'
    with log.open('x') as f:json.dump(record,f,indent=2)
    sys.path.insert(0,str(worker.parent))
    sys.argv=[str(worker),'--case',str(a.case),'--block',str(a.block)]
    exec(compile(ast.fix_missing_locations(tree),str(worker),'exec'),{'__file__':str(worker),'__name__':'__main__'})

if __name__=='__main__':main()
