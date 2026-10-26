import ast,copy,difflib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'tile_kernels/quant'
class Shell(ast.NodeTransformer):
 def visit_With(self,node):
  if any(isinstance(i.context_expr,ast.Call) and ast.unparse(i.context_expr.func)=='T.SimdVF' for i in node.items):
   return ast.Pass()
  return self.generic_visit(node)
 def visit_FunctionDef(self,n):
  self.generic_visit(n)
  n.name=n.name.replace('_vmi','_asc')
  if n.name=='swiglu_forward_kernel': n.name='swiglu_forward_kernel_asc'
  n.decorator_list=[]
  return n
 def visit_Name(self,n):
  n.id=n.id.replace('_vmi','_asc')
  return n

def shell(src):
 t=ast.parse(src);f=t.body[-1]
 # SIMD-only helper bodies may change; outer kernel and scalar helper contracts must match.
 f.body=[n for n in f.body if not isinstance(n,ast.FunctionDef) or any(ast.unparse(d)=='T.prim_func' for d in n.decorator_list) or not any(isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id in ('S','V') for x in ast.walk(n))]
 return ast.unparse(Shell().visit(f))
for d in ['forward','backward']:
 a=shell((root/f'swiglu_{d}_asc.py').read_text());b=shell((root/f'swiglu_{d}_vmi.py').read_text())
 diff=list(difflib.unified_diff(a.splitlines(),b.splitlines(),fromfile='ASC shell',tofile='VMI shell'))
 print(d,'MATCH' if not diff else '\n'.join(diff))
 if diff: raise SystemExit(1)
