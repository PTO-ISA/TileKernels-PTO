import argparse,importlib,importlib.util,json,os,sys,time,traceback,hashlib,subprocess
from pathlib import Path
import torch,torch_npu
import tile_kernels
from tilelang.profiler import do_bench
from tile_kernels.testing.bench import get_cast_params
from tile_kernels.quant import swiglu_forward_vmi as fv,swiglu_backward_vmi as bv
from tile_kernels.quant import swiglu_forward_asc as fa,swiglu_backward_asc as ba
p=argparse.ArgumentParser();p.add_argument('--direction',choices=['forward','backward'],required=True);p.add_argument('--only',default='');p.add_argument('--rounds',type=int,default=3);p.add_argument('--rep',type=int,default=20);p.add_argument('--out',required=True);p.add_argument('--correctness-only',action='store_true');p.add_argument('--tokens',type=int);a=p.parse_args()
torch.npu.set_device(int(os.environ.get('TK_DEVICE','1')))
root=Path(os.environ['SWIGLU_AUDIT_ROOT']);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
expected={'tilelang':'4eb83fdeb2fb317a0485052f51d0b77f4238d8b5','ptoas':'04c785491e844ec6d642e444a4e3ee84d1a60496'}
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
import tilelang,ptoas,ptodsl
identity={}
for name,rev in expected.items():
 actual=subprocess.check_output(['git','-C',str(root/name),'rev-parse','HEAD'],text=True).strip();assert actual==rev
 identity[name]=actual
 identity[name+'_diff']=hashlib.sha256(subprocess.check_output(['git','-C',str(root/name),'diff','HEAD'])).hexdigest()
for mod in [fv,bv,fa,ba]: identity[mod.__name__]=sha(mod.__file__)
identity.update(tilelang_module=tilelang.__file__,ptodsl_module=ptodsl.__file__,ptoas_module=ptoas.__file__,manifest=sha(root/'ptoas-build.json'),atomic_python=sha(root/'ptoas/build/python/ptodsl/_ops_simt.py'),shim=sha(root/'pysite/sitecustomize.py'))
assert str(root/'tilelang') in tilelang.__file__;assert str(root/'ptoas') in ptodsl.__file__
for folder in [out,Path(os.environ['TILELANG_CACHE_DIR'])]:
 marker=folder/'revision_group.json'
 if marker.exists(): assert json.loads(marker.read_text())==identity,'Revision mismatch'
 else: marker.write_text(json.dumps(identity,indent=2))
print(json.dumps(identity),flush=True)
spec=importlib.util.spec_from_file_location('test_swiglu',root/f'tk/tests/quant/test_swiglu_{a.direction}.py');test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
params=test.generate_test_params(0);selected=set(map(int,a.only.split(','))) if a.only else set(range(len(params)))
if a.tokens:
 for param in params:param['num_send_tokens']=a.tokens
vmodule=fv if a.direction=='forward' else bv;amodule=fa if a.direction=='forward' else ba
vname=f'get_swiglu_{a.direction}_kernel_vmi';aname=f'get_swiglu_{a.direction}_kernel_asc';vfactory=getattr(vmodule,vname);afactory=getattr(amodule,aname)
records=out/f'{a.direction}.jsonl';done=set()
if records.exists():
 for line in records.read_text().splitlines():
  try:
   r=json.loads(line)
   if r.get('status')=='pass':done.add(r['id'])
  except ValueError:pass

def compare(x,y,mask,path='out'):
 if x is None: assert y is None;return []
 if isinstance(x,(tuple,list)):
  assert len(x)==len(y);return sum([compare(xx,yy,mask,path+f'.{i}') for i,(xx,yy) in enumerate(zip(x,y))],[])
 assert x.shape==y.shape and x.dtype==y.dtype,(path,x.shape,y.shape)
 if x.dtype==torch.float8_e4m3fn: x=x.view(torch.uint8);y=y.view(torch.uint8)
 if x.ndim and x.shape[0]==mask.numel(): x=x[mask];y=y[mask]
 xb=x.contiguous().view(torch.uint8);yb=y.contiguous().view(torch.uint8)
 mism=int((xb!=yb).sum().item());maxabs=float((x.float()-y.float()).abs().nan_to_num().max().item()) if x.numel() else 0
 # weight_grad is the only repository tolerance exception.
 weight=(a.direction=='backward' and params_current['with_weights'] and path==('out.2' if params_current['fmt']=='e4m3' else 'out.1'))
 if weight: torch.testing.assert_close(x,y,atol=1e-4,rtol=1e-4)
 else: assert mism==0,(path,mism,maxabs)
 return [{'path':path,'mismatch_bytes':mism,'max_abs':maxabs,'tolerance':1e-4 if weight else 0}]
for idx,param in enumerate(params):
 if idx not in selected or idx in done:continue
 params_current=param
 torch.manual_seed(1234+idx);torch.npu.manual_seed_all(1234+idx);tile_kernels.set_token_alignment(param['alignment'])
 if 'num_sms' in param:tile_kernels.set_num_sms(param['num_sms'])
 record={'id':idx,'params':param};print('START',a.direction,idx,param,flush=True)
 try:
  if a.direction=='forward':
   x,mask,psum,w,c=test.generate_test_data(param)
   def make(arm):
    counts=c.clone() if param['clamp_value'] is not None else None
    def call():
     setattr(vmodule,vname,vfactory if arm=='vmi' else afactory)
     kw=dict(x=x,fmt=param['fmt'],psum_num_tokens_per_expert=psum,topk_weights=w if param['with_weights'] else None,routed_scaling_factor=param['routed_scaling_factor'],clamp_value=param['clamp_value'],clamped_count=counts)
     if param['fmt']=='e4m3':return tile_kernels.quant.swiglu_forward_and_per_token_cast(**kw,num_per_channels=param['num_per_channels'],**get_cast_params(param))
     return tile_kernels.quant.swiglu_forward(**kw)
    return call,counts
   ac,acc=make('asc');vc,vcc=make('vmi')
  else:
   x,w,psum,grad=test.generate_test_data(param);mask=torch.arange(grad.shape[0],device=grad.device)<grad.shape[0]
   if psum is not None:
    mask=torch.zeros(grad.shape[0],dtype=torch.bool,device=grad.device);start=0
    for end in psum.cpu().tolist():mask[start:end]=True;start=((end+param['alignment']-1)//param['alignment'])*param['alignment']
   def make(arm):
    def call():
     setattr(vmodule,vname,vfactory if arm=='vmi' else afactory)
     kw=dict(x=x,act_x_grad=grad,fmt=param['fmt'],psum_num_tokens_per_expert=psum,topk_weights=w,routed_scaling_factor=param['routed_scaling_factor'],clamp_value=param['clamp_value'],do_recompute=param['do_recompute'])
     if param['fmt']=='e4m3':return tile_kernels.quant.swiglu_backward_and_per_token_cast(**kw,num_per_channels=param['num_per_channels'],**get_cast_params(param))
     return tile_kernels.quant.swiglu_backward(**kw)
    return call,None
   ac,acc=make('asc');vc,vcc=make('vmi')
  ar=ac();vr=vc();torch.npu.synchronize();record['checks']=compare(ar,vr,mask)
  if acc is not None:record['clamp_checks']=compare(acc,vcc,torch.ones(acc.shape[0],dtype=torch.bool,device=acc.device),'clamp')
  if not a.correctness_only:
   times=[]
   for r in range(a.rounds):
    samples={}
    for label,fn in ([('asc',ac),('vmi',vc)] if r%2==0 else [('vmi',vc),('asc',ac)]):samples[label]=float(do_bench(fn,backend='msprof',rep=a.rep))*1000
    samples['ratio']=samples['asc']/samples['vmi'];times.append(samples)
   record['times']=times
  record['status']='pass'
 except Exception as e:
  record.update(status='fail',error=str(e),traceback=traceback.format_exc());print(record['traceback'],flush=True)
 with records.open('a') as f:f.write(json.dumps(record)+'\n')
 print('RESULT',json.dumps(record),flush=True)
 if record['status']!='pass':raise SystemExit(1)
 torch.npu.empty_cache()
