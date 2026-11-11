import os,sys,tarfile,tempfile,subprocess,json,time
from pathlib import Path
A=Path(os.environ['IFAB_AFS_HOME'])
if '--worker' not in sys.argv:
 with tempfile.TemporaryDirectory(prefix='nt_multi_',dir=Path.cwd()) as d:
  root=Path(d)
  with tarfile.open(A/'imports/fl2va_sources_gfx936.tar.gz') as t:t.extractall(root,filter='data')
  env=os.environ.copy();env['LD_LIBRARY_PATH']=str(root/'InfiniCore-minimax-h3/python/infinicore/lib')+':/usr/local/lib/python3.11/site-packages/torch/lib:/opt/dtk/lib:/opt/dtk/lib64:'+env.get('LD_LIBRARY_PATH','');env['PYTHONPATH']=str(root/'InfiniCore-minimax-h3/python')
  for label,bundle in [('original','fl2va_aot_gfx936.tar.gz'),('candidate','fl2va_aot_gfx936_devicecache_candidate_v1.tar.gz')]:
   target=root/label;target.mkdir()
   with tarfile.open(A/'imports'/bundle) as t:t.extractall(target,filter='data')
   env['NT_PROBE_LABEL']=label;env['INFINI_TIMESTEP_NT']='1';env['INFINI_TIMESTEP_NT_ROOT']=str(target)
   p=subprocess.run([str(A/'venvs/fl2va/bin/python'),'-u',__file__,'--worker',str(target)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   print('VARIANT',label,'EXIT',p.returncode,flush=True);print(p.stdout,flush=True);(A/('diagnostics/nt_multigpu_'+label+'_v1.log')).write_text(p.stdout)
   if label=='candidate':raise SystemExit(p.returncode)
import torch,ctypes as C,concurrent.futures,threading
hip=C.CDLL('libgalaxyhip.so.5');hip.hipCtxGetCurrent.argtypes=[C.POINTER(C.c_void_p)];hip.hipGetDevice.argtypes=[C.POINTER(C.c_int)];hip.hipGetErrorString.restype=C.c_char_p
class T(C.Structure):_fields_=[('data',C.c_void_p),('shape',C.POINTER(C.c_uint64)),('strides',C.POINTER(C.c_int64))]
libs=[]
for so in sorted(Path(sys.argv[-1]).rglob('*.so')):
 name=so.stem;lib=C.CDLL(str(so));f=getattr(lib,'launch_'+name);f.argtypes=[C.c_void_p,T,T,T,T];f.restype=C.c_int;libs.append((name,lib,f))
print('GPUS',torch.cuda.device_count(),'LIBS',len(libs),flush=True)
def run(dev,barrier=None):
 torch.cuda.set_device(dev);torch.cuda.init();ctx=C.c_void_p();c=hip.hipCtxGetCurrent(C.byref(ctx));di=C.c_int();g=hip.hipGetDevice(C.byref(di));print('CONTEXT',dev,c,ctx.value,g,di.value,flush=True)
 if barrier:barrier.wait()
 results=[]
 for name,lib,f in libs:
  dim=256 if 'd256' in name else 16;dtype=torch.float16 if 'fp16' in name else torch.bfloat16 if 'bf16' in name else torch.float32
  for b in [1,4,17]:
   x=torch.linspace(-1000,1000,b,device='cuda',dtype=dtype);y=torch.empty((b,dim),device='cuda',dtype=torch.float32);shape=(C.c_uint64*1)(b);stride=(C.c_int64*1)(1);osh=(C.c_uint64*2)(b,dim//2);ost=(C.c_int64*2)(dim,1);period=C.c_double(10000.)
   status=f(torch.cuda.current_stream().cuda_stream,T(x.data_ptr(),shape,stride),T(C.addressof(period),None,None),T(y.data_ptr(),osh,ost),T(y.data_ptr()+dim//2*4,osh,ost))
   if status:results.append({'device':dev,'name':name,'batch':b,'status':status,'error':hip.hipGetErrorString(status).decode()});continue
   torch.cuda.synchronize();angles=x.float()[:,None]*torch.exp(-torch.log(torch.tensor(10000.,device='cuda'))*torch.arange(dim//2,device='cuda')/(dim//2));ref=torch.cat([angles.cos(),angles.sin()],1);err=(y-ref).abs().max().item();results.append({'device':dev,'name':name,'batch':b,'status':status,'maxdiff':err})
 return results
all_results=[]
for dev in range(torch.cuda.device_count()):all_results.extend(run(dev))
barrier=threading.Barrier(torch.cuda.device_count())
with concurrent.futures.ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as pool:
 for rows in pool.map(lambda d:run(d,barrier),range(torch.cuda.device_count())):all_results.extend(rows)
bad=[r for r in all_results if r['status'] or r.get('maxdiff',0)>0.002]
out={'cases':len(all_results),'bad':bad,'results':all_results};(A/('diagnostics/nt_multigpu_'+os.environ['NT_PROBE_LABEL']+'_v1.json')).write_text(json.dumps(out,indent=2));print('MULTIGPU_RESULT',json.dumps({'cases':len(all_results),'bad_count':len(bad),'first_bad':bad[:12]}),flush=True)
raise SystemExit(bool(bad))
