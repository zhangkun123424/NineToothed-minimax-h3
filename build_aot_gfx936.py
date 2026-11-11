import os,pathlib,functools,importlib
import ninetoothed as nt
from kernel_gfx936 import arrangement,application
aot=importlib.import_module('ninetoothed.aot'); original=aot._enumerate_variant_specs
aot._enumerate_variant_specs=lambda *args,**kw: [s for s in original(*args,**kw) if not s[1] and not s[2]]
aot.calculate_default_configs=lambda:(4,1)
aot._load_launch_func=lambda *args:None
root=pathlib.Path(os.environ['FL2VA_WORK'])/'aot_gfx936'
for dim in (16,256):
 for dtype,label in [(nt.float16,'fp16'),(nt.bfloat16,'bf16'),(nt.float32,'fp32')]:
  name=f'nt_timestep_generic_d{dim}_{label}' if (dim,label)!=(256,'fp32') else 'nt_timestep_aot_d256_fp32'
  directory=root/(f'aot_generic_d{dim}_{label}' if (dim,label)!=(256,'fp32') else 'aot_d256_fp32');directory.mkdir(parents=True,exist_ok=True)
  x=nt.Tensor(1,dtype=dtype);p=nt.Tensor(0,dtype=nt.float32);c=nt.Tensor(2,dtype=nt.float32);s=nt.Tensor(2,dtype=nt.float32)
  for out in (c,s):out.shape=(out.shape[0],dim//2)
  print('BUILD_START',name,flush=True)
  nt.make(functools.partial(arrangement,embedding_dim=dim),application,(x,p,c,s),caller='cuda',kernel_name=name,output_dir=directory,num_warps=4,num_stages=1)
  print('BUILD_OK',name,flush=True)
