import os,tarfile,re,subprocess,json,hashlib
from pathlib import Path
A=Path(os.environ['IFAB_AFS_HOME']);root=Path('nt_aot_device_candidate');root.mkdir(exist_ok=True)
with tarfile.open(A/'imports/fl2va_aot_gfx936.tar.gz') as t:t.extractall(root,filter='data')
changes=[]
old='hipCtx_t current_ctx; hipError_t ctx_status = hipCtxGetCurrent(&current_ctx); if (ctx_status != hipSuccess) return (NineToothedResult)ctx_status; ctx_id = (unsigned long long)(uintptr_t)current_ctx;'
new='int current_device = -1; hipError_t device_status = hipGetDevice(&current_device); if (device_status != hipSuccess) return (NineToothedResult)device_status; ctx_id = (unsigned long long)current_device;'
for p in root.rglob('*.cpp'):
 s=p.read_text()
 if old not in s:continue
 assert s.count(old)==1
 s=s.replace(old,new).replace('static ninetoothed::ThreadSafeUnorderedMap','static thread_local ninetoothed::ThreadSafeUnorderedMap')
 p.write_text(s);changes.append(str(p.relative_to(root)))
assert len(changes)==12,changes
logs=[]
for so in sorted(root.rglob('*.so')):
 files=[str(p) for p in so.parent.glob('*.cpp')]
 cmd=['g++','-std=c++17','-shared','-fPIC','-O2','-D__HIP_PLATFORM_AMD__','-I/opt/dtk/include','-I/opt/dtk/hip/include',*files,'-L/opt/dtk/hip/lib','-lgalaxyhip','-o',str(so)]
 r=subprocess.run(cmd,capture_output=True,text=True);logs.append({'library':str(so),'return_code':r.returncode,'output':r.stdout+r.stderr})
 if r.returncode:Path('nt_build_result.txt').write_text(json.dumps(logs,indent=2));raise SystemExit(r.returncode)
out=A/'imports/fl2va_aot_gfx936_devicecache_candidate_v1.tar.gz'
with tarfile.open(out,'w:gz') as t:
 for p in sorted(root.iterdir()):t.add(p,arcname=p.name)
out.chmod(0o660)
manifest={'status':'UNVALIDATED_GPU_CANDIDATE','changed_host_sources':changes,'device_kernel_code':'unchanged','cache':'thread local, per hipGetDevice logical device','sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'builds':logs}
(A/'diagnostics/nt_devicecache_build_v1.json').write_text(json.dumps(manifest,indent=2));Path('nt_build_result.txt').write_text(json.dumps(manifest,indent=2));print('CANDIDATE_HOST_BUILD_OK',flush=True)
