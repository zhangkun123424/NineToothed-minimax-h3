from pathlib import Path
import os, ast, textwrap
w=Path(os.environ['FL2VA_WORK']); c=w/'compiler310'
p=c/'triton/runtime/cache.py';s=p.read_text()
s=s.replace('os.replace(temp_path, filepath)','shutil.move(temp_path, filepath)').replace('shutil.copyfile(temp_path, filepath)','shutil.move(temp_path, filepath)')
if 'import shutil' not in s:s=s.replace('import os','import os'+chr(10)+'import shutil',1)
compile(s,str(p),'exec');p.write_text(s)
p=c/'triton/backends/hcu/compiler_hcu.py';s=p.read_text()
old="        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))"
assert s.count(old)==1
s=s.replace(old,"        extern_libs['oclc_daz_opt'] = '/opt/dtk/dcc/dccgcn/bitcode/oclc_daz_opt_off.bc'"+chr(10)+old)
s=s.replace('if hcu.need_extern_lib(llvm_mod, name)','if name == "oclc_daz_opt" or hcu.need_extern_lib(llvm_mod, name)')
methods={'make_amdgcn':'''def make_amdgcn(src, metadata, options):
    names = re.findall(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)", src)
    assert len(names) == 1
    metadata["name"] = names[0]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "kernel.ll"
        output = Path(tmp) / "kernel.s"
        source.write_text(str(src))
        command = [str(HIPBackend.path_to_rocm_clang()), "-target", "amdgcn-amd-amdhsa", "-mcpu=" + options.arch, "-O3", "-S", str(source), "-o", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode or not output.exists(): raise RuntimeError(result.stdout + result.stderr)
        return output.read_text()
''','make_hsaco':'''def make_hsaco(src, metadata, options):
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "kernel.s"
        output = Path(tmp) / "kernel.hsaco"
        source.write_text(str(src))
        command = [str(HIPBackend.path_to_rocm_clang()), "-target", "amdgcn-amd-amdhsa", "-mcpu=" + options.arch, "-O3", str(source), "-o", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode or not output.exists(): raise RuntimeError(result.stdout + result.stderr)
        return output.read_bytes()
'''}
lines=s.splitlines(keepends=True);nodes=[n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name in methods]
for n in sorted(nodes,key=lambda n:n.lineno,reverse=True):lines[n.lineno-1:n.end_lineno]=[textwrap.indent(methods[n.name],'    ')]
s=''.join(lines);compile(s,str(p),'exec');p.write_text(s)
for f in (p.parent/'__pycache__').glob('compiler_hcu*.pyc'):f.unlink()
p=w/'build_aot_gfx936.py';s=p.read_text()
s=s.replace("root=pathlib.Path('aot_gfx936')", "root=pathlib.Path(os.environ['FL2VA_WORK'])/'aot_gfx936'")
s=s.replace("name=f'nt_timestep_generic_d{dim}_{label}'", "name=f'nt_timestep_generic_d{dim}_{label}' if (dim,label)!=(256,'fp32') else 'nt_timestep_aot_d256_fp32'")
s=s.replace("directory=root/name;directory.mkdir(parents=True,exist_ok=True)", "directory=root/(f'aot_generic_d{dim}_{label}' if (dim,label)!=(256,'fp32') else 'aot_d256_fp32');directory.mkdir(parents=True,exist_ok=True)")
compile(s,str(p),'exec');p.write_text(s)
print('AOT_GFX936_PREPARED_WITH_DAZ',flush=True)
