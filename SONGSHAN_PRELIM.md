# MiniMax-H3 NineToothed AOT adaptation

Base: NineToothed 0.26.0, `daf55c76c8e8b50904b0b84678d9bca93a1c8948`.

The gfx936 compiler overlay copies `ninetoothed_aot_gfx936.py` and `ninetoothed_generation_gfx936.py` to `ninetoothed/aot.py` and `ninetoothed/generation.py` in the AOT compiler environment. `kernel_gfx936.py` defines TimestepEmbedding and `build_aot_gfx936.py` generates the six dimension/dtype builds using the patched HCU compiler environment.

`build_nt_devicecache_v1.py` rebuilds host wrappers from `$IFAB_AFS_HOME/imports/fl2va_aot_gfx936.tar.gz`, changes the HIP cache key to the logical device and makes the cache thread-local. `ninetoothed_devicecache.patch` records the wrapper delta. `nt_multigpu_paired_v1.py` runs the paired eight-device regression of the base and repaired packages. The compiler preparation script documents the accompanying FlagTree/HCU adjustments.
