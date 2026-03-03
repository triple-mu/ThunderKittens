# You must set the environment variable GPU
# export GPU=H100 | B200

# Default to the new kernel. You can still build the legacy one with:
# make GPU=H100 SRC=all_to_all.cu
SRC ?= all_to_all_new.cu
OUT ?= _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD ?= OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 benchmark.py
CONFIG ?= pytorch
include ../../common.mk
