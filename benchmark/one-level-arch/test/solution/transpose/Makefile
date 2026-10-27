TESTCASE ?= transpose

ifeq ($(TESTCASE),transpose)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose.cpp
TARGET = $(ELF_HEAD)_transpose_PE4.elf
else ifeq ($(TESTCASE),transpose_tail)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_tail.cpp
TARGET = $(ELF_HEAD)_transpose_tail_PE4.elf
else ifeq ($(TESTCASE),transpose_large)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_large.cpp
TARGET = $(ELF_HEAD)_transpose_large_PE4.elf
else ifeq ($(TESTCASE),copy_flat)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/copy_flat.cpp
TARGET = $(ELF_HEAD)_copy_flat_PE4.elf
else ifeq ($(TESTCASE),transpose_bnsd)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_bnsd.cpp
TARGET = $(ELF_HEAD)_transpose_bnsd_PE4.elf
else ifeq ($(TESTCASE),transpose_permute)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_permute.cpp
TARGET = $(ELF_HEAD)_transpose_permute_PE4.elf
else ifeq ($(TESTCASE),transpose_v31)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_v31.cpp
TARGET = $(ELF_HEAD)_transpose_v31_PE4.elf
else ifeq ($(TESTCASE),transpose_v32)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_v32.cpp
TARGET = $(ELF_HEAD)_transpose_v32_PE4.elf
V32_CASE ?= 0
DEFINES += -DV32_CASE=$(V32_CASE)
else ifeq ($(TESTCASE),transpose_simt)
# element-if（SIMT/SMALL_SHAPE）路径：需含 element-if 的 Linx LLVM + generic-element lowering
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_simt.cpp
TARGET = $(ELF_HEAD)_transpose_simt_PE4.elf
SIMT_CASE ?= 0
DEFINES += -DSIMT_CASE=$(SIMT_CASE) -DENABLE_BINARY_OUTPUT
CC_OPTS += -mllvm -linxv5-enable-generic-element=true
else ifeq ($(TESTCASE),transpose_bench)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_bench.cpp
TARGET = $(ELF_HEAD)_transpose_bench_PE4.elf
else ifeq ($(TESTCASE),transpose_dynamic)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_dynamic.cpp
TARGET = $(ELF_HEAD)_transpose_dynamic_PE4.elf
else ifeq ($(TESTCASE),transpose_minrepro)
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/transpose_minrepro.cpp
TARGET = $(ELF_HEAD)_transpose_minrepro_PE4.elf
else
$(error Unsupported TESTCASE=$(TESTCASE); expected transpose | transpose_tail | transpose_large | copy_flat | transpose_bnsd | transpose_permute | transpose_v31 | transpose_bench | transpose_dynamic)
endif

include ../../common/Makefile.common
