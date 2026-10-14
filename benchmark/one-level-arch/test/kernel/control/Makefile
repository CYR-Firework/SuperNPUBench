TARGET = $(ELF_HEAD)_$(TESTCASE)$(SUFFIX).elf

# Override target names
ifeq ($(TESTCASE), hashtable_lookup_simd)
TARGET = $(ELF_DIR)/hashtable_lookup_simd$(SUFFIX).elf
endif

ifeq ($(TESTCASE), hkv)
TARGET = $(ELF_DIR)/hkv_lookup.elf
endif

ifeq ($(TESTCASE), simple_rdadd)
TARGET = $(ELF_DIR)/simple_rdadd$(SUFFIX).elf
endif

ifeq ($(TESTCASE), mgather_with_mask)
TARGET = $(ELF_DIR)/mgather_with_mask$(SUFFIX).elf
endif

# Source file: default is TESTCASE/TESTCASE.cpp
SRC_FILE +=  $(TEST_ROOT)/$(CATEGORY)/$(TESTCASE)/$(TESTCASE).cpp

# Special handling for hashtable_lookup_simd - embed data as object files
EXTRA_OBJ_FILES :=
EXTRA_OBJ_DEPS :=

# Data object files location (relative paths)
DATA_OBJ_DIR := hashtable_lookup_simd/data_obj
OUTPUT_DATA_OBJ_DIR := ../../../output/kernel/control/hashtable_lookup_simd/data_obj

# Generate .data files from Python script if missing
GEN_SCRIPT := hashtable_lookup_simd/gen_data.py
gen_sim_data:
	@if [ ! -f $(DATA_OBJ_DIR)/inserted_slot.data ] || [ ! -f $(DATA_OBJ_DIR)/lookup_keys.data ] || [ ! -f $(DATA_OBJ_DIR)/lookup_values.data ]; then \
		echo "Generating hashtable lookup data..."; \
		cd hashtable_lookup_simd && python3 gen_data.py; \
	fi

# hashtable_lookup_simd uses embedded data (large dataset for 2.55M-entry table)
ifeq ($(TESTCASE), hashtable_lookup_simd)
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/inserted_slot.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/lookup_keys.o
EXTRA_OBJ_FILES += $(OUTPUT_DATA_OBJ_DIR)/lookup_values.o

pre_work: build_sim_data_objs

build_sim_data_objs: gen_sim_data
	@COMPILER_DIR="$(COMPILER_DIR)" TARGET_TRIPLE="$(TARGET_TRIPLE)" \
		RESOURCE_DIR="$(RESOURCE_DIR)" $(DATA_OBJ_DIR)/build_data_obj.sh $(DATA_OBJ_DIR) $(OUTPUT_DATA_OBJ_DIR)

# The helper emits both assembly and objects in OUTPUT_DATA_OBJ_DIR.
$(EXTRA_OBJ_FILES): build_sim_data_objs
	@test -s "$@"

endif

# hkv uses embedded data
HKV_DATA_OBJ_DIR := hkv/data_obj
HKV_OUTPUT_DATA_OBJ_DIR := ../../../output/kernel/control/hkv/data_obj

ifeq ($(TESTCASE), hkv)
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/buckets.bin.o
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/buckets_size.bin.o
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/lookup_keys.bin.o
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/lookedup_values.bin.o
EXTRA_OBJ_FILES += $(HKV_OUTPUT_DATA_OBJ_DIR)/key_score_digest.bin.o

pre_work: build_hkv_data_objs

build_hkv_data_objs:
	@COMPILER_DIR="$(COMPILER_DIR)" $(HKV_DATA_OBJ_DIR)/build_data_obj.sh $(HKV_DATA_OBJ_DIR) $(HKV_OUTPUT_DATA_OBJ_DIR)

$(HKV_OUTPUT_DATA_OBJ_DIR)/%.o: $(HKV_DATA_OBJ_DIR)/%.s pre_work
	@mkdir -p $(shell dirname $@)
	$(AS) $(CC_O_ALL) $(INCLUDE) $(DEFINES) $< -o $@

endif

include ../../common/Makefile.common

DEFINES += $(EXTRA_DEFINES)

# Hashtable arithmetic and indexed fields are compiled from scalar pragma loops.
ifeq ($(PLAT),linx)
ifeq ($(TESTCASE),hashtable_lookup_simd)
ELEMENT_TARGET_FLAGS = $(if $(TARGET_TRIPLE),--target=$(TARGET_TRIPLE)) \
	$(if $(SYSROOT),--sysroot=$(SYSROOT)) \
	$(if $(RESOURCE_DIR),-resource-dir=$(RESOURCE_DIR))
INCLUDE += $(if $(API_INCLUDE),-I$(API_INCLUDE))
CC_OPTS += $(ELEMENT_TARGET_FLAGS) -mllvm -linxv5-enable-generic-element=true \
	-fno-vectorize -fno-slp-vectorize -Werror=unknown-pragmas
CC_LINK += $(ELEMENT_TARGET_FLAGS) -fuse-ld=lld
endif
endif
