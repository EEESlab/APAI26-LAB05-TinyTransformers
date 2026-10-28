APP = pulp_llama2

TRAIN_LIB= ./pulp-trainlib/lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources

APP_SRCS = main.c
APP_SRCS += conf_and_weights_fp16.c
APP_SRCS += Tokenizer.c
APP_SRCS += Sampler.c
APP_SRCS += MHSA.c
APP_SRCS += net_utils.c
APP_SRCS += net.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_rmsnorm_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c

APP_LDFLAGS += -lm

NUM_CORES ?= 8
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)

# enable performance counters:
#APP_CFLAGS += -DSTATS

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include

# fp16 math only: without it, double constants (e.g. M_PI) need fp16<->double conversions missing in the toolchain
APP_CFLAGS += -fsingle-precision-constant
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3

# Prepare the lab (download and patch PULP-TrainLib, see initialization.sh)
INIT_OUTPUT := $(shell bash $(CURDIR)/initialization.sh 2>&1)
ifneq ($(INIT_OUTPUT),)
$(info $(INIT_OUTPUT))
endif

# The lab-local linker script chips/pulp/link.ld (larger L2) is picked up instead of the PULP SDK one,
# since the linker looks for -Tchips/pulp/link.ld in the current directory first.

# Regenerate conf_and_weights_fp16.h/.c from the model in Model/ (check the L1 buffers in net.c against the printed minimum sizes)
get_header:
	gcc genHeader.c -o genHeader
	./genHeader

include $(RULES_DIR)/pmsis_rules.mk
