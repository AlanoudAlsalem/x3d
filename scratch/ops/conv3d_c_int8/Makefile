# Makefile — Build libconv3d_int8.so (int8 3D convolution with requantization)
#
# Mirrors conv3d_c/Makefile but builds the int8 variant.
#
# Targets:
#   make          Auto-detect architecture (RISC-V, x86_64, or ARM)
#   make riscv    Force RISC-V RV64GC flags (native or cross-compile)
#   make native   Force -march=native (x86_64 testing)
#   make clean    Remove build artefacts

CC       ?= gcc
UNAME_M  := $(shell uname -m)

CFLAGS_BASE = -O3 -funroll-loops -ffast-math -fPIC -Wall
LDFLAGS     = -lpthread

# Architecture auto-detection
ifeq ($(UNAME_M),riscv64)
  ARCH_FLAGS = -march=rv64gc
else ifeq ($(UNAME_M),x86_64)
  ARCH_FLAGS = -march=native
else
  ARCH_FLAGS =
endif

TARGET = libconv3d_int8.so

.PHONY: all riscv native clean

all: $(TARGET)

$(TARGET): conv3d_int8.c conv3d_int8.h
	$(CC) $(CFLAGS_BASE) $(ARCH_FLAGS) -shared -o $@ conv3d_int8.c $(LDFLAGS)

# Explicit RISC-V target (cross-compile or native on PolarFire SoC)
riscv:
	$(CC) $(CFLAGS_BASE) -march=rv64gc -shared -o $(TARGET) conv3d_int8.c $(LDFLAGS)

# Explicit x86 native target for testing on dev machine
native:
	$(CC) $(CFLAGS_BASE) -march=native -shared -o $(TARGET) conv3d_int8.c $(LDFLAGS)

clean:
	rm -f $(TARGET)
