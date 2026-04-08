# Makefile — Build libconv3d_fpga.so (int8 FPGA-offload convolution)
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

TARGET = libconv3d_fpga.so

.PHONY: all riscv native clean

all: $(TARGET)

$(TARGET): conv3d_fpga.c conv3d_fpga.h
	$(CC) $(CFLAGS_BASE) $(ARCH_FLAGS) -shared -o $@ conv3d_fpga.c $(LDFLAGS)

riscv:
	$(CC) $(CFLAGS_BASE) -march=rv64gc -shared -o $(TARGET) conv3d_fpga.c $(LDFLAGS)

native:
	$(CC) $(CFLAGS_BASE) -march=native -shared -o $(TARGET) conv3d_fpga.c $(LDFLAGS)

clean:
	rm -f $(TARGET)
