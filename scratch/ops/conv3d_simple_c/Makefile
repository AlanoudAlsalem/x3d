# Minimal int8 3D conv: software reference + FPGA-offload stub + compare test.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
TARGET   = test_conv3d_simple
OBJS     = conv3d_sw.o conv3d_fpga.o main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c conv3d_simple.h
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run clean
