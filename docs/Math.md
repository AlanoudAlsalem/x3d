
### How Data is Stored in Memory

Every single piece of data in this network is a **5-dimensional array** of 32-bit floating-point numbers. The five dimensions are always in this exact order:

**(B, C, T, H, W)**

- **B** = Batch size. How many video clips you're processing at once. Usually 1.
- **C** = Channels. Think of these like "layers" of the image. A raw RGB video has 3 channels (red, green, blue). Deeper in the network, you might have 24, 48, 96, or 192 channels — these are abstract "feature" channels the network has learned.
- **T** = Time. Number of frames. Always 16 in this network, and it never changes.
- **H** = Height. Spatial height in pixels. Starts at 224, gets halved at each stage.
- **W** = Width. Spatial width in pixels. Starts at 224, gets halved at each stage.

**How this maps to physical memory:** NumPy stores this in row-major (C-contiguous) order. That means the data is laid out as one giant flat array of float32 values, and the rightmost index (W) changes fastest. To find the memory address of element `[b][c][t][h][w]`, you compute:

```
address = b*(C*T*H*W) + c*(T*H*W) + t*(H*W) + h*(W) + w
```

So if you're reading one "frame" (a single 2D image for one channel at one time step), that's a contiguous block of `H*W` floats in memory. This matters hugely for FPGA — when you stream data, reading along W is sequential memory access (fast, burst-friendly), while jumping between channels requires skipping `T*H*W` floats.

**Weights/kernels** are also stored as contiguous float32 arrays, shaped **(C_out, C_per_group, kT, kH, kW)** where C_per_group = C_in / groups.

### Convs
The number of independent convolution groups determines the following:
1. groups=1 is the standard convolution where every input channel is convolved with every output filter
2. With groups=in_channels the convolution is depthwise, where each input channel is convolved independently by its own filter (https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec)


#### conv_t
Spatial convolution which processes the (H, W) dimensions.
 - **Kernel size:** (1, 3, 3) — one in time, three in height, three in width
- **Stride:** (1, 2, 2) — no temporal stride, halves H and W
- **Padding:** (0, 1, 1) — no temporal padding, 1 pixel on each side of H and W
- **Groups:** 1 (standard convolution)
- **In channels:** 3 (RGB)
- **Out channels:** 24
- **Weight shape:** (24, 3, 1, 3, 3)
- **Number of multiply-accumulate (MAC) operations per output pixel:** 3 channels × 1 × 3 × 3 = 27 multiplications, 26 additions

Take one frame with RGB channels (3 channels), convolve with a 3x3x3 kernel. With a stride of 2, there is a downsampling of 224→112 in the width and height. This is done 24 times for each of the 16 frames. Hence, it is like having 24 kernels of size 3x3x3 applied to each of the 16 frames

#### conv_xy
Temporal convolution in the (T) dimension
- **Kernel size:** (5, 1, 1) — five in time, one in height, one in width
- **Stride:** (1, 1, 1) — no downsampling anywhere
- **Padding:** (2, 0, 0) — 2 frames of zeros on each side of T, nothing spatial
- **Groups:** 24 (depthwise — equals in_channels)
- **In channels:** 24
- **Out channels:** 24
- **Weight shape:** (24, 1, 5, 1, 1) — each kernel is just 5 numbers
- **MACs per output pixel:** 1 channel × 5 × 1 × 1 = 5 multiplications, 4 additions

Look at 5 consecutive frames at a single (H, W) position and multiply by 5 weights in the kernel. With a padding of 2 and stride of 1, the temporal dimension is preserved at 16. This is repeated for all 24 channels

**Input:** (1, 24, 16, 112, 112) **Output:** (1, 24, 16, 112, 112)

#### conv_a & conv_c
- Kernel size: (1x1x1) - in height, width, and time 
- Groups = 1

At each (t,w,h) position, it multiplies all input channels by a weight, and sums the numbers up into a single output value. If the output channels are 54, it does this 54 times i.e. 54 kernels. 


### conv_b
- **Kernel size:** (3, 3, 3) — 3 frames, 3 pixels high, 3 pixels wide
- **Stride:** (1, stride, stride) where stride is 1 or 2. First block in each stage uses stride=2 to halve spatial dims.
- **Padding:** (1, 1, 1) — 1 on every side in every dimension
- **Groups:** inner_channels (depthwise)
- **In channels:** inner_channels (e.g., 54)
- **Out channels:** inner_channels (e.g., 54)
- **Weight shape:** (54, 1, 3, 3, 3) — each kernel is 27 numbers
- **MACs per output pixel:** 1 × 3 × 3 × 3 = 27 multiplications, 26 additions

This is the actual 3D convolution, but it is depthwise so it is applied to each cannel independently. For each channel independently, it slides a 3×3×3 cube over the (T, H, W) volume. At each position, it takes the 27 values inside the cube, multiplies by the 27 kernel weights, sums them up. Because it's depthwise, channel 0 only talks to channel 0, channel 1 only talks to channel 1, etc.