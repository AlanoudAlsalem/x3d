## How Data is Stored in Memory

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

**Weights/kernels** are also stored as contiguous float32 arrays, shaped **(C_out, C_per_group, kT, kH, kW)** where C_per_group = C_in / groups.---

## Part 2: The 5 Convolution Types — Exact Math

Every convolution in this network is a **3D convolution**, meaning the kernel slides over three spatial dimensions (T, H, W). But the different "types" are really just 3D convolutions with different kernel shapes, different stride settings, and a critical parameter called **groups**. Let me explain groups first, then each conv type.

### What "groups" means

**groups=1 (standard convolution):** Every output channel looks at ALL input channels. If you have 24 input channels and 54 output channels, each of the 54 output kernels has shape `(24, kT, kH, kW)` — it reads all 24 input channels and combines them into one output number.

**groups=C_in (depthwise convolution):** Each output channel looks at exactly ONE input channel. If you have 54 input channels and 54 output channels with groups=54, each kernel has shape `(1, kT, kH, kW)`. Channel 0 of the input is convolved with kernel 0 to produce channel 0 of the output. Channel 1 with kernel 1. And so on. No mixing between channels at all.

This is the single most important concept for understanding the different conv types.

### The universal convolution formula

For every convolution, the output size is computed as:

```
T_out = (T_in + 2*pad_t - kT) / stride_t + 1
H_out = (H_in + 2*pad_h - kH) / stride_h + 1
W_out = (W_in + 2*pad_w - kW) / stride_w + 1
```

And the actual computation for a single output pixel at position `[b, oc, t, h, w]`:

```
output[b, oc, t, h, w] = bias[oc] +
    SUM over c in [0..C_per_group-1]:
        SUM over dt in [0..kT-1]:
            SUM over dh in [0..kH-1]:
                SUM over dw in [0..kW-1]:
                    input_padded[b, group_start + c, t*stride_t + dt, h*stride_h + dh, w*stride_w + dw]
                    * weight[oc, c, dt, dh, dw]
```

where `group_start = (oc / (C_out/groups)) * C_per_group`. For standard conv (groups=1), group_start is always 0 and C_per_group = C_in. For depthwise (groups=C_in), group_start = oc and C_per_group = 1.

Now, every conv in this network has **bias=False** (no bias term, because BatchNorm comes right after and absorbs it). So you can ignore the bias[oc] term everywhere.

### conv_t — Spatial convolution in the Stem

Despite the confusing name (inherited from PyTorchVideo), `conv_t` processes the **spatial** (H, W) dimensions.

- **Kernel size:** (1, 3, 3) — one in time, three in height, three in width
- **Stride:** (1, 2, 2) — no temporal stride, halves H and W
- **Padding:** (0, 1, 1) — no temporal padding, 1 pixel on each side of H and W
- **Groups:** 1 (standard convolution)
- **In channels:** 3 (RGB)
- **Out channels:** 24
- **Weight shape:** (24, 3, 1, 3, 3)
- **Number of multiply-accumulate (MAC) operations per output pixel:** 3 channels × 1 × 3 × 3 = 27 multiplications, 26 additions

What it does in plain English: for each of the 24 output channels, it takes a 3×3 window of ALL 3 color channels (RGB) at a single time step, multiplies element-by-element with a 3×3×3 kernel, sums everything up, and writes one output number. Since stride is (1,2,2), it skips every other pixel spatially, so 224→112.

**Input:** (1, 3, 16, 224, 224) — 3 channels, 16 frames, 224×224 **Output:** (1, 24, 16, 112, 112) — 24 channels, 16 frames, 112×112

The temporal kernel size is 1, so each output frame depends on exactly one input frame. Frame 0 of the output is computed only from frame 0 of the input.

### conv_xy — Temporal depthwise convolution in the Stem

Again, confusing name — `conv_xy` actually processes the **temporal** (T) dimension.

- **Kernel size:** (5, 1, 1) — five in time, one in height, one in width
- **Stride:** (1, 1, 1) — no downsampling anywhere
- **Padding:** (2, 0, 0) — 2 frames of zeros on each side of T, nothing spatial
- **Groups:** 24 (depthwise — equals in_channels)
- **In channels:** 24
- **Out channels:** 24
- **Weight shape:** (24, 1, 5, 1, 1) — each kernel is just 5 numbers
- **MACs per output pixel:** 1 channel × 5 × 1 × 1 = 5 multiplications, 4 additions

What it does: for each channel independently (because depthwise), it looks at 5 consecutive frames at a single (h, w) pixel position, multiplies by 5 weights, sums them up. It's essentially a 1D convolution along the time axis, done separately per channel and per spatial position.

**Input:** (1, 24, 16, 112, 112) **Output:** (1, 24, 16, 112, 112) — same shape because stride=1 and padding=2 preserves T=16

With padding=2, the input is padded from 16 to 20 frames (2 zeros on each side). Output frames = (20 - 5)/1 + 1 = 16. So the temporal dimension is preserved.

### conv_a — Pointwise expansion in the Bottleneck

- **Kernel size:** (1, 1, 1) — a single point, no spatial or temporal extent
- **Stride:** (1, 1, 1)
- **Padding:** (0, 0, 0)
- **Groups:** 1 (standard convolution)
- **In channels:** varies (e.g., 24 in Stage 2)
- **Out channels:** inner_channels (e.g., 54 in Stage 2)
- **Weight shape:** e.g., (54, 24, 1, 1, 1)
- **MACs per output pixel:** C_in × 1 × 1 × 1 = C_in multiplications

What it does: at each (t, h, w) position, it reads all C_in channel values (e.g., 24 numbers), multiplies each by a weight, and sums them into one output number. It does this 54 times (once per output channel) with 54 different sets of weights. It's a pure channel-mixing operation — no spatial or temporal context at all. Think of it as a tiny fully-connected layer applied identically at every pixel position.

**It does NOT change T, H, or W.** Only changes C.

Example: input (1, 24, 16, 56, 56) → output (1, 54, 16, 56, 56)

### conv_b — Depthwise 3×3×3 in the Bottleneck

This is where the actual spatial and temporal feature extraction happens.

- **Kernel size:** (3, 3, 3) — 3 frames, 3 pixels high, 3 pixels wide
- **Stride:** (1, stride, stride) where stride is 1 or 2. First block in each stage uses stride=2 to halve spatial dims.
- **Padding:** (1, 1, 1) — 1 on every side in every dimension
- **Groups:** inner_channels (depthwise)
- **In channels:** inner_channels (e.g., 54)
- **Out channels:** inner_channels (e.g., 54)
- **Weight shape:** (54, 1, 3, 3, 3) — each kernel is 27 numbers
- **MACs per output pixel:** 1 × 3 × 3 × 3 = 27 multiplications, 26 additions

What it does: for each channel independently, it slides a 3×3×3 cube over the (T, H, W) volume. At each position, it takes the 27 values inside the cube, multiplies by the 27 kernel weights, sums them up. Because it's depthwise, channel 0 only talks to channel 0, channel 1 only talks to channel 1, etc.

When stride=(1,2,2): T stays the same (always), but H and W are halved. For example, (1, 54, 16, 56, 56) → (1, 54, 16, 28, 28).

When stride=(1,1,1): everything stays the same size due to padding=1.

### conv_c — Pointwise projection in the Bottleneck

- **Kernel size:** (1, 1, 1)
- **Stride:** (1, 1, 1)
- **Padding:** (0, 0, 0)
- **Groups:** 1 (standard convolution)
- **In channels:** inner_channels (e.g., 54)
- **Out channels:** out_channels (e.g., 24)
- **Weight shape:** e.g., (24, 54, 1, 1, 1)
- **MACs per output pixel:** inner_channels multiplications

Identical in structure to conv_a, but it projects back down to a smaller channel count. It's the inverse of conv_a — conv_a expands channels, conv_c compresses them back.

### Other convolutions

There are a few more convolution layers that don't have the conv_a/b/c/t/xy naming:

**branch1_conv (skip connection):** A 1×1×1 pointwise conv used only in the first block of each stage, where in_channels ≠ out_channels (or stride ≠ 1). It transforms the shortcut path to match the dimensions of the bottleneck output so they can be added together. Same math as conv_a/conv_c.

**SE conv1 and conv2 (Squeeze-and-Excitation):** Two 1×1×1 pointwise convolutions. These operate on a (B, C, 1, 1, 1) tensor (the global average has already been computed), so they're basically just matrix-vector multiplications. SE conv1 reduces channels (e.g., 54 → 8), SE conv2 expands back (8 → 54). These DO have bias=True, unlike all other convs.

**Head pre_conv and post_conv:** Two more 1×1×1 pointwise convolutions. pre_conv goes 192→432, post_conv goes 432→2048. Standard, groups=1.---

## Part 3: The Stem — Step by Step

The input to the entire network is **(1, 3, 16, 224, 224)** — one video clip, 3 color channels (RGB), 16 frames, 224×224 pixels per frame.

### Step 3.1: conv_t (spatial conv)

**What's loaded from memory:**

- Input tensor: 3 × 16 × 224 × 224 = 2,408,448 floats (≈9.2 MB)
- Weights: 24 × 3 × 1 × 3 × 3 = 648 floats (≈2.5 KB)

**What happens:**

1. First, the input is padded. Padding is (0, 1, 1), meaning: no padding on T, one row/column of zeros added on all four sides of H and W. New padded shape: (1, 3, 16, 226, 226).
2. For each of the 24 output channels, for each of the 16 frames, the engine reads a 3×3 window across all 3 input channels from the padded input, multiplies by the 3×3×3 kernel (3 channels × 3 × 3 = 27 weights), sums everything to produce one output pixel.
3. Because stride=(1,2,2), the window jumps by 2 pixels in H and W. So output is 112×112 spatially.
4. For one output channel, one frame: 112 × 112 = 12,544 output positions, each requiring 27 MACs = 338,688 MACs.
5. Total for conv_t: 24 output channels × 16 frames × 338,688 = ~130 million MACs.

**Output: (1, 24, 16, 112, 112)** — 24 channels, 16 frames, 112×112.

Since the temporal kernel size is 1, every frame is computed independently. Frame 5 of the output depends only on frame 5 of the input.

### Step 3.2: conv_xy (temporal depthwise conv)

**What's loaded from memory:**

- Input: the output from conv_t, 24 × 16 × 112 × 112 = 4,816,896 floats (≈18.4 MB)
- Weights: 24 × 1 × 5 × 1 × 1 = 120 floats (480 bytes — tiny!)

**What happens:**

1. Padding (2, 0, 0): two frames of zeros are prepended and appended to the time dimension. New padded shape: (1, 24, 20, 112, 112).
2. For each of the 24 channels independently (depthwise!), for each (h, w) position, the engine reads 5 consecutive frame values, multiplies by the 5-element kernel, and sums. That's 5 MACs per output pixel.
3. Stride is (1,1,1), so output is 16 frames × 112 × 112.
4. Total MACs: 24 channels × 16 × 112 × 112 × 5 = ~24 million MACs.

**Output: (1, 24, 16, 112, 112)** — same shape. Channels still don't interact.

### Step 3.3: BatchNorm

**Parameters loaded:** 4 vectors, each of length 24 (the channel count): gamma (weight), beta (bias), running_mean, running_var. That's 96 floats total.

**What happens at each position [b, c, t, h, w]:**

```
output = gamma[c] * (input - running_mean[c]) / sqrt(running_var[c] + 1e-5) + beta[c]
```

This is entirely per-element. For each channel, there's one mean, one variance, one scale, one shift — applied identically to every (t, h, w) position. It's 3 arithmetic operations per element (subtract, divide, multiply-add). No spatial/temporal dependencies at all.

### Step 3.4: ReLU

The simplest operation: `output = max(0, input)`. Every element is processed independently. If the number is negative, it becomes 0. If positive, it stays the same.

**Stem final output: (1, 24, 16, 112, 112)**---

## Part 4: The Four Stages After the Stem

Each stage is a series of ResBlocks (residual blocks). Each ResBlock contains a Bottleneck (conv_a → conv_b → conv_c with norms and activations) plus a skip connection. I'll describe the big picture here, and go into the full Bottleneck detail in Part 5.

### Stage 2 (blocks[1]): 3 ResBlocks

- **Input:** (1, 24, 16, 112, 112) from the Stem
- **Inner channels:** 54
- **Out channels:** 24
- **First block stride:** 2 (halves H,W from 112→56)
- **Remaining blocks stride:** 1

Block 0: input (1, **24**, 16, **112**, 112) → conv_a expands to 54 ch → conv_b depthwise 3×3×3 with stride 2 (→ 56×56) → conv_c projects to 24 ch → output (1, **24**, 16, **56**, 56). Because in_channels=24 and out_channels=24 are the same BUT stride=2, this block needs a branch1_conv (1×1×1, stride (1,2,2)) on the skip path to downsample the shortcut. SE is applied (block index 0 is even).

Block 1: input (1, 24, 16, 56, 56) → same bottleneck with stride=1 → output (1, 24, 16, 56, 56). No branch1_conv needed (same channels, same spatial size). No SE (block index 1 is odd).

Block 2: same as Block 1 but with SE (index 2 is even).

**Stage 2 output: (1, 24, 16, 56, 56)**

### Stage 3 (blocks[2]): 5 ResBlocks

- **Input:** (1, 24, 16, 56, 56)
- **Inner channels:** 108
- **Out channels:** 48
- **First block stride:** 2 (56→28)

Block 0: (1, 24, 16, 56, 56) → expand to 108 → depthwise 3×3×3 stride 2 → project to 48 → (1, 48, 16, 28, 28). Branch1_conv needed (24→48 and stride 2). With SE.

Blocks 1-4: (1, 48, 16, 28, 28) → (1, 48, 16, 28, 28). SE on even indices (2, 4).

**Stage 3 output: (1, 48, 16, 28, 28)**

### Stage 4 (blocks[3]): 11 ResBlocks — the heaviest stage

- **Input:** (1, 48, 16, 28, 28)
- **Inner channels:** 216
- **Out channels:** 96
- **First block stride:** 2 (28→14)

Block 0: (1, 48, 16, 28, 28) → expand to 216 → depthwise 3×3×3 stride 2 → project to 96 → (1, 96, 16, 14, 14). Branch1_conv. SE.

Blocks 1-10: (1, 96, 16, 14, 14) → same. That's 10 more blocks, each with inner_channels=216. SE on even indices. This is where most of the compute goes because 216 channels × 3×3×3 depthwise, repeated 11 times.

**Stage 4 output: (1, 96, 16, 14, 14)**

### Stage 5 (blocks[4]): 7 ResBlocks

- **Input:** (1, 96, 16, 14, 14)
- **Inner channels:** 432
- **Out channels:** 192
- **First block stride:** 2 (14→7)

Block 0: (1, 96, 16, 14, 14) → expand to 432 → depthwise 3×3×3 stride 2 → project to 192 → (1, 192, 16, 7, 7). Branch1_conv. SE.

Blocks 1-6: (1, 192, 16, 7, 7) → same. SE on even indices.

**Stage 5 output: (1, 192, 16, 7, 7)**

### Head (blocks[5])

1. **pre_conv:** 1×1×1 pointwise, 192→432 channels. Standard. Output: (1, 432, 16, 7, 7).
2. **BatchNorm + ReLU** on the 432-channel output.
3. **AvgPool3d** with kernel (16, 7, 7): averages over the entire T×H×W volume per channel. This means: for each of the 432 channels, take all 16×7×7 = 784 values and compute their average. Output: (1, 432, 1, 1, 1).
4. **post_conv:** 1×1×1 pointwise, 432→2048 channels. Output: (1, 2048, 1, 1, 1).
5. **ReLU.**
6. **Dropout** (50%, disabled during inference — just passes data through).
7. **Reshape** to (1, 2048) — flatten.
8. **Linear layer** (fully connected): 2048→400. Weight shape (400, 2048), bias shape (400). Each of the 400 outputs is: `sum(weight[class_i, :] * input[:]) + bias[class_i]`. That's 2048 MACs per output, 400 outputs = 819,200 MACs total.

**Final output: (1, 400)** — 400 class logits for Kinetics-400 action recognition.---

## Part 5: Inside a Bottleneck Block — Every Detail

Let me use a concrete example: **Stage 3, Block 0** (the first block of Stage 3, which has the dimension change). Input is (1, 24, 16, 56, 56), inner=108, out=48, stride=2.

### Step 5.1: conv_a — Pointwise Expansion (1×1×1, groups=1)

**Weight tensor:** shape (108, 24, 1, 1, 1) = 2,592 floats stored contiguously.

**Data loading pattern for one output pixel at [b=0, oc=37, t=5, h=20, w=30]:**

- Read 24 input values: `input[0, 0..23, 5, 20, 30]`. In memory, these are NOT contiguous — they're spaced `T*H*W = 16*56*56 = 50,176` floats apart (because you're stepping along the C dimension).
- Read 24 weight values: `weight[37, 0..23, 0, 0, 0]`. These ARE contiguous in memory.
- Multiply pairwise and sum: `output = w[0]*in[0] + w[1]*in[1] + ... + w[23]*in[23]`

**This is a dot product.** 24 multiplications, 23 additions.

Do this for all 108 output channels × 16 frames × 56 × 56 positions = 108 × 50,176 = 5,419,008 output pixels, each needing 24 MACs.

**Total MACs: ~130 million.**

**Output: (1, 108, 16, 56, 56)**

### Step 5.2: norm_a — BatchNorm3d(108)

Loads 4 vectors of 108 values each (gamma, beta, running_mean, running_var).

For every single element in the (1, 108, 16, 56, 56) tensor:

```
normalized = (value - running_mean[channel]) / sqrt(running_var[channel] + 0.00001)
output = gamma[channel] * normalized + beta[channel]
```

This can be precomputed as a simple multiply-add per element: `output = scale[c] * value + offset[c]` where `scale[c] = gamma[c] / sqrt(running_var[c] + eps)` and `offset[c] = beta[c] - scale[c] * running_mean[c]`.

**For FPGA: BN can be fused with the preceding conv.** Instead of storing the conv output then doing BN as a second pass, you can fold the BN scale/offset into the conv computation. Each output pixel from conv just gets one extra multiply and one extra add.

**Output: (1, 108, 16, 56, 56)** — same shape.

### Step 5.3: ReLU

`output = max(0, input)` per element. Zero cost in terms of data movement — can be done inline as data exits the previous stage.

### Step 5.4: conv_b — Depthwise 3×3×3 with stride (1,2,2)

**Weight tensor:** shape (108, 1, 3, 3, 3) = 2,916 floats. 108 independent kernels, each just 27 numbers.

**Padding (1, 1, 1):** the input (1, 108, 16, 56, 56) becomes (1, 108, 18, 58, 58) by adding zeros on all six faces.

**Data loading pattern for output pixel at [b=0, oc=37, t=5, h=10, w=15]:**

- Because depthwise, we only read from channel 37 of the input.
- Stride is (1, 2, 2), so the window starts at: t_start = 5×1 = 5, h_start = 10×2 = 20, w_start = 15×2 = 30 in the padded input.
- Read a 3×3×3 cube: `padded_input[0, 37, 5:8, 20:23, 30:33]` — that's 27 values.
- Read the 27 kernel weights: `weight[37, 0, 0:3, 0:3, 0:3]`.
- Multiply pairwise and sum. Result is one output number.

**Memory access pattern for reading the 3×3×3 cube:** The 3 values along W at each (t,h) position are contiguous. But stepping to the next row (h+1) jumps by 58 floats, and stepping to the next frame (t+1) jumps by 58×58 = 3,364 floats. So the 27 values come from 9 separate memory locations (3 time steps × 3 rows), each a burst of 3 contiguous floats.

**Output size:** T_out = (18-3)/1 + 1 = 16. H_out = (58-3)/2 + 1 = 28. W_out = same = 28.

**Total MACs:** 108 channels × 16 × 28 × 28 × 27 = ~36.7 million.

**Output: (1, 108, 16, 28, 28)** — spatial halved because stride=2.

### Step 5.5: norm_b — BatchNorm3d(108) + Squeeze-and-Excitation

**BatchNorm** same as step 5.2, but on the (1, 108, 16, 28, 28) tensor.

**SE block (applied on even-indexed blocks only):**

1. **Global Average Pool:** For each of 108 channels, average ALL values across T×H×W = 16×28×28 = 12,544 positions. Output: (1, 108, 1, 1, 1). This is 108 averages, each summing 12,544 numbers.
    
2. **SE conv1:** 1×1×1 pointwise, 108→8 channels (with bias). `mid = round_width(108, 0.0625) = round(6.75) = 8`. Weight shape (8, 108, 1, 1, 1). This is 8 dot products of length 108. Output: (1, 8, 1, 1, 1).
    
3. **ReLU** on those 8 values.
    
4. **SE conv2:** 1×1×1 pointwise, 8→108 channels (with bias). Weight shape (108, 8, 1, 1, 1). 108 dot products of length 8. Output: (1, 108, 1, 1, 1).
    
5. **Sigmoid:** `scale = 1 / (1 + exp(-value))` on each of 108 values. Produces a number between 0 and 1 for each channel.
    
6. **Multiply:** The original (1, 108, 16, 28, 28) tensor is multiplied element-wise by the (1, 108, 1, 1, 1) scale tensor (broadcasting — each channel's scale is applied to all T×H×W positions).
    

SE is a lightweight attention mechanism. It learns "this frame has a lot of activity in channel 37, so let's amplify channel 37 and suppress channel 82."

### Step 5.6: SiLU (Swish) activation

`output = input * sigmoid(input) = input / (1 + exp(-input))`

Per-element, like ReLU, but smoother. Unlike ReLU (which zeroes negatives), SiLU allows small negative values through.

### Step 5.7: conv_c — Pointwise Projection (1×1×1, groups=1)

**Weight tensor:** shape (48, 108, 1, 1, 1) = 5,184 floats.

Same pattern as conv_a but in reverse — reads 108 channel values at each position, multiplies by weights, sums to produce one of 48 output channels. This is the "information compression" step.

**Total MACs:** 48 × 16 × 28 × 28 × 108 = ~65.3 million.

**Output: (1, 48, 16, 28, 28)**

### Step 5.8: norm_c — BatchNorm3d(48)

Same deal, 48-channel BN. Output: (1, 48, 16, 28, 28).

### Step 5.9: Skip Connection + ReLU (the ResBlock wrapper)

Now we step outside the bottleneck back into the ResBlock.

The original input was (1, 24, 16, 56, 56). The bottleneck output is (1, 48, 16, 28, 28). These don't match (different channels, different spatial size), so the skip path needs a transformation:

**branch1_conv:** 1×1×1, in=24, out=48, stride=(1,2,2), no padding. Weight shape (48, 24, 1, 1, 1). Takes the original input, reduces spatial by 2, changes channels from 24→48. Output: (1, 48, 16, 28, 28).

**branch1_norm:** BatchNorm3d(48) on the skip path output.

Then: **output = ReLU(bottleneck_output + skip_output)**. Element-wise addition of two (1, 48, 16, 28, 28) tensors, then ReLU.

**ResBlock output: (1, 48, 16, 28, 28)**---

## Part 6: Parallelism Opportunities for FPGA

Here's every type of parallelism you can exploit, from simplest to most impactful.

### 6.1: Element-wise parallelism (ReLU, SiLU, BN, residual add)

All of these operations work on each element independently. You can process as many elements simultaneously as your FPGA has compute units. If you have 64 multiply-add units, you can do 64 BN outputs in the same clock cycle. There are zero data dependencies between elements (for BN in inference mode — the mean/var are precomputed constants).

**Where:** After every conv layer (BN, activation, residual add). **How much:** Process N elements per cycle where N = your parallelism width.

### 6.2: Output-channel parallelism (conv_a, conv_c, conv_t, branch1_conv, head convs)

For standard (groups=1) convolutions, each output channel is computed completely independently from every other output channel. They all read the same input data, but use different weights and write to different output locations.

**Practical meaning:** If conv_a has 108 output channels, you could instantiate 4 "MAC arrays" on the FPGA, each computing a different output channel from the same input data. They all read the same input values on the same clock cycle, just multiply by different weights.

**This is the single biggest parallelism win for pointwise convs.** Pointwise convs (conv_a, conv_c) dominate compute in this network. Since they're just dot products, you can do multiple output channels in parallel by broadcasting the same input values to multiple weight banks.

### 6.3: Spatial parallelism (all conv types)

Every output position (t, h, w) is independent of every other output position. You could compute output[t=0,h=0,w=0] and output[t=0,h=0,w=1] at the same time. The only overlap is that their input windows may share some values (for 3×3 kernels, adjacent positions share 2/3 of their inputs).

**For FPGA:** You can use a sliding window / line buffer. As input data streams in row by row, you keep a small buffer of the last few rows and compute multiple output pixels as the window slides. This is the classic "systolic array" approach.

### 6.4: Temporal parallelism (conv_b, conv_xy)

For depthwise 3×3×3 convolutions (conv_b), each of the 16 output frames depends only on 3 input frames (its local temporal window). Frames that don't share input frames can be computed simultaneously. Even frames that share some input can be partially parallelized using the line buffer approach.

For conv_xy (5×1×1), each output frame depends on 5 input frames. But all 112×112 spatial positions for a given output frame are independent.

**Practical meaning:** You can process multiple time steps in parallel, each with its own set of MAC units, sharing the same kernel weights.

### 6.5: Channel parallelism (conv_b depthwise)

For depthwise convolutions, every channel is completely independent. Channel 0 uses kernel 0 on input channel 0. Channel 1 uses kernel 1 on input channel 1. No interaction at all.

**Practical meaning:** You can have multiple conv engines on the FPGA, each processing a different channel simultaneously. For conv_b with 216 inner channels in Stage 4, if you have 4 engines, you process 4 channels at a time and finish in 54 rounds instead of 216.

### 6.6: Within-dot-product parallelism (conv_a, conv_c)

A pointwise conv is a dot product. For conv_a in Stage 4, that's a dot product of length 48 (48 input channels). You can split this into partial sums: multiply elements 0-11 in parallel, 12-23 in parallel, etc., then add the partial sums. This is a "tree reduction."

**Practical meaning:** If your dot product length is 48, you could use 16 multipliers computing 16 products in parallel, do that 3 times (48/16=3 cycles for the multiplies), then combine the 48 partial products in a tree adder in ~log2(48)≈6 cycles.

### 6.7: Kernel decomposition (conv_b 3×3×3)

The 3D convolution in conv_b can be decomposed. A 3×3×3 kernel has 27 weights, but you can think of it as 3 separate 3×3 2D kernels (one per time step), each applied to a 2D frame, with results summed. This is exactly what the software does with `cv2.filter2D` — it loops over the 3 time offsets, does a 2D convolution for each, and accumulates.

**For FPGA:** 2D convolution with a 3×3 kernel is extremely well-studied. You can use a 3×3 systolic array or a row-buffer architecture that computes 2D convolution in a streaming fashion with exactly 9 multiply-add units. Then you need 3 of these (one per temporal offset) running in parallel, with their outputs summed.

### 6.8: Pipeline parallelism (between layers)

Conv_a, BN, ReLU, conv_b, BN, SE, SiLU, conv_c, BN — these are sequential. But you don't have to finish ALL of conv_a before starting conv_b. As soon as conv_a produces a spatial tile of output (say, a 4×4 patch for all channels at one time step), that tile can be fed into BN → ReLU → conv_b while conv_a is still computing the next tile.

**Practical meaning:** The FPGA can have all stages "running" simultaneously, each working on a different spatial region of the data. This is pipelining — the same concept as a CPU instruction pipeline, but applied to neural network layers.

### 6.9: Summary table of what parallelism applies where

|Layer|Output-ch parallel|Channel parallel|Spatial parallel|Temporal parallel|Dot-product parallel|Pipeline-able|
|---|---|---|---|---|---|---|
|conv_t (1×3×3, groups=1)|Yes (24 output ch)|No (standard)|Yes|Yes (kT=1, all frames independent)|Yes (length 9)|Yes|
|conv_xy (5×1×1, depthwise)|N/A|Yes (24 ch)|Yes|Partially (window=5)|Trivial (length 5)|Yes|
|conv_a (1×1×1, groups=1)|Yes|No (standard)|Yes|Yes (kT=1)|Yes (length=C_in)|Yes|
|conv_b (3×3×3, depthwise)|N/A|Yes (inner_ch)|Yes|Partially (window=3)|Trivial (length 27)|Yes|
|conv_c (1×1×1, groups=1)|Yes|No (standard)|Yes|Yes (kT=1)|Yes (length=inner_ch)|Yes|
|SE pool|N/A|Yes|N/A (global)|N/A (global)|N/A|Limited|
|SE conv1/conv2|Yes|No|N/A (1×1×1 input)|N/A|Yes|Yes|
|BN/ReLU/SiLU|All elements independent — unlimited parallelism|||||Yes|

That covers everything. To recap the key takeaways for your FPGA implementation:

**There are really only 3 distinct compute patterns you need to build in hardware:**

1. **Pointwise (1×1×1) standard conv** — used by conv_a, conv_c, branch1_conv, SE convs, and head convs. It's a dot product at each spatial position. This is the most compute-intensive operation overall. Parallelize across output channels and within the dot product.
    
2. **Depthwise 3D conv** — used by conv_b (3×3×3) and conv_xy (5×1×1). Each channel is independent. Decompose the 3D conv into a series of 2D convolutions across time steps. Parallelize across channels and spatial positions with line buffers.
    
3. **Element-wise ops** — BN (multiply-add), ReLU (threshold), SiLU (multiply by sigmoid), residual add. All trivially parallel. Fuse BN into the preceding conv to save a memory round-trip.
    

The biggest thing to watch for on this specific hardware (PolarFire with 2GB LPDDR4): the activation tensors are large early on (Stage 2: 24×16×112×112 ≈ 4.8M floats ≈ 19MB) but shrink rapidly. By Stage 5, they're 192×16×7×7 ≈ 150K floats ≈ 600KB, which fits comfortably in on-chip memory. Design your dataflow accordingly — tile aggressively in the early stages, and you may be able to keep everything on-chip in the later ones.