

### Unique conv configurations (26 total Conv3d instances)

| # instances | Type | In→Out Ch | Kernel | Stride | Pad | Groups | Input Shape |
|:-----------:|------|-----------|--------|--------|-----|--------|-------------|
| 1 | conv_t | 3→24 | (1,3,3) | (1,2,2) | (0,1,1) | 1 | (1,3,16,224,224) |
| 1 | conv_xy | 24→24 | (5,1,1) | (1,1,1) | (2,0,0) | 24 | (1,24,16,112,112) |
| 1 | branch1 | 24→24 | (1,1,1) | (1,2,2) | (0,0,0) | 1 | (1,24,16,112,112) |
| 1 | branch1 | 24→48 | (1,1,1) | (1,2,2) | (0,0,0) | 1 | (1,24,16,56,56) |
| 1 | branch1 | 48→96 | (1,1,1) | (1,2,2) | (0,0,0) | 1 | (1,48,16,28,28) |
| 1 | branch1 | 96→192 | (1,1,1) | (1,2,2) | (0,0,0) | 1 | (1,96,16,14,14) |
| 1 | conv_a | 24→54 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,24,16,112,112) |
| 2 | conv_a | 24→54 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,24,16,56,56) |
| 1 | conv_a | 24→108 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,24,16,56,56) |
| 4 | conv_a | 48→108 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,48,16,28,28) |
| 1 | conv_a | 48→216 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,48,16,28,28) |
| 10 | conv_a | 96→216 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,96,16,14,14) |
| 1 | conv_a | 96→432 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,96,16,14,14) |
| 6 | conv_a | 192→432 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,192,16,7,7) |
| 1 | conv_b | 54→54 dw | (3,3,3) | (1,2,2) | (1,1,1) | 54 | (1,54,16,112,112) |
| 2 | conv_b | 54→54 dw | (3,3,3) | (1,1,1) | (1,1,1) | 54 | (1,54,16,56,56) |
| 1 | conv_b | 108→108 dw | (3,3,3) | (1,2,2) | (1,1,1) | 108 | (1,108,16,56,56) |
| 4 | conv_b | 108→108 dw | (3,3,3) | (1,1,1) | (1,1,1) | 108 | (1,108,16,28,28) |
| 1 | conv_b | 216→216 dw | (3,3,3) | (1,2,2) | (1,1,1) | 216 | (1,216,16,28,28) |
| 10 | conv_b | 216→216 dw | (3,3,3) | (1,1,1) | (1,1,1) | 216 | (1,216,16,14,14) |
| 1 | conv_b | 432→432 dw | (3,3,3) | (1,2,2) | (1,1,1) | 432 | (1,432,16,14,14) |
| 6 | conv_b | 432→432 dw | (3,3,3) | (1,1,1) | (1,1,1) | 432 | (1,432,16,7,7) |
| 3 | conv_c | 54→24 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,54,16,56,56) |
| 5 | conv_c | 108→48 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,108,16,28,28) |
| 11 | conv_c | 216→96 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,216,16,14,14) |
| 7 | conv_c | 432→192 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,432,16,7,7) |
| 1 | pre_conv | 192→432 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,192,16,7,7) |
| 1 | post_conv | 432→2048 | (1,1,1) | (1,1,1) | (0,0,0) | 1 | (1,432,1,1,1) |

For example, for the first instance of conv_t, the parameters are: 
``` c
    const int B      = 1;
    const int C_in   = 3;
    const int T      = 16;
    const int H      = 224;
    const int W      = 224;
    const int C_out  = 24;
    const int kT = 1, kH = 3, kW = 3;
    const int stride_t = 1, stride_h = 2, stride_w = 2;
    const int pad_t    = 0, pad_h    = 1, pad_w    = 1;
    const int groups   = 1;
```

Branch1 is the skip connection (also called the shortcut) inside a ResBlock. Every ResBlock computes two branches and adds them together:

`output = ReLU(branch2(x) + branch1(x))`

branch2 is the bottleneck path (conv_a → conv_b → conv_c). branch1 is the identity shortcut, it passes the input through unchanged so the residual learning works. But, if the input and output have different channel counts or different spatial dimensions (because of a stride-2 downsample), you can't just add them since the shapes don't match. In that case, branch1 becomes a 1×1×1 Conv3d that reshapes the input to match the bottleneck's output.

This only happens on the first ResBlock of each stage, because that's where the spatial resolution halves (stride 2) and/or the channel count changes. Specifically:

- Stage 2 block 0: spatial 112→56 but channels stay 24→24, so branch1 is just a stride-2 1×1×1 conv (no channel change, just downsampling).
- Stage 3 block 0: spatial 56→28 and channels 24→48, so branch1 is a stride-2 1×1×1 conv that also expands channels.
- Stage 4 block 0: 28→14, 48→96. Same idea.
- Stage 5 block 0: 14→7, 96→192. Same idea.

Every other ResBlock (block 1, 2, 3, …) within a stage has matching input/output dimensions, so branch1 is just the identity i.e. no conv at all.

In terms testing, branch1 convs are structurally identical to conv_a or conv_c, they are all 1×1×1 pointwise convolutions with groups=1. The only difference is that branch1 convs have stride=(1,2,2) while conv_a/conv_c always have stride=(1,1,1).