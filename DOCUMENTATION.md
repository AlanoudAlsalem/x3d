# X3D-M Scratch Library: Comprehensive Technical Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background Concepts](#2-background-concepts)
   - 2.1 [Neural Networks Fundamentals](#21-neural-networks-fundamentals)
   - 2.2 [Convolutional Neural Networks (CNNs)](#22-convolutional-neural-networks-cnns)
   - 2.3 [From 2D to 3D CNNs: Understanding Video](#23-from-2d-to-3d-cnns-understanding-video)
   - 2.4 [Residual Networks (ResNets)](#24-residual-networks-resnets)
   - 2.5 [Depthwise Separable Convolutions](#25-depthwise-separable-convolutions)
   - 2.6 [Batch Normalization](#26-batch-normalization)
   - 2.7 [Activation Functions](#27-activation-functions)
   - 2.8 [Pooling Operations](#28-pooling-operations)
   - 2.9 [Squeeze-and-Excitation (SE) Attention](#29-squeeze-and-excitation-se-attention)
   - 2.10 [Dropout Regularization](#210-dropout-regularization)
3. [The X3D Architecture Family](#3-the-x3d-architecture-family)
   - 3.1 [Design Philosophy](#31-design-philosophy)
   - 3.2 [X3D-M Variant Specification](#32-x3d-m-variant-specification)
   - 3.3 [Complete Data Flow with Tensor Shapes](#33-complete-data-flow-with-tensor-shapes)
   - 3.4 [Parameter Count Analysis](#34-parameter-count-analysis)
4. [The Scratch Library: Architecture and Design](#4-the-scratch-library-architecture-and-design)
   - 4.1 [Why "From Scratch"? Motivation and Design Goals](#41-why-from-scratch-motivation-and-design-goals)
   - 4.2 [Directory Structure](#42-directory-structure)
   - 4.3 [Two-Layer Design: ops vs nn](#43-two-layer-design-ops-vs-nn)
5. [Low-Level Operations (scratch/ops/)](#5-low-level-operations-scratchops)
   - 5.1 [3D Convolution (conv3d.py)](#51-3d-convolution-conv3dpy)
   - 5.2 [Batch Normalization (batchnorm3d.py)](#52-batch-normalization-batchnorm3dpy)
   - 5.3 [Activation Functions (activations.py)](#53-activation-functions-activationspy)
   - 5.4 [Pooling (pooling.py)](#54-pooling-poolingpy)
   - 5.5 [Linear / Fully Connected (linear.py)](#55-linear--fully-connected-linearpy)
   - 5.6 [Dropout (dropout.py)](#56-dropout-dropoutpy)
6. [Neural Network Layers (scratch/nn/)](#6-neural-network-layers-scratchnn)
   - 6.1 [Module Base Class (module.py)](#61-module-base-class-modulepy)
   - 6.2 [Sequential and ModuleList (sequential.py)](#62-sequential-and-modulelist-sequentialpy)
   - 6.3 [Conv3d Layer (conv3d.py)](#63-conv3d-layer-conv3dpy)
   - 6.4 [BatchNorm3d Layer (batchnorm3d.py)](#64-batchnorm3d-layer-batchnorm3dpy)
   - 6.5 [Squeeze-Excitation Block (squeeze_excitation.py)](#65-squeeze-excitation-block-squeeze_excitationpy)
   - 6.6 [Bottleneck Block (bottleneck.py)](#66-bottleneck-block-bottleneckpy)
   - 6.7 [Residual Block (resblock.py)](#67-residual-block-resblockpy)
   - 6.8 [Residual Stage (resstage.py)](#68-residual-stage-resstagepy)
   - 6.9 [Stem (stem.py)](#69-stem-stempy)
   - 6.10 [Head (head.py)](#610-head-headpy)
7. [Full Model Assembly (scratch/models/x3d_m.py)](#7-full-model-assembly-scratchmodelsx3d_mpy)
8. [Weight Loading and Conversion](#8-weight-loading-and-conversion)
   - 8.1 [Weight Conversion Script](#81-weight-conversion-script)
   - 8.2 [Weight Loading on the SoC](#82-weight-loading-on-the-soc)
9. [Profiling and Statistics (stats.py)](#9-profiling-and-statistics-statspy)
10. [Visualization (visualize_stats.py)](#10-visualization-visualize_statspy)
11. [Inference Entry Point (main.py)](#11-inference-entry-point-mainpy)
12. [PyTorch Reference Implementation (x3d_layers.py)](#12-pytorch-reference-implementation-x3d_layerspy)
13. [PolarFire SoC Icicle Kit: Hardware Background](#13-polarfire-soc-icicle-kit-hardware-background)
    - 13.1 [Architecture Overview](#131-architecture-overview)
    - 13.2 [The RISC-V Processor Cores](#132-the-risc-v-processor-cores)
    - 13.3 [Memory Subsystem](#133-memory-subsystem)
    - 13.4 [The FPGA Fabric](#134-the-fpga-fabric)
    - 13.5 [Why This Matters for Neural Network Inference](#135-why-this-matters-for-neural-network-inference)
14. [Multi-Threading Acceleration Opportunities](#14-multi-threading-acceleration-opportunities)
    - 14.1 [Threading Context on the PolarFire SoC](#141-threading-context-on-the-polarfire-soc)
    - 14.2 [Strategy 1: Output-Channel Parallelism in conv3d_forward_fast](#142-strategy-1-output-channel-parallelism-in-conv3d_forward_fast)
    - 14.3 [Strategy 2: Temporal Parallelism in conv3d_core](#143-strategy-2-temporal-parallelism-in-conv3d_core)
    - 14.4 [Strategy 3: Adaptive Hybrid Parallelism](#144-strategy-3-adaptive-hybrid-parallelism)
    - 14.5 [Additional Acceleration Techniques](#145-additional-acceleration-techniques)
    - 14.6 [Python Threading and the GIL](#146-python-threading-and-the-gil)
    - 14.7 [Estimated Impact](#147-estimated-impact)
15. [C Backend: Native Convolution via ctypes](#15-c-backend-native-convolution-via-ctypes)
    - 15.1 [Motivation and Architecture](#151-motivation-and-architecture)
    - 15.2 [Building the Shared Library](#152-building-the-shared-library)
    - 15.3 [C Implementation Details](#153-c-implementation-details)
    - 15.4 [Python ctypes Wrapper](#154-python-ctypes-wrapper)
    - 15.5 [Selecting the Convolution Method](#155-selecting-the-convolution-method)
16. [Int8 Post-Training Quantization (PTQ)](#16-int8-post-training-quantization-ptq)
17. [Archive and Legacy Code](#17-archive-and-legacy-code)
18. [Glossary](#18-glossary)
19. [Int8 Quantized Runtime (scratch/quantized/)](#19-int8-quantized-runtime-scratchquantized)
    - 19.1 [Why a Separate Quantized Runtime?](#191-why-a-separate-quantized-runtime)
    - 19.2 [Architecture Overview](#192-architecture-overview)
    - 19.3 [The Software Reference Int8 Convolution Kernel](#193-the-software-reference-int8-convolution-kernel-conv3d_int8py)
    - 19.4 [QuantizedConv3d Layer](#194-quantizedconv3d-layer-layerspy)
    - 19.5 [QuantizedLinear Layer](#195-quantizedlinear-layer-layerspy)
    - 19.6 [The QuantizedX3D_M Model](#196-the-quantizedx3d_m-model-modelpy)
    - 19.7 [Int8 Weight Loader](#197-int8-weight-loader-load_int8_weightspy)
20. [Int8 Inference Entry Point (main_int8.py)](#20-int8-inference-entry-point-main_int8py)
21. [FPGA Per-Layer Validation Harness (fpga_tests/)](#21-fpga-per-layer-validation-harness-fpga_tests)
    - 21.1 [The Problem It Solves](#211-the-problem-it-solves)
    - 21.2 [Four Execution Paths](#212-four-execution-paths)
    - 21.3 [Quantization Primitives (quant.py)](#213-quantization-primitives-quantpy)
    - 21.4 [Three Int8 Convolution Kernels (kernels.py)](#214-three-int8-convolution-kernels-kernelspy)
    - 21.5 [Layer Configurations (layer_configs.py)](#215-layer-configurations-layer_configspy)
    - 21.6 [The Test Runner (test_layer.py)](#216-the-test-runner-test_layerpy)
22. [FPGA Int8 C Backend](#22-fpga-int8-c-backend-scratchopsconv3d_fpga_c-and-conv3d_fpgapy)
    - 22.1 [Purpose](#221-purpose)
    - 22.2 [C Implementation](#222-c-implementation-conv3d_fpgac)
    - 22.3 [Python ctypes Wrapper](#223-python-ctypes-wrapper-conv3d_fpgapy)
    - 22.4 [Building](#224-building)
23. [Minimal Int8 C Test Harness](#23-minimal-int8-c-test-harness-testing-and-scratchopsconv3d_simple_c)
    - 23.1 [Purpose](#231-purpose)
    - 23.2 [Architecture](#232-architecture)
    - 23.3 [Test Harness (main.c)](#233-test-harness-mainc)
    - 23.4 [Building and Running](#234-building-and-running)
24. [Quantization Validation Script](#24-quantization-validation-script-scriptsvalidate_quantizationpy)
25. [Profiling Dashboard (dashboard.py)](#25-profiling-dashboard-dashboardpy)
26. [Complete Convolution Layer Catalog](#26-complete-convolution-layer-catalog)
    - 26.1 [Stem Convolutions](#261-stem-convolutions)
    - 26.2 [Branch1 (Skip Connection) Convolutions](#262-branch1-skip-connection-convolutions)
    - 26.3 [Bottleneck Convolutions](#263-bottleneck-convolutions-conv_a-conv_b-conv_c)
    - 26.4 [Head Convolutions](#264-head-convolutions)
    - 26.5 [SE Convolutions](#265-se-convolutions-inside-even-indexed-blocks)
27. [FPGA Integration Plan](#27-fpga-integration-plan-fpga_flowmd)
    - 27.1 [Three-Phase Deployment](#271-three-phase-deployment)
    - 27.2 [CPU/FPGA Handshake Protocol](#272-cpufpga-handshake-protocol)
    - 27.3 [Freeze-One-Layer-At-A-Time Discipline](#273-freeze-one-layer-at-a-time-discipline)
28. [Updated File Layout](#28-updated-file-layout)
29. [Suggested Diagrams and Figures](#29-suggested-diagrams-and-figures)
30. [Summary of All Implemented Components](#30-summary-of-all-implemented-components)

---

## 1. Project Overview

This project is a complete, PyTorch-free implementation of the X3D-M (eXpand-3D, Medium variant) 3D convolutional neural network for video action recognition. The implementation is called the "scratch library" because every operation -- from convolution to batch normalization to activation functions -- is built from scratch using only NumPy and optionally OpenCV, without any deep learning framework dependency.

The primary purpose of this implementation is to enable inference of the X3D-M model on the Microchip PolarFire SoC Icicle Kit, a RISC-V based System-on-Chip with integrated FPGA fabric. PyTorch and other major deep learning frameworks do not support the RISC-V architecture, which necessitated this ground-up reimplementation.

The model takes a video clip as input -- specifically a tensor of shape `(B, 3, 16, 224, 224)` representing `B` video clips, each with 3 color channels (RGB), 16 temporal frames, and 224x224 pixel spatial resolution -- and outputs a vector of 400 class logits corresponding to the Kinetics-400 action recognition dataset. These logits can be converted to probabilities via the softmax function to classify human actions in video.

The total parameter count of the model is approximately 3.79 million, making it one of the most lightweight video classification models available. This efficiency is what makes it a candidate for edge deployment on resource-constrained hardware like the PolarFire SoC.

---

## 2. Background Concepts

### 2.1 Neural Networks Fundamentals

A neural network is a computational model inspired by the structure of biological neural systems. At its most basic level, a neural network consists of layers of interconnected "neurons" (also called nodes or units), where each connection carries a learned numerical weight. Data flows forward through the network: each neuron receives inputs, multiplies them by the corresponding weights, sums the results, optionally adds a bias term, and passes the output through a non-linear activation function.

The process of computing an output from an input is called the **forward pass** (or forward propagation). During training, the network's weights are adjusted using an algorithm called backpropagation, which computes the gradient of a loss function with respect to each weight and updates the weights to minimize the loss. During **inference** (which is what this project performs), the weights are fixed and the network simply computes outputs from inputs.

A neural network with multiple layers between its input and output is called a "deep" neural network, and the field of using such networks is called **deep learning**. Depth allows networks to learn hierarchical representations -- early layers learn simple features (like edges), middle layers learn combinations of those features (like textures and shapes), and deep layers learn high-level concepts (like objects and actions).

**Tensors** are the fundamental data structure in neural network computation. A tensor is a multi-dimensional array of numbers. A 0D tensor is a scalar, a 1D tensor is a vector, a 2D tensor is a matrix, and higher-dimensional tensors extend this pattern. In this project, the primary data tensors are 5-dimensional, with axes representing batch, channels, temporal depth, height, and width.

### 2.2 Convolutional Neural Networks (CNNs)

A Convolutional Neural Network is a specialized type of neural network designed for processing data that has a grid-like structure, such as images (2D grids of pixels) or video (3D grids of pixels over time). CNNs use **convolution operations** instead of general matrix multiplication in at least one of their layers.

**Convolution** is a mathematical operation that slides a small learnable filter (also called a kernel) across the input, computing a dot product at each position. For a 2D image convolution, a kernel might be a 3x3 grid of weights. This kernel slides across every position of the input image, and at each position, the 3x3 patch of the image under the kernel is multiplied element-wise with the kernel weights and summed to produce a single output value. The result is a new 2D grid called a **feature map** that highlights the presence of the pattern captured by the kernel.

Key properties of convolution that make it powerful for visual data are:

**Parameter sharing**: the same kernel (same set of weights) is applied at every spatial position. A 3x3 kernel has only 9 learnable parameters regardless of the input image size. This dramatically reduces the number of parameters compared to a fully connected layer and makes the model translation-invariant -- it can detect a feature regardless of where it appears in the image.

**Local connectivity**: each output value depends only on a small local region of the input (the "receptive field" of the kernel). This matches the observation that visual features are typically local in nature.

Important convolution parameters include:

- **Kernel size**: the dimensions of the filter (e.g., 3x3, 5x5, 1x1). Larger kernels capture broader spatial patterns.
- **Stride**: how many positions the kernel moves between applications. A stride of 1 moves one pixel at a time (preserving spatial resolution). A stride of 2 moves two pixels, halving the spatial resolution (spatial downsampling).
- **Padding**: adding zeros around the border of the input. With appropriate padding, the output can be kept the same size as the input. For a 3x3 kernel with stride 1, padding of 1 on each side preserves dimensions.
- **Groups**: the number of independent convolution groups. With `groups=1` (standard convolution), every input channel is convolved with every output filter. With `groups=in_channels` (depthwise convolution), each input channel is convolved independently by its own filter. See Section 2.5.

**Output size formula**: for a single spatial dimension with input size `I`, kernel size `K`, stride `S`, and padding `P`, the output size is `O = floor((I + 2P - K) / S) + 1`.

### 2.3 From 2D to 3D CNNs: Understanding Video

A standard 2D CNN processes a single image. The input to a 2D convolution is a 3D tensor of shape `(C, H, W)` -- channels, height, and width -- and the kernel is shaped `(C, kH, kW)`, sliding over the H and W dimensions.

Video data adds a temporal dimension: a video clip is a sequence of frames over time. To process video, we extend 2D convolutions to **3D convolutions**, which operate on tensors of shape `(C, T, H, W)` -- channels, time (frames), height, and width -- using kernels of shape `(C, kT, kH, kW)` that slide over all three spatial-temporal dimensions simultaneously. This allows the network to learn patterns that span both space and time, which is essential for understanding motion and temporal dynamics in video.

The tensor layout used throughout this project is `(B, C, T, H, W)`:
- `B`: batch size (number of video clips processed simultaneously)
- `C`: channels (3 for RGB input, or the number of learned feature channels in intermediate layers)
- `T`: temporal dimension (number of frames; always 16 in X3D-M)
- `H`: height (spatial)
- `W`: width (spatial)

**3D convolution** is significantly more computationally expensive than 2D convolution. A 2D convolution with kernel 3x3 on `C` channels requires `C * 9` multiplications per output position. A 3D convolution with kernel 3x3x3 on `C` channels requires `C * 27` multiplications per output position -- 3 times more, multiplied across a larger output volume. This is why efficient architectures like X3D use design techniques such as depthwise separable convolutions and (2+1)D factorization to reduce the computational cost.

**(2+1)D Factorization**: instead of applying a single 3D kernel (like 5x3x3 in the temporal-height-width dimensions), the convolution is factored into two separate operations: a 2D spatial convolution (kernel 1x3x3, operating only on height and width) followed by a 1D temporal convolution (kernel 5x1x1, operating only across time). This factorization has two benefits: it dramatically reduces the number of parameters (768 vs 3,240 for the example above, a ~4x reduction), and it doubles the number of non-linearities (an activation function can be applied between the two operations), which helps the network learn more complex features.

### 2.4 Residual Networks (ResNets)

Training very deep neural networks is notoriously difficult because of the **vanishing gradient problem**: as gradients are backpropagated through many layers, they tend to shrink exponentially, making it hard for early layers to learn. Residual Networks (ResNets), introduced by He et al. in 2015, solve this with a simple but powerful idea: **skip connections** (also called shortcut connections or residual connections).

Instead of learning a function `H(x)` directly, a residual block learns the residual `F(x) = H(x) - x`, so that the output is `y = F(x) + x`. The skip connection adds the input `x` directly to the output of the transformation `F(x)`. This means the gradient can flow directly through the skip connection (the identity path) during backpropagation, avoiding the vanishing gradient problem. If the optimal transformation is close to the identity function (which is common in deep networks), it is much easier for the network to learn a small residual `F(x) ≈ 0` than to learn the identity mapping from scratch.

In the X3D-M model, each **ResBlock** (residual block) has two paths:
1. **Branch 2** (main path / bottleneck): the input goes through a sequence of convolutions, batch normalizations, and activations. This computes the residual `F(x)`.
2. **Branch 1** (shortcut / skip connection): the input either passes through unchanged (identity) or through a 1x1x1 convolution to match the dimensions of the main path's output.

The two paths are added element-wise, and a ReLU activation is applied to the sum: `output = ReLU(branch2(x) + branch1(x))`.

A shortcut convolution (branch1) is only needed when the input and output dimensions differ, which happens in two cases: (a) when the spatial resolution changes (stride > 1, meaning downsampling), or (b) when the number of channels changes (e.g., from 24 to 48 at a stage boundary).

### 2.5 Depthwise Separable Convolutions

Standard convolution mixes information across both the spatial dimensions and the channel dimension simultaneously. A depthwise separable convolution splits this into two steps:

1. **Depthwise convolution**: each input channel is convolved independently with its own filter. If the input has `C` channels and the kernel size is `K x K x K`, this produces `C` output channels using `C * K^3` parameters (instead of `C * C * K^3` for a standard convolution). This performs spatial/temporal filtering without any cross-channel interaction.

2. **Pointwise convolution**: a 1x1x1 convolution that mixes information across channels. If the input has `C_in` channels and the output has `C_out` channels, this requires `C_in * C_out` parameters.

The key advantage is parameter efficiency. Consider a standard 3D convolution with 54 input and 54 output channels using a 3x3x3 kernel: that is `54 * 54 * 27 = 78,732` parameters. The depthwise separable equivalent is `54 * 27` (depthwise) + `54 * 54` (pointwise) = `1,458 + 2,916 = 4,374` parameters -- an 18x reduction.

In the X3D-M bottleneck blocks, the pattern is:
- `conv_a`: pointwise 1x1x1 convolution (expands channels, provides cross-channel mixing)
- `conv_b`: depthwise 3x3x3 convolution (`groups=inner_channels`, spatial/temporal filtering per channel)
- `conv_c`: pointwise 1x1x1 convolution (projects channels back down, provides cross-channel mixing)

This is sometimes called an "inverted bottleneck" because the middle layer (depthwise conv) operates in a higher-dimensional channel space than the input and output, which are at a narrower channel count.

In the code, depthwise convolution is implemented by setting the `groups` parameter equal to the number of input channels. When `groups=C`, the convolution kernel has shape `(C, 1, kT, kH, kW)` instead of `(C, C, kT, kH, kW)`, and each of the `C` input channels is convolved independently by its corresponding single-channel filter.

### 2.6 Batch Normalization

Batch Normalization (BatchNorm), introduced by Ioffe and Szegedy in 2015, is a technique that normalizes the activations within a mini-batch of training examples. It has become a near-ubiquitous component of modern deep learning architectures because it stabilizes training, allows higher learning rates, and acts as a regularizer.

For 3D data with shape `(B, C, T, H, W)`, BatchNorm operates independently per channel. For each channel `c`:

1. **Compute statistics**: calculate the mean `μ` and variance `σ²` across all elements in that channel across the entire mini-batch (across dimensions B, T, H, W).
2. **Normalize**: subtract the mean and divide by the standard deviation: `x̂ = (x - μ) / √(σ² + ε)`, where `ε` is a small constant (typically 1e-5) that prevents division by zero.
3. **Scale and shift**: apply two learned parameters per channel -- `γ` (gamma, called "weight" in the code) and `β` (beta, called "bias" in the code): `y = γ * x̂ + β`. These learnable parameters allow the network to undo the normalization if that is optimal, ensuring that BatchNorm cannot reduce the representational capacity of the network.

During training, the batch mean and variance are computed from the current mini-batch and also used to maintain **running statistics** (exponential moving averages of the mean and variance) via the momentum parameter: `running_mean = (1 - momentum) * running_mean + momentum * batch_mean`.

During inference (evaluation mode), the running statistics are used instead of computing batch statistics, because: (a) inference may process a single sample (batch size 1), making batch statistics unreliable, and (b) the model's output should be deterministic and not depend on what other samples happen to be in the batch.

BatchNorm's learnable parameters consist of a `weight` vector (gamma) and `bias` vector (beta), each of size `C` (the number of channels). Additionally, the `running_mean` and `running_var` are stored as buffers (not trainable parameters, but part of the model's state that must be saved and loaded).

### 2.7 Activation Functions

Activation functions introduce non-linearity into neural networks. Without non-linear activation functions, a multi-layer neural network would be equivalent to a single linear transformation, regardless of how many layers it has. Activation functions allow the network to learn complex, non-linear mappings from inputs to outputs.

Three activation functions are used in the X3D-M model:

**ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)`. The simplest and most widely used activation function. It outputs the input directly if positive, and zero otherwise. ReLU is computationally cheap, promotes sparsity (many zero activations), and mitigates the vanishing gradient problem (gradient is 1 for positive inputs). However, it can suffer from the "dying ReLU" problem where neurons that output zero for all inputs can never recover during training.

In the code: `np.maximum(0, x)`.

**Sigmoid**: `f(x) = 1 / (1 + e^(-x))`. Squashes any real number into the range (0, 1). Used in the Squeeze-and-Excitation blocks to produce per-channel attention weights that represent "how important is this channel?" as a value between 0 and 1. The code clips the input to [-500, 500] before computing the exponential to prevent numerical overflow.

In the code: `1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))`.

**SiLU (Sigmoid Linear Unit, also called Swish)**: `f(x) = x * sigmoid(x)`. A smooth, non-monotonic activation function that was shown to outperform ReLU in many deep learning architectures. Unlike ReLU, SiLU has a smooth curve near zero, which can improve gradient flow. It is used after the depthwise convolution in each bottleneck block.

In the code: `x * sigmoid(x)`.

### 2.8 Pooling Operations

Pooling reduces the spatial dimensions of a tensor by summarizing local regions. It serves two purposes: (1) reducing computational cost by making tensors smaller, and (2) introducing a degree of translation invariance.

**Average Pooling**: replaces each local region with the mean of its values. For 3D average pooling with a kernel size of `(kT, kH, kW)` and stride `(sT, sH, sW)`, a sliding window of that kernel size moves across the input with the given stride, and each position in the output is the average of all values in the window.

**Adaptive Average Pooling**: a variant where instead of specifying the kernel size, you specify the desired output size, and the kernel size and stride are computed automatically to achieve that output size. Adaptive pooling to output size `(1, 1, 1)` is called **global average pooling** because it averages all values across the entire spatial-temporal volume into a single value per channel.

In the X3D-M model, average pooling is used in two places:
1. In the head's ProjectedPool, `AvgPool3d(kernel_size=(16, 7, 7))` collapses the entire spatial-temporal volume `(16, 7, 7)` into `(1, 1, 1)`.
2. In the Squeeze-and-Excitation blocks, `AdaptiveAvgPool3d(output_size=1)` performs global average pooling as the "squeeze" step.

### 2.9 Squeeze-and-Excitation (SE) Attention

Squeeze-and-Excitation blocks, introduced by Hu et al. in 2018, add a form of channel-wise attention to a neural network. The core idea is that not all feature channels are equally important for a given input, so the network should learn to dynamically recalibrate channel responses.

The SE mechanism works in three steps:

1. **Squeeze**: global average pooling reduces each channel to a single scalar value, producing a channel descriptor of shape `(B, C, 1, 1, 1)`. This captures the "global context" of each channel across the entire spatial-temporal volume.

2. **Excitation**: two small fully-connected layers (implemented as 1x1x1 convolutions because the tensor is 5D) with a bottleneck structure learn to produce per-channel attention weights:
   - First layer: `C → C * se_ratio` with ReLU (compress to a bottleneck)
   - Second layer: `C * se_ratio → C` with Sigmoid (expand back and produce weights in [0, 1])

   The `se_ratio` is 0.0625 (1/16), and the bottleneck width is rounded to the nearest multiple of 8 for hardware efficiency. For example, 54 channels * 0.0625 = 3.375, rounded to 8.

3. **Scale**: the original feature map is multiplied element-wise by the attention weights. Broadcasting expands the `(B, C, 1, 1, 1)` weights across all spatial-temporal positions, so each channel is uniformly scaled by its learned importance weight.

In X3D-M, SE is applied to even-indexed blocks within each stage (blocks 0, 2, 4, ...), while odd-indexed blocks use an Identity (no-op) instead.

### 2.10 Dropout Regularization

Dropout is a regularization technique that helps prevent overfitting during training. With a probability `p` (0.5 in the X3D-M head), each activation is randomly set to zero during the forward pass. The remaining activations are scaled by `1/(1-p)` to maintain the expected sum. This forces the network to not rely on any single neuron and encourages learning redundant, robust representations.

During inference (evaluation mode), dropout is disabled -- all activations pass through unchanged. This is why the code checks `if not training or p == 0: return x`.

---

## 3. The X3D Architecture Family

### 3.1 Design Philosophy

X3D (eXpand 3D), introduced by Feichtenhofer in 2020 at Facebook AI Research (FAIR), is a family of efficient 3D CNN architectures designed through a progressive network expansion approach. Rather than designing a fixed architecture manually, X3D starts from a tiny base 2D image model and progressively expands it along multiple axes -- temporal duration, frame rate, spatial resolution, network width (channels), bottleneck width, and network depth (number of layers) -- to find an optimal configuration at each computational budget.

The key design principles are:

1. **Channel-wise separability**: X3D uses depthwise separable convolutions for all spatial-temporal filtering, dramatically reducing parameter counts and FLOPs compared to standard 3D convolutions.

2. **(2+1)D factorization in the stem**: the initial convolution that processes raw video is factored into a spatial convolution followed by a temporal convolution, reducing parameters by approximately 4x.

3. **Inverted bottleneck**: channels are expanded before the depthwise convolution and compressed after, so that the expensive spatial-temporal filtering happens in a richer feature space.

4. **Squeeze-and-Excitation attention**: channel attention on select blocks improves accuracy with minimal overhead.

5. **Temporal preservation**: unlike some video models that progressively downsample the temporal dimension, X3D-M maintains all 16 frames throughout the network. Temporal context is captured through the growing receptive field of stacked 3x3x3 convolutions.

### 3.2 X3D-M Variant Specification

X3D comes in several size variants (XS, S, M, L, XL). This implementation uses X3D-M ("Medium"), which represents a good balance between accuracy and efficiency. The architecture is:

```
Block          Depth   Input Channels   Inner Channels   Output Channels   Spatial Stride
─────────────────────────────────────────────────────────────────────────────────────────
Stem           -       3                -                24                2
Stage 2        3       24               54               24                2
Stage 3        5       24               108              48                2
Stage 4        11      48               216              96                2
Stage 5        7       96               432              192               2
Head           -       192              432 / 2048       400               -
```

Total depth: 26 residual blocks (3 + 5 + 11 + 7), plus the stem and head.

The model classifies video clips into the 400 action categories of the Kinetics-400 dataset, which includes actions like "playing basketball", "cooking", "dancing", "painting", etc.

### 3.3 Complete Data Flow with Tensor Shapes

For a single input clip (`B=1`):

```
Layer                              Output Shape              Notes
───────────────────────────────────────────────────────────────────────────────
Input                              (1, 3, 16, 224, 224)      RGB video, 16 frames
Stem: conv_t (1x3x3, stride 1,2,2) (1, 24, 16, 112, 112)   Spatial filter + downsample
Stem: conv_xy (5x1x1, depthwise)  (1, 24, 16, 112, 112)    Temporal filter
Stem: BatchNorm + ReLU             (1, 24, 16, 112, 112)    Normalize and activate

Stage 2 Block 0 (stride=2, SE)    (1, 24, 16, 56, 56)      Spatial 112→56
Stage 2 Block 1 (stride=1)        (1, 24, 16, 56, 56)      Same resolution
Stage 2 Block 2 (stride=1, SE)    (1, 24, 16, 56, 56)      Same resolution

Stage 3 Block 0 (stride=2, SE)    (1, 48, 16, 28, 28)      Spatial 56→28, channels 24→48
Stage 3 Blocks 1-4                 (1, 48, 16, 28, 28)      Same resolution

Stage 4 Block 0 (stride=2, SE)    (1, 96, 16, 14, 14)      Spatial 28→14, channels 48→96
Stage 4 Blocks 1-10                (1, 96, 16, 14, 14)      Same resolution

Stage 5 Block 0 (stride=2, SE)    (1, 192, 16, 7, 7)       Spatial 14→7, channels 96→192
Stage 5 Blocks 1-6                 (1, 192, 16, 7, 7)       Same resolution

Head: pre_conv (1x1x1, 192→432)   (1, 432, 16, 7, 7)       Channel expansion
Head: BatchNorm + ReLU             (1, 432, 16, 7, 7)       Normalize and activate
Head: AvgPool3d (16x7x7)          (1, 432, 1, 1, 1)        Global pool
Head: post_conv (1x1x1, 432→2048) (1, 2048, 1, 1, 1)       Final expansion
Head: ReLU                         (1, 2048, 1, 1, 1)       Activate
Head: Dropout(0.5)                 (1, 2048, 1, 1, 1)       Regularize (training only)
Head: Flatten                      (1, 2048)                 Remove spatial dims
Head: Linear(2048→400)             (1, 400)                  Class logits
```

Key observation: the temporal dimension `T=16` never changes throughout the entire network. All downsampling happens in the spatial dimensions (H, W). The temporal receptive field grows organically through the stacking of 3x3x3 convolutions.

### 3.4 Parameter Count Analysis

The approximately 3.79 million parameters are distributed across the network:

- **Stem**: 3 * 24 * 1 * 3 * 3 (conv_t) + 24 * 1 * 5 * 1 * 1 (conv_xy) + 24 * 2 (BN) = 648 + 120 + 48 = **816 parameters**
- **Each ResBlock**: parameters from conv_a (pointwise), conv_b (depthwise), conv_c (pointwise), three BatchNorms, and optionally SE blocks
- **Head**: 192 * 432 (pre_conv) + 432 * 2 (pre_BN) + 432 * 2048 (post_conv) + 2048 * 400 + 400 (linear) = ~1.7M parameters

Stage 4 (with 11 blocks) and Stage 5 (with 7 blocks and the widest channels) contain the majority of the parameters.

---

## 4. The Scratch Library: Architecture and Design

### 4.1 Why "From Scratch"? Motivation and Design Goals

The scratch library exists because PyTorch, TensorFlow, and other major deep learning frameworks do not support the RISC-V instruction set architecture. The PolarFire SoC Icicle Kit's U54 RISC-V cores cannot run PyTorch. Therefore, every operation must be implemented using libraries that do work on RISC-V -- namely NumPy (which has RISC-V support via OpenBLAS) and optionally OpenCV.

The design goals of the scratch library are:

1. **Functional equivalence with PyTorch**: given the same weights, the scratch implementation must produce identical outputs (within floating-point tolerance) to the PyTorch implementation.
2. **No PyTorch dependency at inference time**: the only runtime dependencies are NumPy and optionally OpenCV.
3. **FPGA-readiness**: the convolution code is structured to facilitate offloading to the PolarFire's FPGA fabric, with explicit comments marking where the FPGA driver would be called.
4. **Readability and modularity**: the code mirrors PyTorch's module hierarchy for maintainability and to simplify weight loading from PyTorch checkpoints.

### 4.2 Directory Structure

```
x3d/
├── scratch/                     # The PyTorch-free neural network library
│   ├── ops/                     # Stateless mathematical operations
│   │   ├── conv3d.py            # 3D convolution (slow/fast/threaded/native)
│   │   ├── conv3d_c/            # C shared-library backend
│   │   │   ├── conv3d.c         # Pthreads implementation with tiling
│   │   │   ├── conv3d.h         # C header
│   │   │   └── Makefile         # Build for RISC-V or native x86
│   │   ├── batchnorm3d.py       # 3D batch normalization
│   │   ├── activations.py       # ReLU, SiLU, Sigmoid
│   │   ├── pooling.py           # Average pooling (3D and adaptive)
│   │   ├── linear.py            # Dense/fully-connected layer
│   │   └── dropout.py           # Dropout regularization
│   ├── nn/                      # Stateful neural network layers (with learnable parameters)
│   │   ├── module.py            # Base Module class (analogous to torch.nn.Module)
│   │   ├── sequential.py        # Sequential and ModuleList containers
│   │   ├── conv3d.py            # Conv3d layer (wraps ops/conv3d.py with parameters)
│   │   ├── batchnorm3d.py       # BatchNorm3d layer (wraps ops/batchnorm3d.py)
│   │   ├── squeeze_excitation.py # SE attention block
│   │   ├── bottleneck.py        # X3D bottleneck (conv_a → conv_b → conv_c pipeline)
│   │   ├── resblock.py          # Residual block (bottleneck + skip connection)
│   │   ├── resstage.py          # Stage of residual blocks
│   │   ├── stem.py              # (2+1)D factorized stem convolution
│   │   └── head.py              # Classification head (pool + linear)
│   ├── models/
│   │   └── x3d_m.py             # Full X3D-M model definition
│   ├── load_weights.py          # Load .npz pretrained weights (NumPy only)
│   └── stats.py                 # Profiling and statistics collection
├── main.py                      # Inference entry point with profiling
├── visualize_stats.py           # Statistics visualization and comparison
├── x3d_layers.py                # PyTorch reference implementation
├── scripts/
│   └── convert_pytorch_weights_to_numpy.py  # Weight conversion tool
└── archive/                     # Legacy C++/PyTorch implementations
```

### 4.3 Two-Layer Design: ops vs nn

The library is organized into two layers, mirroring PyTorch's own design:

**`scratch/ops/`** (operations): pure functions that take NumPy arrays and return NumPy arrays. These functions have no state -- they do not hold learnable parameters. They are the mathematical building blocks. For example, `conv3d_forward(x, weight, bias, stride, padding, groups)` takes all necessary data as explicit arguments.

**`scratch/nn/`** (neural network layers): classes that wrap the operations with persistent state (learnable parameters). Each class inherits from `Module` and stores its parameters (weights, biases, running statistics) in a `_parameters` dictionary. The `forward()` method calls the corresponding operation from `ops/` with the stored parameters. For example, `Conv3d.forward(x)` calls `conv3d_forward(x, self._parameters["weight"], ...)`.

This separation means the mathematical operations can be tested independently of the module system, and the module system handles parameter management, hierarchical composition, and training/evaluation mode switching.

---

## 5. Low-Level Operations (scratch/ops/)

### 5.1 3D Convolution (conv3d.py)

This is the most important and computationally intensive file in the entire project. It provides four selectable implementations of 3D convolution — `"slow"` (pure NumPy), `"fast"` (single-threaded OpenCV), `"threaded"` (multi-threaded OpenCV), and `"native"` (C shared library via ctypes) — behind a unified dispatch function `conv3d_forward()`.

#### 5.1.1 The _pad_3d Helper

```python
def _pad_3d(x, pad_t, pad_h, pad_w):
```

Zero-pads the input tensor on the T, H, and W dimensions. Padding is symmetric: `pad_t` zeros are added to both the beginning and end of the temporal dimension, `pad_h` to both sides of height, and `pad_w` to both sides of width. The function allocates a new, larger array filled with zeros and copies the original data into the center.

For example, an input of shape `(1, 24, 16, 112, 112)` with `pad_t=0, pad_h=1, pad_w=1` produces shape `(1, 24, 16, 114, 114)`.

#### 5.1.2 The Dispatch Function

```python
def conv3d_forward(x, weight, bias, stride, padding, groups, method=None):
```

This function dispatches to the selected implementation based on the `method` parameter (or the global default set by `set_conv3d_method()`). Four methods are available:

| Method | Function | Description |
|--------|----------|-------------|
| `"slow"` | `conv3d_forward_slow` | Pure NumPy, 6-deep nested loops |
| `"fast"` | `conv3d_forward_fast` | Single-threaded OpenCV `cv2.filter2D` **(default)** |
| `"threaded"` | `conv3d_forward_threaded` | Multi-threaded OpenCV with adaptive parallelism |
| `"native"` | `conv3d_forward_native` | C shared library (`libconv3d.so`) via ctypes |

The global default can be changed at runtime:

```python
from scratch import set_conv3d_method
set_conv3d_method("native")  # all subsequent calls use the C backend
```

Individual `Conv3d` layers can also override the global default via their `method` constructor parameter.

#### 5.1.3 The Slow Implementation (conv3d_forward_slow)

This is a straightforward, textbook implementation of 3D convolution using nested loops. It iterates over every output position `(b, oc, tt, hh, ww)` and computes the dot product of the input patch with the kernel:

```
for b in range(B):                   # Each sample in the batch
    for oc in range(out_c):          # Each output channel
        for c in range(c_per_group): # Each input channel in this group
            for tt in range(T_out):  # Each temporal output position
                for hh in range(H_out):  # Each spatial output row
                    for ww in range(W_out):  # Each spatial output column
                        # Extract the input patch and multiply with kernel
                        acc[tt, hh, ww] += np.sum(
                            input_patch[t0:t0+kT, h0:h0+kH, w0:w0+kW] * weight
                        )
```

This is a 6-deep nested loop. For a moderately-sized convolution (e.g., 216 output channels on a 16x14x14 spatial volume with 3x3x3 kernels), this results in billions of scalar operations and is extremely slow. It exists purely as a reference implementation for correctness verification and as a fallback for platforms where OpenCV is not available.

#### 5.1.4 The Fast Implementation (conv3d_forward_fast)

The fast implementation is specifically structured for FPGA offloading and now includes **adaptive multi-threaded parallelism** targeting the PolarFire SoC's 4 U54 RISC-V application cores (see Section 14 for the full design rationale). It separates the convolution into phases:

**Phase 1: Padding.** The input tensor is padded in software using `_pad_3d`.

**Phase 2: Adaptive strategy selection.** The function inspects the kernel size and group count to choose the optimal threading strategy:

```python
is_pointwise = kT * kH * kW == 1
is_depthwise = groups == out_c and groups > 1

if is_pointwise or not is_depthwise:
    _conv3d_oc_parallel(...)   # Strategy 1: output-channel parallelism
else:
    _conv3d_temporal_parallel(...)  # Strategy 2: temporal parallelism
```

- **Pointwise (1x1x1) and standard convolutions** use `_conv3d_oc_parallel`, which distributes `(batch, output_channel)` pairs across 4 threads. This covers `conv_a`, `conv_c`, the stem's `conv_t`, and all head convolutions.
- **Depthwise convolutions** (where `groups == out_channels`) use `_conv3d_temporal_parallel`, which parallelizes the temporal output positions within each `conv3d_core` call via `_conv3d_core_threaded`. This covers `conv_b` in every bottleneck block and the stem's `conv_xy`.

**Phase 3: Dense convolution.** Within each thread, the work follows the same structure as before -- extract the relevant input channels and kernel, call `conv3d_core` (or `_conv3d_core_threaded` for the depthwise path) to compute the dense stride-1 convolution.

**Phase 4: Apply stride.** The dense output is subsampled using NumPy slicing: `dense_out[::st, ::sh, ::sw]`.

This separation exists because FPGA hardware accelerators typically operate most efficiently on dense, stride-1 convolutions. Separating the stride application into a post-processing step (which is a simple memory access pattern) makes the FPGA design simpler.

The module maintains a persistent `ThreadPoolExecutor` with `NUM_THREADS = 4` workers at module scope, amortizing thread-creation overhead across all convolution calls during inference. NumPy and OpenCV both release the GIL during their C-level internals, so the Python threads achieve genuine parallelism on the 4 U54 cores (see Section 14.6).

#### 5.1.5 The Core Convolution (conv3d_core)

```python
def conv3d_core(volume, kernel):
```

This is the computational heart of the library. It decomposes the 3D convolution into a series of 2D convolutions using OpenCV's `cv2.filter2D`:

```
for c in range(C):                    # Each input channel
    for tt in range(T_out):           # Each temporal output position
        for dt in range(kT):          # Each temporal kernel depth
            k_2d = kernel[c, dt]      # Extract 2D spatial kernel slice
            if not np.any(k_2d):      # Skip zero kernels (optimization)
                continue
            # Apply 2D convolution using OpenCV
            filtered = cv2.filter2D(volume[c, tt + dt], cv2.CV_32F, k_2d, ...)
            out_volume[tt] += filtered[:H_out, :W_out]
```

The key insight is that a 3D convolution with kernel `(C, kT, kH, kW)` can be decomposed into `C * kT` individual 2D convolutions of kernel size `(kH, kW)`. For each input channel `c` and temporal kernel position `dt`, a 2D input slice `volume[c, tt + dt]` is convolved with the 2D kernel slice `kernel[c, dt]`, and the results are accumulated into the output at temporal position `tt`.

`cv2.filter2D` is OpenCV's optimized 2D correlation/convolution function. It uses SIMD instructions (SSE, AVX on x86; NEON on ARM) internally and is significantly faster than a pure Python loop. The `anchor=(0,0)` parameter positions the kernel at the top-left corner, and `borderType=cv2.BORDER_CONSTANT` zero-pads implicitly (though padding is handled externally by `_pad_3d`).

The `np.ascontiguousarray` calls ensure that the input data is stored contiguously in memory (C-order), which is necessary for OpenCV functions to operate correctly and efficiently.

#### 5.1.6 Multi-Threading Helpers

Three internal helper functions implement the adaptive hybrid parallelism described in Section 14.4. All three use the module-level `_thread_pool` (`ThreadPoolExecutor(max_workers=4)`).

**`_conv3d_oc_parallel`** (Strategy 1 -- output-channel parallelism):

```python
def _conv3d_oc_parallel(x_pad, weight, bias, out,
                        B, out_c, groups, c_per_group,
                        st, sh, sw):
```

Enumerates all `(b, oc)` pairs, divides them into `NUM_THREADS` contiguous chunks, and submits each chunk to the thread pool. Each thread iterates over its assigned `(b, oc)` pairs, calling `conv3d_core` for each and writing the strided result directly into `out[b, oc]`. Because different chunks write to non-overlapping slices of the output tensor, no synchronisation is needed. This strategy is used for pointwise (1x1x1) and standard convolutions, where individual `conv3d_core` calls are lightweight but there are many output channels to process.

**`_conv3d_temporal_parallel`** (Strategy 2 -- temporal parallelism wrapper):

```python
def _conv3d_temporal_parallel(x_pad, weight, bias, out,
                               B, out_c, groups, c_per_group,
                               st, sh, sw):
```

Iterates sequentially over `(b, oc)` pairs (since depthwise convolutions have moderate per-call work), but replaces `conv3d_core` with `_conv3d_core_threaded` to parallelise the temporal dimension within each call. This strategy is used for depthwise convolutions (where `groups == out_channels`).

**`_conv3d_core_threaded`** (temporal-parallel core):

```python
def _conv3d_core_threaded(volume, kernel):
```

A multi-threaded variant of `conv3d_core`. It partitions the `T_out` temporal output positions evenly across `NUM_THREADS` threads using integer boundaries: `boundaries[i] = i * T_out // NUM_THREADS`. Each thread computes its slice into a private `local_out` buffer (avoiding concurrent writes to shared memory), performing the full `C * kT` inner loop of `cv2.filter2D` calls for its assigned temporal range. The results are assembled back into `out_volume` after all threads complete. If `T_out < NUM_THREADS`, it falls back to the sequential `conv3d_core` to avoid thread-overhead on trivially small workloads.

For a typical depthwise 3x3x3 convolution with `T_out = 16`, each of the 4 threads processes 4 temporal positions, each accumulating 3 `cv2.filter2D` calls (one per `kT` slice) -- a clean, balanced split across the 4 U54 cores.

### 5.2 Batch Normalization (batchnorm3d.py)

```python
def batchnorm3d_forward(x, weight, bias, running_mean, running_var, eps, training, momentum):
```

Implements the BatchNorm formula described in Section 2.6. The function handles both training and inference modes:

**Training mode** (`training=True`): computes the per-channel mean and variance from the current mini-batch data, uses them for normalization, and updates the running statistics.

**Inference mode** (`training=False`): uses the stored running mean and running variance for normalization. This is the mode used in this project.

The normalization is implemented using NumPy broadcasting. `mean.reshape(1, C, 1, 1, 1)` reshapes the per-channel mean from shape `(C,)` to `(1, C, 1, 1, 1)`, which NumPy then broadcasts across all batch, temporal, height, and width dimensions during the subtraction `x - mean`. Similarly, division by `sqrt(var)` and the scale/shift by `weight` and `bias` are all broadcast operations.

### 5.3 Activation Functions (activations.py)

Three simple, stateless activation functions:

**`relu(x)`**: `np.maximum(0, x)` -- element-wise maximum of zero and the input.

**`sigmoid(x)`**: `1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))` -- the input is clipped to prevent numerical overflow in the exponential. Without clipping, very large negative values would cause `np.exp(500)` which overflows to infinity.

**`silu(x)`**: `x * sigmoid(x)` -- composed from the sigmoid function.

All three functions operate element-wise on arbitrarily shaped arrays and return arrays of the same shape.

### 5.4 Pooling (pooling.py)

#### avg_pool3d_forward

Implements fixed-kernel average pooling. Three nested loops iterate over each output position `(tt, hh, ww)`, and at each position, `np.mean` is computed over the corresponding `(kT, kH, kW)` window of the input. The `:, :` at the beginning of the indexing preserves the batch and channel dimensions, so the mean is taken only over the spatial-temporal axes `(2, 3, 4)`.

Output size formula: `T_out = (T - kT) // st + 1` (and similarly for H and W).

#### adaptive_avg_pool3d_forward

Instead of a fixed kernel size, this function accepts a desired output size `(oT, oH, oW)` and computes the necessary kernel boundaries for each output position using integer arithmetic:

```python
t_start = (tt * T) // oT
t_end = ((tt + 1) * T) // oT
```

This produces non-uniform window sizes that exactly tile the input to achieve the desired output dimensions. The most common use is `output_size=1`, which produces global average pooling (a single value per channel).

### 5.5 Linear / Fully Connected (linear.py)

```python
def linear_forward(x, weight, bias):
    return np.dot(x, weight.T) + bias
```

The simplest operation in the library. `np.dot(x, weight.T)` performs matrix multiplication between the input `x` of shape `(B, in_features)` and the transposed weight matrix of shape `(in_features, out_features)`, producing output of shape `(B, out_features)`. Adding the bias vector `(out_features,)` broadcasts across the batch dimension.

This is used in the classification head to project from the 2048-dimensional feature vector to the 400-dimensional class logit vector.

### 5.6 Dropout (dropout.py)

```python
def dropout_forward(x, p, training, rng=None):
```

During training, generates a binary mask where each element is 0 with probability `p` and 1 with probability `1-p`. The input is multiplied by this mask (zeroing out random elements) and divided by `(1-p)` to maintain the expected value (inverted dropout). An optional random number generator (`rng`) can be provided for reproducibility.

During inference (`training=False`) or when `p=0`, the input is returned unchanged.

---

## 6. Neural Network Layers (scratch/nn/)

### 6.1 Module Base Class (module.py)

The `Module` class is the foundation of the entire neural network layer system. It is analogous to `torch.nn.Module` in PyTorch and provides:

**`_parameters`**: a dictionary mapping parameter names (strings) to NumPy arrays. For example, a Conv3d layer stores `{"weight": np.ndarray, "bias": np.ndarray}`. This is how learnable weights are managed.

**`_modules`**: a dictionary mapping names to child `Module` instances. This creates a tree structure where the full model is the root and individual layers are leaves. This hierarchy is essential for recursive operations like parameter collection and mode switching.

**`parameters()`**: recursively collects all parameter arrays from this module and all child modules. Used for counting total parameters and for weight loading.

**`train(mode)`** and **`eval()`**: recursively set the `training` flag on this module and all children. This flag affects BatchNorm (whether to use batch or running statistics) and Dropout (whether to zero out activations).

**`forward(x)`**: the main computation method, which subclasses override to define their specific operation.

### 6.2 Sequential and ModuleList (sequential.py)

**`Sequential`**: chains multiple modules into a pipeline. Its `forward(x)` method passes the input through each contained module in order, with each module's output becoming the next module's input: `x = module_n(...(module_1(x)))`. Used in the bottleneck block to combine BatchNorm and SE into a single module.

**`ModuleList`**: stores a list of modules for iteration. Unlike Sequential, it does not define a `forward` method -- the parent module iterates over the list manually in its own `forward`. It provides `__iter__`, `__len__`, and `__getitem__` for indexed access. Used for the blocks list in the full model and for the residual blocks within each stage.

### 6.3 Conv3d Layer (conv3d.py)

The `Conv3d` class in `scratch/nn/` wraps the stateless `conv3d_forward` function with persistent parameters:

**Initialization**: allocates the weight tensor of shape `(out_channels, in_channels // groups, kT, kH, kW)` and optionally a bias tensor of shape `(out_channels,)`. Weights are initialized with Xavier-like initialization: uniform random values in `[-bound, bound]` where `bound = sqrt(1 / fan_in)` and `fan_in` is the total number of input elements per output neuron (i.e., `in_channels_per_group * kT * kH * kW`).

The `_triple` helper converts a single integer to a 3-tuple (e.g., `3` becomes `(3, 3, 3)`) for kernel_size, stride, and padding, matching PyTorch's behavior.

**Forward pass**: calls `conv3d_forward(x, self._parameters["weight"], self._parameters["bias"], self.stride, self.padding, self.groups)`.

### 6.4 BatchNorm3d Layer (batchnorm3d.py)

Wraps `batchnorm3d_forward` with four stored parameter arrays: `weight` (gamma, initialized to ones), `bias` (beta, initialized to zeros), `running_mean` (initialized to zeros), and `running_var` (initialized to ones).

### 6.5 Squeeze-Excitation Block (squeeze_excitation.py)

Implements the SE attention mechanism described in Section 2.9:

The **`_round_width`** helper computes the bottleneck channel count by multiplying the input width by the SE ratio and rounding to the nearest multiple of 8 (minimum 8). This rounding ensures hardware-friendly channel counts.

The block contains two `Conv3d` layers (conv1 and conv2) that implement the excitation pathway. Both have `bias=True` because there is no BatchNorm following them (bias is usually omitted before BatchNorm since BN absorbs it, but SE's small FC layers use bias directly).

The `forward` method:
1. Squeeze: `adaptive_avg_pool3d_forward(x, 1)` -- global average pool to `(B, C, 1, 1, 1)`
2. Excite: conv1 → ReLU → conv2 → Sigmoid -- learn channel attention weights
3. Scale: `x * scale` -- multiply the original input by the attention weights

### 6.6 Bottleneck Block (bottleneck.py)

The `BottleneckBlock` is the core computational unit of X3D-M. It implements the three-phase convolution pattern:

**conv_a** (1x1x1 pointwise): expands channels from `in_channels` to `inner_channels`. `bias=False` because BatchNorm follows. This is a standard convolution (`groups=1`).

**conv_b** (3x3x3 depthwise): performs spatial-temporal filtering independently per channel. `groups=inner_channels` makes it depthwise. Stride is `(1, stride, stride)` -- temporal stride is always 1 (preserving all 16 frames), spatial stride is 1 or 2. Padding is 1 on all sides to maintain spatial dimensions (before stride).

**norm_b** is a `Sequential` containing a `BatchNorm3d` followed by either a `SqueezeExcitation` block or an `Identity` no-op, depending on `use_se`.

**conv_c** (1x1x1 pointwise): projects channels from `inner_channels` back to `out_channels`. No activation after this -- that is handled by the ResBlock after the residual addition.

The forward pass:
```
x → conv_a → norm_a → ReLU → conv_b → norm_b(BN + SE/Identity) → SiLU → conv_c → norm_c → output
```

The `Identity` class is a trivial `Module` that returns its input unchanged, used as a placeholder when SE is not applied.

### 6.7 Residual Block (resblock.py)

`ResBlock` wraps a `BottleneckBlock` with a skip connection:

**branch2**: the `BottleneckBlock` (main path).

**branch1** (shortcut): depends on whether dimensions change:
- If `in_channels == out_channels` and `stride == 1`: no branch1 needed; the shortcut is identity.
- If `stride != 1` (spatial downsampling): a 1x1x1 `Conv3d` with the same stride adapts the spatial dimensions.
- If `in_channels != out_channels`: the Conv3d also changes the channel count, and a `BatchNorm3d` is applied after it.

The subtle distinction between `has_branch1` (need any shortcut conv) and `has_branch1_norm` (need BN on shortcut) matches PyTorchVideo's behavior: Stage 2's first block has stride=2 but channels stay at 24, so it needs a shortcut conv (for spatial downsampling) but not a shortcut BN (no channel change).

The forward pass: `output = ReLU(branch2(x) + shortcut(x))`.

### 6.8 Residual Stage (resstage.py)

A `ResStage` is a sequence of `ResBlock` instances. The constructor builds `depth` blocks:
- Block 0: uses `in_channels` as input, applies the stage's stride (typically 2), and has SE.
- Blocks 1, 2, ...: use `out_channels` as input (since block 0's output has `out_channels`), stride=1, SE on even indices.

The forward pass simply iterates through all blocks sequentially.

### 6.9 Stem (stem.py)

The stem processes the raw video input and produces the initial feature maps.

**Conv2plus1dStem**: implements (2+1)D factorized convolution:
- `conv_t` (the spatial convolution, despite the confusing name): a standard 3D convolution with kernel `(1, 3, 3)`, stride `(1, 2, 2)`, padding `(0, 1, 1)`. Maps 3 RGB channels to 24 feature channels. The stride of 2 in H and W halves the spatial resolution from 224 to 112. The temporal kernel size of 1 means each frame is processed independently.
- `conv_xy` (the temporal convolution, despite the confusing name): a depthwise 3D convolution with kernel `(5, 1, 1)`, stride 1, padding `(2, 0, 0)`, groups=24. Each of the 24 channels gets its own independent 5-frame temporal filter. Padding of 2 preserves the temporal dimension size (16 frames in, 16 frames out).

The naming convention (`conv_t` for spatial, `conv_xy` for temporal) comes from PyTorchVideo's original implementation and is kept to match pretrained weight key names.

**Stem**: wraps `Conv2plus1dStem` with `BatchNorm3d(24)` and `ReLU`.

Transform: `(B, 3, 16, 224, 224) → (B, 24, 16, 112, 112)`.

### 6.10 Head (head.py)

The classification head converts spatial feature maps into class predictions.

**ProjectedPool**: a sub-module that performs channel expansion and spatial collapse:
1. `pre_conv`: 1x1x1 Conv3d expanding 192 → 432 channels
2. `pre_norm`: BatchNorm3d(432) + ReLU
3. `AvgPool3d(kernel_size=(16, 7, 7))`: collapses the entire `(16, 7, 7)` spatial-temporal volume to `(1, 1, 1)`, effectively computing the global average for each of the 432 channels
4. `post_conv`: 1x1x1 Conv3d expanding 432 → 2048 channels + ReLU

**Head**: the top-level classification module:
1. `pool` (ProjectedPool): `(B, 192, 16, 7, 7) → (B, 2048, 1, 1, 1)`
2. `dropout_forward(x, 0.5, training)`: during training, randomly zeros 50% of the 2048-dimensional feature vector
3. `x.reshape(x.shape[0], -1)`: flattens from `(B, 2048, 1, 1, 1)` to `(B, 2048)`
4. `linear_forward(x, proj_weight, proj_bias)`: maps 2048 → 400 (class logits)

The linear layer's parameters (`proj_weight` and `proj_bias`) are stored directly in the Head's `_parameters` dictionary rather than in a separate Linear module. This is a design choice to match the weight key names in the pretrained checkpoint.

---

## 7. Full Model Assembly (scratch/models/x3d_m.py)

The `X3D_M` class ties everything together:

```python
self.blocks = ModuleList([
    Stem(),                                                                    # blocks[0]
    ResStage(depth=3,  in_channels=24,  inner_channels=54,  out_channels=24),  # blocks[1]
    ResStage(depth=5,  in_channels=24,  inner_channels=108, out_channels=48),  # blocks[2]
    ResStage(depth=11, in_channels=48,  inner_channels=216, out_channels=96),  # blocks[3]
    ResStage(depth=7,  in_channels=96,  inner_channels=432, out_channels=192), # blocks[4]
    Head(num_classes=400),                                                      # blocks[5]
])
```

The flat `ModuleList` with indices 0-5 matches PyTorchVideo's `blocks` attribute, which is critical for weight loading -- the pretrained weights use keys like `blocks.0.conv.conv_t.weight`, `blocks.1.res_blocks.0.branch2.conv_a.weight`, etc.

The forward pass is simply:
```python
for block in self.blocks:
    x = block.forward(x)
return x
```

---

## 8. Weight Loading and Conversion

### 8.1 Weight Conversion Script

`scripts/convert_pytorch_weights_to_numpy.py` runs on a machine with PyTorch (e.g., a laptop or workstation) and performs:

1. **Load pretrained weights**: either from the PyTorchVideo model hub (downloading the official Facebook-trained weights for Kinetics-400) or from a local `.pth` checkpoint file.

2. **Convert to NumPy**: each PyTorch tensor is detached from the computation graph, moved to CPU, cast to float32, and converted to a NumPy array.

3. **Rewrite key names**: some PyTorch key names differ from the scratch module hierarchy:
   - `blocks.5.proj.weight` → `blocks.5.proj_weight` (Head stores projection in `_parameters` directly)
   - `blocks.5.proj.bias` → `blocks.5.proj_bias`
   - `*.norm_b.1.block.0.*` → `*.norm_b.1.conv1.*` (SE first conv, different naming)
   - `*.norm_b.1.block.2.*` → `*.norm_b.1.conv2.*` (SE second conv)

4. **Save as .npz**: `np.savez_compressed` produces a single compressed archive of all weight arrays.

This script needs to run only once. The resulting `.npz` file can then be copied to the PolarFire SoC.

### 8.2 Weight Loading on the SoC

`scratch/load_weights.py` provides two functions:

**`load_pretrained_numpy(model, path_or_archive)`**: the main loading function. It:
1. Loads the `.npz` file into a dictionary of NumPy arrays (or accepts a pre-loaded dictionary).
2. For each key in the file, splits it into path parts (e.g., `blocks.0.conv.conv_t.weight` → `["blocks", "0", "conv", "conv_t"]` + parameter name `"weight"`).
3. Traverses the module hierarchy to find the target module.
4. Verifies that the shape matches.
5. Copies the array into the module's `_parameters`.
6. Returns lists of missing and unexpected keys for diagnostic purposes.

**`load_pretrained_numpy_if_available(model, path)`**: a convenience wrapper that silently skips if the file doesn't exist (useful for development when weights may not be present).

---

## 9. Profiling and Statistics (stats.py)

The profiling system measures per-layer execution time, parameter counts, and FLOPs estimates. It consists of:

**`get_platform_info()`**: detects the current platform by inspecting `platform.node()`, `platform.machine()`, and `/proc/cpuinfo`. It specifically identifies the PolarFire SoC Icicle Kit by looking for "polarfire", "icicle", or "mpfs" in the hostname, or "sifive" / "riscv" in the CPU info.

**`LayerStats`**: a dataclass holding statistics for a single layer (name, type, shapes, latency, parameter count, FLOPs).

**`RunStats`**: a dataclass for a complete inference run (platform info, total stats, list of LayerStats, section breakdowns).

**`StatsCollector`**: the main profiling class that provides:
- `start_run()` / `end_run()`: bracket an inference run
- `start_section()` / `end_section()`: bracket model sections (Stem, Stage2, etc.)
- `add_layer()`: record a single layer's statistics
- `time_layer()`: a context manager that automatically times the enclosed code block
- `save()`: export to JSON
- `save_text_report()`: export a human-readable text report

**`estimate_conv3d_flops()`**: estimates the number of floating-point operations for a 3D convolution as `2 * output_elements * kernel_elements * channels_per_group` (factor of 2 because each multiply-accumulate counts as 2 FLOPs).

**`estimate_linear_flops()`**: estimates FLOPs for a linear layer as `2 * batch * in_features * out_features`.

---

## 10. Visualization (visualize_stats.py)

This script loads profiling results from JSON files and generates analysis artifacts:

**Text outputs**: comparison tables across platforms, section-by-section breakdowns, bottleneck analysis (top-N slowest layers), and layer-by-layer comparisons.

**Charts** (using matplotlib, if available):
- Total latency bar chart across platforms
- Stacked section breakdown chart
- Per-platform pie charts showing latency distribution
- Horizontal bar charts of top bottleneck layers
- Latency-by-layer-type charts
- Speedup comparison charts (relative to a baseline platform)

**HTML report**: a standalone HTML page with embedded tables comparing platforms and sections.

Usage: `python visualize_stats.py --dir run_stats --output charts --format png`.

---

## 11. Inference Entry Point (main.py)

`main.py` provides the primary interface for running X3D-M inference:

**`build_x3d_m(num_classes, weights_path)`**: constructs the model and optionally loads pretrained weights.

**`run_forward(model, x)`**: runs a simple forward pass without profiling.

**`run_forward_profiled(model, x, collector)`**: runs the forward pass with detailed per-layer profiling. This function manually invokes each layer (rather than calling the model's `forward` directly) so that it can time each operation individually with the `StatsCollector.time_layer()` context manager. It profiles:
- Each convolution in the stem
- Each operation within every bottleneck block of every residual block in every stage
- The SE block's individual operations (squeeze, excite, scale)
- All head operations (pre_conv, pool, post_conv, dropout, linear)

**Command-line interface**:
- `python main.py`: basic inference, stem + full forward pass, reports shapes and latency
- `python main.py --profile`: full profiled inference, saves JSON and text reports
- `python main.py --profile --stem-only`: profile only the stem (quick test)
- `python main.py --profile --notes "Testing on MacBook Pro"`: add notes to the report

---

## 12. PyTorch Reference Implementation (x3d_layers.py)

This file contains the PyTorch-based implementation of the exact same X3D-M architecture. It serves two purposes:

1. **Verification**: by running both the PyTorch and scratch implementations with the same weights and input, the outputs can be compared to verify correctness (they should match within floating-point tolerance of ~1e-4).

2. **Documentation**: the file contains extensive inline comments explaining every architectural decision, every parameter choice, and every PyTorch API used. It serves as a heavily-annotated reference for understanding the architecture.

The PyTorch implementation uses standard PyTorch modules (`nn.Conv3d`, `nn.BatchNorm3d`, `nn.ReLU`, `nn.SiLU`, `nn.Linear`, `nn.Dropout`, `nn.Sequential`, `nn.ModuleList`, `nn.Identity`, `nn.AdaptiveAvgPool3d`, `nn.AvgPool3d`) and loads pretrained weights from `torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)`.

When run as a script, it builds both the custom and official models, loads pretrained weights, runs a forward pass on identical random input, and compares outputs. It also prints a layer-by-layer shape trace and top-5 predictions.

---

## 13. PolarFire SoC Icicle Kit: Hardware Background

### 13.1 Architecture Overview

The Microchip PolarFire SoC is a heterogeneous System-on-Chip that combines a RISC-V processor subsystem with an FPGA (Field-Programmable Gate Array) fabric on a single die. The Icicle Kit is Microchip's official development board for this SoC. This combination is what makes it interesting for edge AI applications: the RISC-V cores handle general-purpose computation (control flow, data management, operating system) while the FPGA fabric can be configured to accelerate specific compute-intensive operations like convolution.

The SoC is based on the MPFS250T device, which integrates:
- A RISC-V multi-core processor complex
- Up to 254K logic elements of PolarFire FPGA fabric
- 784 Math blocks (multiply-accumulate units on the FPGA)
- Interfaces: PCIe Gen2, DDR4, Gigabit Ethernet, USB, SPI, I2C, UART
- Low-power operation suitable for edge deployment

### 13.2 The RISC-V Processor Cores

RISC-V is an open-source instruction set architecture (ISA) that has been gaining significant traction in embedded and edge computing. Unlike proprietary ISAs like x86 (Intel/AMD) or ARM, RISC-V's specification is freely available and can be implemented by anyone without licensing fees.

The PolarFire SoC contains five RISC-V cores:

**1x E51 Monitor Core**: a small, single-issue, in-order core from SiFive's E5 series. It runs the Hart Software Services (HSS) -- the boot firmware responsible for initializing the system, configuring the memory controller, and bootstrapping the application cores. In a typical deployment, this core acts as a system monitor and does not run application workloads. It implements the RV64IMAC ISA (64-bit integer, multiply, atomic, compressed instructions).

**4x U54 Application Cores**: these are 64-bit, single-issue, in-order application cores from SiFive's U5 series. They implement the RV64GC ISA (general-purpose with compressed instructions), which includes integer arithmetic, multiplication/division, atomic operations, and single/double-precision floating-point. Each core has:
- 32KB L1 instruction cache
- 32KB L1 data cache
- Access to a shared 2MB L2 cache
- Memory management unit (MMU) capable of running Linux

These four cores are capable of running a full Linux operating system (typically Yocto-based or Buildroot-based distributions), Python, NumPy, and the scratch library.

At the time of writing, the PolarFire SoC's RISC-V cores do not implement the RISC-V Vector Extension (RVV), which would provide SIMD (Single Instruction, Multiple Data) capabilities similar to ARM NEON or x86 SSE/AVX. This means that data-level parallelism must be achieved through multi-threading (task-level parallelism across the 4 cores) or FPGA offloading rather than through vector instructions.

### 13.3 Memory Subsystem

The Icicle Kit provides:
- **2GB LPDDR4 memory** connected via a 32-bit wide interface to the processor complex
- **2MB shared L2 cache** accessible by all five cores, configurable as cache, scratchpad, or a mix
- **Coherent memory system**: the L2 cache maintains coherence across all cores, which simplifies multi-threaded programming

Memory bandwidth is a critical bottleneck for neural network inference. The 3D convolution operations in X3D-M involve large data movement: for Stage 2, the input tensor alone is `24 * 16 * 112 * 112 * 4 bytes ≈ 19.3 MB`, which is roughly 10x the size of the L2 cache. This means most convolution operations are memory-bound, making efficient memory access patterns and cache utilization crucial for performance.

### 13.4 The FPGA Fabric

The PolarFire FPGA fabric provides up to 254,000 logic elements and 784 Math blocks. Each Math block can perform a multiply-accumulate operation in a single clock cycle, making them ideal for the multiply-accumulate-heavy operations of convolution.

The FPGA communicates with the processor complex through:
- AXI (Advanced eXtensible Interface) buses for memory-mapped access
- DMA (Direct Memory Access) controllers for high-bandwidth data transfers between the shared DDR4 memory and the FPGA fabric

The scratch library's `conv3d_forward_fast` is specifically designed to facilitate FPGA offloading: it carves the convolution into independent `(input_volume, kernel)` pairs that can be DMA-transferred to the FPGA, processed by a hardware convolution engine, and DMA-transferred back. The stride application is kept on the CPU side as a simple array slice.

PolarFire FPGAs are also notable for their low power consumption compared to SRAM-based FPGAs (like those from Xilinx/AMD), using flash-based configuration that consumes no static configuration power. This makes them attractive for always-on edge inference applications.

### 13.5 Why This Matters for Neural Network Inference

Running X3D-M on the PolarFire SoC is a challenging edge deployment scenario:

1. **No GPU**: unlike workstations or even Jetson-class edge devices, there is no GPU for parallel computation.
2. **No vector ISA**: without RISC-V Vector Extension, there is no SIMD to accelerate element-wise operations.
3. **Moderate CPU power**: the U54 cores, while capable of running Linux, are in-order single-issue cores clocked at ~600 MHz, far slower than desktop processors.
4. **Limited memory bandwidth**: the 32-bit LPDDR4 interface provides less bandwidth than modern desktop DDR5 or even mobile LPDDR5.

These constraints make optimization essential. The two primary acceleration strategies are: (a) multi-threading to utilize all 4 U54 cores, and (b) FPGA offloading for compute-intensive convolution operations. The scratch library is designed to support both approaches.

---

## 14. Multi-Threading Acceleration Opportunities

### 14.1 Threading Context on the PolarFire SoC

The PolarFire SoC provides 4 application-class U54 cores. With a coherent shared memory system and Linux running on all four cores, standard POSIX threading (pthreads) is fully supported. Python's `threading` module, `concurrent.futures.ThreadPoolExecutor`, and `multiprocessing` module all work on this platform.

The scratch library's convolution operations use **adaptive multi-threaded parallelism** via a persistent `concurrent.futures.ThreadPoolExecutor` with `NUM_THREADS = 4` workers. The implementation lives in `scratch/ops/conv3d.py` and automatically selects the optimal threading strategy (output-channel parallelism or temporal parallelism) based on the convolution type. Batch normalization, activation, and pooling operations remain single-threaded, as they are memory-bandwidth-bound and benefit less from threading on the PolarFire SoC's shared memory bus (see Section 14.7).

### 14.2 Strategy 1: Output-Channel Parallelism in conv3d_forward_fast

> **Status: IMPLEMENTED** -- see `_conv3d_oc_parallel()` in `scratch/ops/conv3d.py`.

**Where**: the outer loop in `conv3d_forward_fast`.

**Why it works**: each iteration of the `for b in range(B): for oc in range(out_c):` loop computes one independent slice `out[b, oc]` of the output tensor. There are zero data dependencies between iterations -- different output channels read from the same input (read-only) and write to non-overlapping output memory. This is a textbook case for task-parallel decomposition.

**Implementation**: `_conv3d_oc_parallel` enumerates all `(b, oc)` pairs, divides them into `NUM_THREADS` contiguous chunks, and submits each chunk to the module-level `_thread_pool`. Each thread iterates over its chunk, calling `conv3d_core` per pair and writing into non-overlapping output slices (no locks needed). Chunking reduces task-submission overhead compared to one future per `(b, oc)` pair -- for 432 output channels this means 4 pool submissions instead of 432.

```python
def _conv3d_oc_parallel(x_pad, weight, bias, out,
                        B, out_c, groups, c_per_group,
                        st, sh, sw):
    tasks = [(b, oc) for b in range(B) for oc in range(out_c)]
    n_tasks = len(tasks)
    chunk_size = max(1, (n_tasks + NUM_THREADS - 1) // NUM_THREADS)

    def _process_chunk(start, end):
        for idx in range(start, min(end, n_tasks)):
            b, oc = tasks[idx]
            g = oc % groups
            c_start = g * c_per_group
            inp_volume = x_pad[b, c_start:c_start + c_per_group]
            dense_out = conv3d_core(inp_volume, weight[oc])
            out[b, oc] = dense_out[::st, ::sh, ::sw].astype(x_pad.dtype)
            if bias is not None:
                out[b, oc] += bias[oc]

    futures = [_thread_pool.submit(_process_chunk, i * chunk_size,
               i * chunk_size + chunk_size)
               for i in range(NUM_THREADS) if i * chunk_size < n_tasks]
    for f in futures:
        f.result()
```

**Why Python threads work here**: normally, Python's Global Interpreter Lock (GIL) prevents true parallel execution of Python threads. However, NumPy and OpenCV operations release the GIL while executing their C/C++ internals. Since `conv3d_core` spends nearly all its time in `cv2.filter2D` and NumPy operations, threads achieve genuine parallelism.

**Expected benefit**: for pointwise convolutions (conv_a, conv_c) and standard convolutions (stem conv_t) where `out_c` ranges from 24 to 432, there are many independent iterations to distribute across 4 cores, leading to near-linear scaling.

### 14.3 Strategy 2: Temporal Parallelism in conv3d_core

> **Status: IMPLEMENTED** -- see `_conv3d_core_threaded()` and `_conv3d_temporal_parallel()` in `scratch/ops/conv3d.py`.

**Where**: inside `_conv3d_core_threaded`, a multi-threaded variant of `conv3d_core`.

**Why it works**: the temporal output positions `tt = 0, 1, ..., T_out-1` write to independent slices `out_volume[tt]`. Each temporal position accumulates contributions from `kT` input frames, but different temporal positions write to different memory locations.

**Implementation**: `_conv3d_core_threaded` partitions `T_out` temporal positions into `NUM_THREADS` contiguous ranges using integer boundaries (`boundaries[i] = i * T_out // NUM_THREADS`). Each thread computes its slice into a private `local_out` buffer, performing the full `C * kT` inner loop of `cv2.filter2D` calls. After all threads complete, the local buffers are assembled into the final output volume. If `T_out < NUM_THREADS`, the function falls back to sequential `conv3d_core` to avoid thread-overhead on trivially small workloads. The outer loop over `(b, oc)` pairs is handled by `_conv3d_temporal_parallel`, which calls `_conv3d_core_threaded` for each depthwise channel.

```python
def _conv3d_core_threaded(volume, kernel):
    # ... setup and T_out < NUM_THREADS fallback ...
    boundaries = [i * T_out // NUM_THREADS for i in range(NUM_THREADS + 1)]

    def _compute_chunk(t_start, t_end):
        local_out = np.zeros((t_end - t_start, H_out, W_out), dtype=np.float32)
        for c in range(C):
            for i, tt in enumerate(range(t_start, t_end)):
                for dt in range(kT):
                    k_2d = kernel[c, dt]
                    if not np.any(k_2d):
                        continue
                    filtered = cv2.filter2D(volume[c, tt + dt], ...)
                    local_out[i] += filtered[:H_out, :W_out]
        return t_start, local_out

    futures = [_thread_pool.submit(_compute_chunk, boundaries[i], boundaries[i+1])
               for i in range(NUM_THREADS) if boundaries[i] < boundaries[i+1]]
    for f in futures:
        t_start, local_out = f.result()
        out_volume[t_start:t_start + local_out.shape[0]] = local_out
```

**When this is most useful**: for the 3x3x3 depthwise convolutions (`conv_b`), where `C=1` per group and `T_out=16`. This gives 4 temporal positions per core -- a clean split. For the 1x1x1 pointwise convolutions, `T_out * H_out * W_out` positions are computed, but each `conv3d_core` call does trivially little work, making the thread overhead not worth it (hence they use Strategy 1 instead).

### 14.4 Strategy 3: Adaptive Hybrid Parallelism

> **Status: IMPLEMENTED** -- the dispatch logic lives in `conv3d_forward_fast()` in `scratch/ops/conv3d.py`.

The optimal threading strategy depends on the specific convolution:

- **Pointwise 1x1x1 convolutions** (conv_a, conv_c, head convolutions): the kernel has size 1x1x1, so `conv3d_core` does very little per call. Parallelizing the outer `(b, oc)` loop (Strategy 1) is more effective because there are many output channels and each call is lightweight.

- **Depthwise 3x3x3 convolutions** (conv_b): each output channel processes only one input channel (groups=inner_channels), so `conv3d_core` does moderate work per call. Both strategies can work, but Strategy 2 (temporal parallelism within `conv3d_core`) is attractive because it parallelizes a single, larger task rather than many tiny tasks.

- **Standard convolutions** (stem conv_t): neither pointwise nor depthwise; Strategy 1 is used since there are multiple output channels to distribute.

The adaptive dispatch in `conv3d_forward_fast` checks the kernel size and group count to select the parallelism axis:

```python
is_pointwise = kT * kH * kW == 1
is_depthwise = groups == out_c and groups > 1

if is_pointwise or not is_depthwise:
    _conv3d_oc_parallel(...)   # Strategy 1
else:
    _conv3d_temporal_parallel(...)  # Strategy 2
```

A persistent module-level `ThreadPoolExecutor(max_workers=NUM_THREADS)` is reused across all convolution calls, amortizing thread-creation cost over the entire inference pass.

### 14.5 Additional Acceleration Techniques

Beyond multi-threading in the convolution layers, several other techniques can improve performance on the PolarFire SoC:

**Batch Normalization Fusion**: during inference, BatchNorm is a linear operation: `y = gamma * (x - mean) / sqrt(var + eps) + beta`. This can be algebraically fused into the preceding convolution's weights and bias:
```
fused_weight = gamma / sqrt(var + eps) * original_weight
fused_bias = gamma * (original_bias - mean) / sqrt(var + eps) + beta
```
This eliminates entire BatchNorm layers from the forward pass, saving memory bandwidth and computation. Every ResBlock has three BatchNorms, so this optimization removes 78 BatchNorm operations (26 blocks * 3) from the network.

**im2col + GEMM for the slow path**: on RISC-V platforms without OpenCV, the 6-deep nested loop in `conv3d_forward_slow` is extremely slow. An alternative approach is im2col (image to column): unroll each input patch into a column of a matrix, forming a large matrix, and then perform the convolution as a single matrix multiplication using `np.dot`. OpenBLAS (which has RISC-V support) internally parallelizes matrix multiplication across available cores, providing multi-threading "for free."

**Reduce memory allocation in `_pad_3d`**: every convolution call currently allocates a new padded tensor. For early-stage tensors (e.g., 24 channels x 16 frames x 112x112 spatial ≈ 19 MB), this is significant memory pressure. Pre-allocating padding buffers or using NumPy stride tricks to create zero-padded views without copying could reduce allocation overhead.

**Fixed-point arithmetic**: the U54 cores have hardware floating-point, but fixed-point (integer) arithmetic can be faster and more energy-efficient. Quantizing the model weights and activations from float32 to int8 would reduce memory bandwidth requirements by 4x and potentially improve throughput. This requires careful calibration to maintain accuracy.

### 14.6 Python Threading and the GIL

Python's Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously. This is often cited as a limitation of Python threading. However, for this project, the GIL is largely not a problem because:

1. **NumPy releases the GIL** during array operations. When a thread calls `np.sum`, `np.zeros`, array slicing, or other NumPy C functions, the GIL is released, allowing other threads to execute simultaneously.

2. **OpenCV releases the GIL** during `cv2.filter2D` and other compute-intensive operations. Since `conv3d_core` spends the vast majority of its time in `cv2.filter2D`, threads achieve genuine parallelism.

3. **`multiprocessing.Pool`** as a GIL-free alternative: if GIL contention is observed, Python's multiprocessing module spawns separate processes (each with its own GIL). The overhead is higher (data must be serialized/deserialized between processes), but there is zero GIL contention. Libraries like `joblib` with the `loky` backend handle NumPy array sharing efficiently in this scenario.

### 14.7 Estimated Impact

The theoretical maximum speedup from multi-threading across 4 U54 cores is 4x. In practice, due to thread management overhead, memory bandwidth contention, and Amdahl's law (the non-parallelizable fraction limits overall speedup), realistic estimates are:

- **Convolution operations** (which account for 80-90% of total inference time): **2.5x - 3.5x speedup** with output-channel parallelism.
- **BatchNorm, activations, pooling**: these are memory-bandwidth-bound operations where multi-threading provides limited benefit due to shared memory bus contention. Estimated **1.2x - 1.5x speedup**.
- **Overall inference**: estimated **2x - 3x speedup** depending on the specific layer mix and memory access patterns.

The convolution layers in Stage 4 (11 blocks with 216 inner channels) and Stage 5 (7 blocks with 432 inner channels) contain the most parallelizable work and would benefit the most.

---

## 15. C Backend: Native Convolution via ctypes

### 15.1 Motivation and Architecture

While the Python+OpenCV convolution implementations (`"fast"` and `"threaded"`) offer good performance on most platforms, they still incur Python interpreter overhead for loop control and array bookkeeping. On the PolarFire SoC's 600 MHz U54 cores, this overhead is proportionally more significant than on a desktop processor.

The `"native"` method eliminates Python overhead entirely for the convolution kernel by implementing the full 3D convolution in C, compiled as a shared library (`libconv3d.so`) and called from Python via ctypes. The C implementation targets the specific hardware characteristics of the PolarFire SoC:

- **RV64GC ISA** — no vector extension (RVV), so no SIMD intrinsics; performance comes from scalar optimisations and compiler auto-optimisation with `-O3 -funroll-loops -ffast-math` 
- **4× U54 cores** — pthreads with 4 worker threads, partitioning `(batch × output_channel)` work items.
- **32 KiB L1 data cache per core** — spatial tiling with `TILE_H=8, TILE_W=16` keeps the input receptive field of each tile inside L1.

### 15.2 Building the Shared Library

The C source lives in `scratch/ops/conv3d_c/`. To build:

```bash
# Auto-detect architecture (uses -march=rv64gc on RISC-V, -march=native on x86_64)
make -C scratch/ops/conv3d_c

# Explicit RISC-V target (cross-compile or native on PolarFire SoC)
make -C scratch/ops/conv3d_c riscv
# Compiler flags: -O3 -march=rv64gc -funroll-loops -ffast-math -shared -fPIC -lpthread

# Explicit x86 target for testing on a dev machine
make -C scratch/ops/conv3d_c native
# Compiler flags: -O3 -march=native -funroll-loops -ffast-math -shared -fPIC -lpthread

# Clean build artefacts
make -C scratch/ops/conv3d_c clean
```

The output is `scratch/ops/conv3d_c/libconv3d.so`, which is loaded automatically at Python import time if present.

### 15.3 C Implementation Details

The C function `conv3d_forward_c()` in `conv3d.c` takes contiguous float32 arrays for input, weight, bias (nullable), and a pre-allocated output buffer, plus all shape and convolution parameters. It handles zero-padding internally using an inline bounds-checking helper with an unsigned-cast trick that folds the `< 0` and `>= dim` checks into a single comparison:

```c
static inline float padval(const float *x,
                           int b, int c, int t, int h, int w,
                           int C, int T, int H, int W,
                           int pt, int ph, int pw) {
    int ti = t - pt, hi = h - ph, wi = w - pw;
    if ((unsigned)ti >= (unsigned)T ||
        (unsigned)hi >= (unsigned)H ||
        (unsigned)wi >= (unsigned)W)
        return 0.f;
    return x[idx5(b, c, ti, hi, wi, C, T, H, W)];
}
```

**Thread dispatch:** The entry point creates `min(4, B×C_out)` pthreads, each processing a contiguous range of `(batch, output_channel)` pairs. Since different pairs write to non-overlapping output slices, no synchronisation is required beyond `pthread_join`.

**Three internal fast paths** are selected at runtime per-thread based on kernel size and groups:

1. **Pointwise (1×1×1):** No spatial kernel overlap. The hot loop is a channel accumulation over `c_per_group` values per output position. Early-exit rows where temporal or height falls entirely in the padding zone. `#pragma GCC unroll 8` on the channel loop.

2. **Depthwise (groups == C_out, c_per_group == 1):** Each output channel reads from exactly one input channel. Small kernels (typically 3×3×3 or 5×1×1). Spatial tiling with `TILE_H=8, TILE_W=16` keeps the input receptive field in L1 cache. The inner kernel loops use `#pragma GCC unroll` with hints matching the common kernel dimensions (3 and 5).

3. **General (everything else):** Handles the stem's (1,3,3) standard convolution and any future kernel shapes. Spatial tiling plus channel-first accumulation. The loop structure is:
   ```
   for each (b, oc) in thread's range:
     for each temporal output to:
       for each spatial tile:
         for each (ho, wo) in tile:
           accumulate over (c, dt, dh, dw)
   ```

**Cache analysis for TILE_H=8, TILE_W=16 on the U54 (32 KiB L1):**

| Conv type | c_per_group | Kernel | Input tile | Kernel data | Total |
|-----------|-------------|--------|------------|-------------|-------|
| Depthwise 3×3×3 | 1 | 27 × 4B | 10 × 18 × 4B ≈ 0.7 KiB/t-pos | 108 B | ~2.5 KiB |
| Stem 1×3×3 | 3 | 9 × 4B | 3 × 10 × 18 × 4B ≈ 2.1 KiB/t-pos | 108 B | ~6.5 KiB |
| Pointwise (no tiling) | 24+ | 1 × 4B | — (sequential) | 96 B | < 1 KiB |

All cases fit comfortably within 32 KiB.

### 15.4 Python ctypes Wrapper

At import time, `scratch/ops/conv3d.py` calls `_load_c_backend()`, which searches for `libconv3d.so` (or `.dylib` on macOS) in the `conv3d_c/` subdirectory. If found, it registers the C function signature with ctypes and prints a one-time confirmation message. If not found, a warning is printed and the `"native"` method will raise `RuntimeError` when called.

The wrapper function `conv3d_forward_native()`:

1. Ensures all input arrays are contiguous float32 via `np.ascontiguousarray`.
2. Computes output dimensions and pre-allocates the output array.
3. Marshals numpy arrays to C pointers via `.ctypes.data_as(ctypes.POINTER(ctypes.c_float))`.
4. Passes `None` (ctypes NULL) for the bias pointer when no bias is present.
5. Calls the C function and returns the output array.

### 15.5 Selecting the Convolution Method

The convolution method can be selected at three levels:

**Global default** (affects all `Conv3d` layers):

```python
from scratch import set_conv3d_method
set_conv3d_method("native")
```

**Per-layer** (overrides the global default for one layer):

```python
from scratch.nn.conv3d import Conv3d
conv = Conv3d(3, 16, kernel_size=3, padding=1, method="native")
```

**CLI** (sets the global default for the inference run):

```bash
python main.py --method native --profile
```

The `is_native_available()` function can be used to check whether the C library loaded successfully before selecting `"native"`.

---

## 16. Int8 Post-Training Quantization (PTQ)

### 16.1 Motivation

The float32 scratch library is sufficient as a correctness reference and as a baseline for CPU benchmarking, but it is not the final deployment target. The end goal of this project is to accelerate X3D-M on the PolarFire SoC's FPGA fabric, where multiply-accumulate (MAC) resources are dominated by integer DSPs and on-chip memory is scarce. Running int8 × int8 → int32 MACs instead of float32 × float32 → float32 gives roughly a 4× reduction in weight memory footprint, a proportional reduction in DRAM bandwidth (which is the primary bottleneck on the Icicle Kit's 32-bit LPDDR4), and allows each FPGA DSP block to perform multiple int8 multiplies in parallel. For these reasons, the project standardizes on int8 post-training quantization (PTQ) as the bridge between the float32 software reference and the future FPGA accelerator.

X3D-M does not ship with an official pre-quantized checkpoint from Facebook or PyTorchVideo. The `scripts/quantize_x3d_ptq.py` script performs this quantization offline on a development machine and exports the result to a flat `.npz` file that can be loaded on the SoC with no PyTorch dependency.

### 16.2 Quantization Scheme

The scheme is deliberately chosen to be the simplest one that (a) preserves accuracy on a pretrained model and (b) maps cleanly onto an FPGA MAC array without requiring zero-point subtraction logic in the hot path.

- **Weights**: symmetric, per-output-channel, int8. The zero-point is fixed at zero, so the hardware never has to subtract a bias term from the weight stream. Per-channel scales are applied only at requantization time (one scale per output channel), which is effectively free in hardware.
- **Activations**: symmetric, per-tensor, int8. A single scale per activation tensor keeps the activation streaming path uniform. Per-tensor (rather than per-channel) activation quantization is standard because activations are data-dependent and cannot be recalibrated per channel without significantly complicating the runtime.
- **Accumulator**: int32. Each convolution produces an int32 accumulator that is then requantized back to int8 using `(acc * input_scale * weight_scale[c]) / output_scale`.
- **Bias**: int32, with `bias_scale[c] = input_scale * weight_scale[c]`. This allows biases to be added directly into the int32 accumulator with no additional rescaling, which is the canonical pattern used by every production int8 inference engine (TFLite, QNNPACK, CMSIS-NN).

Why symmetric? Asymmetric quantization (non-zero zero-point) gives slightly better accuracy on activations with skewed distributions, but it requires every MAC to subtract a zero-point offset from either the weight or activation operand, which doubles the DSP usage on the FPGA. Symmetric quantization loses less than one percentage point of top-1 accuracy on X3D-M in our measurements and is worth the tradeoff.

### 16.3 BatchNorm Folding

Before quantization, every `BatchNorm3d` in the model is folded into the preceding `Conv3d`. This is essential for two reasons. First, quantizing BN separately would require an extra pair of scales and a separate int8 affine operation, none of which the FPGA accelerator needs to support. Second, folding improves quantization accuracy: after folding, the conv weights absorb the BN `gamma / sqrt(var + eps)` scaling, which smooths out the per-channel weight distribution and makes per-channel weight quantization more effective.

The folding math is:

```
W'[c] = W[c] * gamma[c] / sqrt(var[c] + eps)
b'[c] = (b[c] - mean[c]) * gamma[c] / sqrt(var[c] + eps) + beta[c]
```

After folding, each BatchNorm3d is replaced with `nn.Identity()` and has no runtime cost. The `fold_all_bn()` function in `quantize_x3d_ptq.py` performs this pass automatically on the entire model graph.

### 16.4 Calibration

PTQ requires a small set of representative inputs to collect activation statistics (specifically, the absolute maximum of each activation tensor). Forward hooks attached to every `Conv3d` and `Linear` layer observe both the input and output tensors during calibration and maintain a running maximum.

The script defaults to 128 calibration batches of random normal tensors, which is adequate to capture rough activation magnitudes on a pretrained model and completes in about one to two minutes on the M3 Max (MPS backend is auto-detected). For a more accurate calibration, pass `--calib-dir` pointing to a directory of preprocessed Kinetics clips saved as `.npy` files shaped `(3, 16, 224, 224)`. In practice, 64 to 256 real calibration clips are sufficient to reach the accuracy floor of PTQ.

The scale for each observed tensor is computed as `abs_max / 127`, which produces a symmetric int8 range of `[-127, 127]`. The value `-128` is deliberately not used; this keeps the quantized range symmetric around zero and eliminates a corner case that complicates requantization in fixed-point hardware.

### 16.5 Output Format

The script exports a single flat `.npz` file (default `weights/x3d_m_int8.npz`). For every quantized layer `L`, the file contains:

- `L.weight_q` — int8, shape `(out_c, in_c/groups, kT, kH, kW)` for Conv3d or `(out, in)` for Linear.
- `L.weight_scale` — float32, shape `(out_c,)`.
- `L.bias_q` — int32, shape `(out_c,)`, only present if the layer has a bias.
- `L.input_scale` — float32 scalar, the symmetric per-tensor scale of the layer's input activation.
- `L.output_scale` — float32 scalar, the symmetric per-tensor scale of the layer's output activation.

Global metadata is stored under the `__meta__` prefix (`num_quantized_layers`, `weight_scheme`, `act_scheme`, `bias_scheme`) so that the SoC-side loader can sanity-check the file and refuse to load weights exported with an incompatible scheme.

### 16.6 Running the Script

The script must be run on a machine with PyTorch installed. On the development MacBook (M3 Max, 48 GB RAM), a typical invocation looks like:

```bash
# Quantize directly from the PyTorchVideo hub checkpoint
python scripts/quantize_x3d_ptq.py -o weights/x3d_m_int8.npz

# Use a local float32 checkpoint and 256 random calibration batches
python scripts/quantize_x3d_ptq.py \
    -i weights/x3d_m_kinetics400.pth \
    -o weights/x3d_m_int8.npz \
    --num-calib-batches 256

# Use real Kinetics calibration clips for best accuracy
python scripts/quantize_x3d_ptq.py \
    --calib-dir data/kinetics_calib \
    -o weights/x3d_m_int8.npz
```

MPS (Apple Silicon GPU) is auto-detected and used when available. The full 128-batch calibration takes roughly one to two minutes on the M3 Max, and the resulting `.npz` is approximately one quarter the size of the float32 version.

### 16.7 Known Caveats and Future Work

- **SiLU activations**: SiLU (Swish) is non-monotonic near zero and is harder to represent exactly with int8 than ReLU. The current export assumes the SoC runtime will dequantize before SiLU, apply SiLU in float, and requantize the result. A future refinement is to precompute a 256-entry int8 lookup table indexed by the int8 activation value, which replaces the dequantize/SiLU/requantize sequence with a single table load on the FPGA.
- **Depthwise 3×3×3 convolutions**: these are the most quantization-sensitive layers in X3D-M because each output channel has only 27 weights and any quantization error is not averaged across many input channels. If PTQ accuracy drops by more than about two points on Kinetics-400, the recommended next step is quantization-aware training (QAT) for a small number of epochs, focused on the depthwise layers.
- **Squeeze-Excitation**: SE blocks contain a sigmoid and two small FC layers. They are quantized uniformly with the rest of the model, but their dynamic range is much smaller than the main convolutional path, and per-tensor activation quantization loses some precision. This has not been a problem in practice but is worth monitoring during accuracy validation.
- **No runtime int8 kernel yet**: the `.npz` file produced by this script is consumed by the future int8 convolution kernel (to be implemented in both C and FPGA). Until those kernels exist, the scratch library continues to run in float32 at inference time.

---

## 17. Archive and Legacy Code

The `archive/` directory contains earlier implementations that have been superseded by the scratch library:

- **`main.cpp` / `conv3d.cpp`**: C++ implementations using LibTorch (PyTorch's C++ frontend) and FFmpeg for video decoding. These required LibTorch to be compiled for RISC-V, which proved impractical.
- **`model.py`**: earlier Python model definitions.
- **`train.py`**: training script for fine-tuning (not applicable to inference-only deployment).
- **`infer_x3d_m.py`**: earlier inference script using PyTorch directly.
- **`live_x3d_webcam.py`**: webcam-based live inference demo.
- **`export_x3d_m_torchscript.py`**: TorchScript export for deployment (superseded by the NumPy approach).
- **`sw_3d_conv.py`** / **`conv.py`**: earlier standalone convolution implementations.

These files are retained for historical reference but are not used by the current system.

---

## 19. Int8 Quantized Runtime (scratch/quantized/)

The `scratch/quantized/` subpackage is a completely self-contained int8 inference runtime for X3D-M. It is the bridge between the float32 software reference and the future FPGA accelerator. Nothing in this subpackage is imported by the float32 path -- you can run both `scratch.models.x3d_m.X3D_M` (float32) and `scratch.quantized.model.QuantizedX3D_M` (int8) in the same Python process on separate model instances without interference.

### 19.1 Why a Separate Quantized Runtime?

The float32 `scratch` library is a faithful NumPy reimplementation of PyTorch's X3D-M, designed for correctness verification and CPU-side inference. But the PolarFire SoC's FPGA fabric operates on integer arithmetic -- specifically int8 × int8 multiplications accumulated into int32. To deploy convolutions on the FPGA, we need a parallel runtime where:

1. Weights are stored as int8 (4× smaller than float32, reducing DRAM bandwidth demands on the 32-bit LPDDR4 bus).
2. Activations are quantized from float32 to int8 at layer boundaries.
3. The convolution itself runs entirely in integer arithmetic.
4. The output is requantized back to int8 (or dequantized to float32 for the next float operation).

Rather than modifying the float32 library and breaking its role as a correctness reference, the int8 runtime lives in its own subpackage and reuses the float32 model's *structure* (same module tree, same forward pass logic) while swapping out the computational layers.

### 19.2 Architecture Overview

The quantized runtime is composed of four files:

```
scratch/quantized/
├── __init__.py           # Public API: exports all key classes and functions
├── conv3d_int8.py        # Software reference int8 3D convolution kernel
├── layers.py             # QuantizedConv3d and QuantizedLinear module classes
├── load_int8_weights.py  # Loader for int8 .npz weight files
└── model.py              # QuantizedX3D_M builder (swap Conv3d→QuantizedConv3d)
```

The dependency flow is: `model.py` → `layers.py` → `conv3d_int8.py`, with `load_int8_weights.py` operating on the constructed model.

### 19.3 The Software Reference Int8 Convolution Kernel (conv3d_int8.py)

This file implements, in pure NumPy, the exact sequence of operations that the FPGA accelerator will perform for one quantized Conv3d layer. It is not designed for speed -- it uses nested loops over spatial positions. Its purpose is to be a bit-accurate model of the hardware so the rest of the pipeline can be developed and validated without waiting for the FPGA bitstream.

**Function signature:**

```python
def conv3d_int8_forward(
    x_q: np.ndarray,       # int8, (B, C_in, T, H, W)
    weight_q: np.ndarray,  # int8, (C_out, C_in//groups, kT, kH, kW)
    bias_q: np.ndarray,    # int32, (C_out,) or None
    input_scale: float,    # per-tensor float32 scale
    weight_scale: np.ndarray,  # per-channel float32 scale, (C_out,)
    output_scale: float,   # per-tensor float32 scale
    stride, padding, groups
) -> np.ndarray:           # int8, (B, C_out, T', H', W')
```

**Step-by-step operation:**

1. **Pad the int8 input** with zeros using `_pad_3d_int8()`. Under symmetric quantization, integer zero corresponds to float 0.0, so padding with zero is semantically correct.

2. **Promote to int32** for accumulation: `x_padded.astype(np.int32)` and `weight_q.astype(np.int32)`. This prevents overflow during the multiply-accumulate loop.

3. **Compute the int32 accumulator** by iterating over every output spatial position `(t, h, w)`, extracting the input patch, and performing a matrix multiply against the flattened weights: `patch_flat @ w_flat.T`. This produces exact int32 results because int8 × int8 fits comfortably in int32 (worst case: 127 × 127 × 27 kernel elements × say 432 channels ≈ 188M, well within int32 range of 2.1B).

4. **Add int32 bias** directly into the accumulator: `acc += bias_q.reshape(1, C_out, 1, 1, 1)`. The bias was pre-scaled during PTQ such that `bias_q[c]` has an implicit scale of `input_scale * weight_scale[c]`, matching the accumulator's implicit scale.

5. **Requantize to int8** using the per-channel multiplier `M[c] = (input_scale * weight_scale[c]) / output_scale`. The accumulator is cast to float32, multiplied by M, rounded to nearest integer, clipped to [-127, 127], and cast to int8. The current implementation uses float32 for this step; the FPGA will use the fixed-point `(M0, n)` form described in Section 21.

### 19.4 QuantizedConv3d Layer (layers.py)

`QuantizedConv3d` is the int8 drop-in replacement for `scratch.nn.conv3d.Conv3d`. It has the same constructor signature (in_channels, out_channels, kernel_size, stride, padding, groups) plus a `backend` parameter.

**Stored parameters** (in `_parameters` dict):

| Parameter | Dtype | Shape | Description |
|-----------|-------|-------|-------------|
| `weight_q` | int8 | `(C_out, C_in//groups, kT, kH, kW)` | Quantized weights |
| `weight_scale` | float32 | `(C_out,)` | Per-channel weight scale |
| `bias_q` | int32 | `(C_out,)` or None | Quantized bias (scaled by `s_in * s_w[c]`) |
| `input_scale` | float32 | scalar | Per-tensor input activation scale |
| `output_scale` | float32 | scalar | Per-tensor output activation scale |

**Forward pass (the three-step quantize-compute-dequantize dance):**

```
x_f32 ──quantize──> x_q ──int8 conv──> y_q ──dequantize──> y_f32
```

1. **`_quantize_input(x_f32)`**: divides by `input_scale`, rounds, clips to [-127, 127], casts to int8.
2. **`_run_backend(x_q)`**: dispatches to either `conv3d_int8_forward` (the reference NumPy kernel) or a future FPGA driver.
3. **`_dequantize_output(y_q)`**: multiplies int8 result by `output_scale` to recover float32.

This boundary quantize/dequantize strategy means the quantized model accepts float32 input and produces float32 output, making it interchangeable with the float32 model from the caller's perspective. Activations that flow between layers (like the SiLU activation between conv_b's BN and conv_c) are processed in float32 -- only the convolution itself runs in int8. This is the "hybrid" approach described in `fpga_flow.md`.

**The `backend` parameter:**
- `"reference"`: uses `conv3d_int8_forward` from `conv3d_int8.py` (pure NumPy, slow but correct).
- `"fpga"`: placeholder that raises `NotImplementedError`. When the real FPGA driver is ready, this branch will DMA the int8 tensors to FPGA memory, trigger the accelerator, and read back the int8 result.

### 19.5 QuantizedLinear Layer (layers.py)

`QuantizedLinear` does for the classification head what `QuantizedConv3d` does for convolution layers. It performs `x_q @ W_q^T` in int32, adds the int32 bias, requantizes with per-channel M, and dequantizes back to float32.

In the current X3D-M model, the Head's final linear projection (2048 → 400) is **not** quantized -- it stays in float32. `QuantizedLinear` exists for completeness and for future work where the linear layer might also be offloaded.

### 19.6 The QuantizedX3D_M Model (model.py)

Rather than re-declaring the entire X3D-M architecture from scratch, `QuantizedX3D_M` inherits from `X3D_M` and rewrites its module tree in-place:

1. **Every `Conv3d`** is replaced with a `QuantizedConv3d` having the same constructor arguments.
2. **Every `BatchNorm3d`** is replaced with an `_IdentityModule` (a no-op). This reflects the fact that BN has already been folded into the preceding Conv3d during PTQ (see Section 16.3).
3. **The Head's Linear** (`proj_weight`, `proj_bias`) stays in float32.

The recursive swap is performed by `_swap_conv_and_bn(parent, backend)`, which walks the module tree and replaces children by type. Both the `_modules` dict entry and the instance attribute (e.g., `self.conv_a`) are updated to keep `forward()` calls working.

**Builder function:**

```python
model = build_quantized_x3d_m(
    num_classes=400,
    weights_path="weights/x3d_m_int8.npz",
    backend="reference",
)
model.eval()
logits = model.forward(x_f32)   # float32 in, float32 out
```

### 19.7 Int8 Weight Loader (load_int8_weights.py)

The int8 weight loader is separate from the float32 loader (`scratch/load_weights.py`) because the float32 loader assumes all parameters are float32 and would corrupt int8 and int32 arrays. The int8 loader:

1. Loads the `.npz` file and iterates over its keys.
2. Splits each key (e.g., `blocks.1.res_blocks.0.branch2.conv_a.weight_q`) into a module path and a parameter suffix.
3. Walks the module tree to find the target `QuantizedConv3d` or `QuantizedLinear`.
4. Enforces the expected dtype for each suffix (int8 for `weight_q`, float32 for `weight_scale`, int32 for `bias_q`, etc.).
5. Performs shape checking and copies the array into the module's `_parameters`.
6. Returns lists of missing and unexpected keys for diagnostics.

Recognized parameter suffixes: `weight_q`, `weight_scale`, `bias_q`, `input_scale`, `output_scale`, `proj_weight`, `proj_bias`.

---

## 20. Int8 Inference Entry Point (main_int8.py)

`main_int8.py` is the int8 counterpart to `main.py`. It provides a self-contained command-line interface for running the quantized model:

```bash
python main_int8.py --weights weights/x3d_m_int8.npz
python main_int8.py --weights weights/x3d_m_int8.npz --backend reference
python main_int8.py --dry-run    # Build model only, verify weight loading
```

**Key differences from main.py:**
- Uses `build_quantized_x3d_m()` instead of `build_x3d_m()`
- Does not include profiling (the reference kernel is intentionally slow)
- Includes a `--dry-run` flag for testing that the model builds and weights load cleanly without running the expensive forward pass
- Prints a warning that the reference kernel is "pure NumPy nested loops and intended for correctness, not speed"
- Supports `--backend` selection (currently only "reference")

---

## 21. FPGA Per-Layer Validation Harness (fpga_tests/)

The `fpga_tests/` directory validates the int8 FPGA-offload path one convolution layer at a time. It implements the "tensor-by-tensor dataflow" discipline from `fpga_flow.md`, where each layer is brought up independently before integrating into the full pipeline.

### 21.1 The Problem It Solves

On the PolarFire SoC, we want to move 3D convolutions off the RISC-V CPU and onto the FPGA fabric. The FPGA only sees int8 data. Three questions must be answered before we can trust the FPGA:

1. **Does the quantization math work?** Given float32 input and weights, can we produce int8 equivalents, run an int8 conv, and get a float32 result close to the original?
2. **Does the FPGA's fixed-point requantizer agree with the float32 reference?** The hardware uses `(M0, n)` shift-and-multiply instead of float multiply.
3. **Does the C backend (FPGA offload library) agree bit-for-bit with the Python simulator?** The C library mirrors the FPGA's exact datapath.

### 21.2 Four Execution Paths

The harness runs each convolution layer through four parallel paths and compares them:

| Path | Where it runs | Arithmetic | Requantize | Purpose |
|------|--------------|------------|------------|---------|
| Float reference | CPU | float32 | none | Accuracy ceiling |
| Software int8 (gold) | CPU | float32→int32 | float M | "Correct" int8 output |
| FPGA simulator | CPU (Python) | float32→int32 | fixed-point (M0, n) | Python stand-in for FPGA |
| **FPGA HW (C backend)** | CPU (C lib) | **int8→int32** | **fixed-point (M0, n)** | **FPGA offload library** |

### 21.3 Quantization Primitives (quant.py)

This file implements symmetric int8 quantization from the ground up:

**Scale computation:**
- `compute_tensor_scale(x)`: `s = max(|x|) / 127` — per-tensor symmetric scale
- `compute_weight_scales(W)`: per-output-channel scale, shape `(C_out,)`

**Quantize/dequantize:**
- `quantize_tensor(x, s)`: `round(x / s)`, clip to [-127, 127], cast to int8
- `quantize_weights(W, s_w)`: per-channel weight quantization
- `dequantize_tensor(q, s)`: `q * s` back to float32

**The clipping range is [-127, +127]** (deliberately dropping -128). This keeps the range symmetric around zero and eliminates a corner case in fixed-point requantization hardware.

**Requantization multiplier:**
- `compute_M(s_in, s_w, s_out)`: `M[c] = (s_in * s_w[c]) / s_out` — per-channel float32

**Fixed-point decomposition:**
- `quantize_multiplier_fixed_point(M)`: decomposes each float32 M[c] into `M0[c] * 2^(-n[c])` where M0 is an int32 in [2^30, 2^31-1]. This is the same decomposition used by TFLite and CMSIS-NN. The CPU computes (M0, n) once at model-build time; the FPGA only ever sees the integer pair.

**Two requantization implementations:**
- `apply_requantize_float(acc32, M)`: software reference, float32 multiply. `round(acc32 * M[c])`, clip, cast to int8.
- `apply_requantize_fixed_point(acc32, M0, n)`: FPGA-style, int64 multiply + rounding right-shift. `saturate_int8(round_nearest((acc32 * M0[c]) >> n[c]))`. Uses round-half-away-from-zero, matching TFLite/CMSIS-NN convention.

### 21.4 Three Int8 Convolution Kernels (kernels.py)

All three share the same calling convention: `(x_q, W_q, <requant params>, stride, padding, groups) -> int8 output`.

**`sw_int8_conv3d` (Software reference):** Uses the existing float32 conv kernel by passing int8 values as float32. The float32 kernel produces exact integer accumulators as long as intermediate sums fit in the 24-bit float mantissa. For X3D-M's layer shapes, the worst case is well under 2^24, so the int32 cast is lossless. Requantizes with float M.

**`fpga_sim_int8_conv3d` (FPGA simulator in Python):** Same integer accumulation trick, but requantizes with fixed-point (M0, n) instead of float M. This is the Python stand-in for what the FPGA fabric will do.

**`fpga_hw_int8_conv3d` (C backend):** Imported from `scratch.ops.conv3d_fpga`. Does true int8×int8→int32 arithmetic natively in C with pthreads. Uses the same (M0, n) requantization. This function gets swapped for a DMA call once the real FPGA fabric is wired up.

### 21.5 Layer Configurations (layer_configs.py)

Each entry in `LAYER_CONFIGS` is a `LayerConfig` dataclass specifying everything needed to instantiate and test a single Conv3d:

```python
@dataclass(frozen=True)
class LayerConfig:
    name: str                    # e.g., "conv_b"
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int,int,int]
    stride: Tuple[int,int,int]
    padding: Tuple[int,int,int]
    groups: int
    input_shape: Tuple[int,int,int,int,int]  # (B, C, T, H, W)
    description: str
```

Five conv types are pre-configured, covering every convolution variety in X3D-M:

| Layer | Kernel | Groups | Description |
|-------|--------|--------|-------------|
| `conv_b` | (3,3,3) | 54 (depthwise) | Bottleneck depthwise conv |
| `conv_a` | (1,1,1) | 1 (standard) | Bottleneck expand pointwise |
| `conv_c` | (1,1,1) | 1 (standard) | Bottleneck project pointwise |
| `conv_t` | (1,3,3) | 1 (standard) | Stem spatial conv |
| `conv_xy` | (5,1,1) | 24 (depthwise) | Stem temporal depthwise |

Adding a new conv type is one dict entry -- no other file needs to change.

### 21.6 The Test Runner (test_layer.py)

The CLI entry point orchestrates the full test:

```bash
python -m fpga_tests.test_layer                     # default: conv_b, seed 42
python -m fpga_tests.test_layer --layer conv_a      # test pointwise conv
python -m fpga_tests.test_layer --layer conv_t --seed 7
python -m fpga_tests.test_layer --skip-fpga-hw      # skip C backend
```

**Step-by-step execution:**

1. Build a float Conv3d from the LayerConfig with Xavier-initialized weights from a fixed seed.
2. Generate a reproducible float32 input with `np.random.default_rng(seed)`.
3. Run the float conv → `y_ref_f32` (accuracy ceiling).
4. Compute quantization parameters: `s_in`, `s_w`, `s_out`, M, (M0, n).
5. Quantize input and weights to int8.
6. Run three int8 convolutions (software, FPGA sim, FPGA HW).
7. Dequantize all results back to float32.
8. Compare with diff statistics (max abs, mean abs, RMS).
9. Save all tensors to `.npz` and a summary to `.json`.

**Pass criteria:**
- **Sim vs SW**: max int8 disagreement ≤ `--tol-lsb` (default 2 LSBs). This absorbs float-vs-fixed-point rounding differences.
- **HW vs Sim**: must be **bit-identical** (0 LSB tolerance). Both implement the exact same (M0, n) requantization.

**Reference numbers (seed 42):**

| Layer | Output Shape | Sim vs SW (int8 max) | HW vs Sim (int8 max) | SW vs float ref (RMS) |
|-------|-------------|---------------------|---------------------|---------------------|
| conv_a | (1, 54, 16, 56, 56) | ≤ 2 | 0 | ~1e-2 |
| conv_b | (1, 54, 16, 28, 28) | ≤ 1 | 0 | ~1e-2 |
| conv_c | (1, 24, 16, 28, 28) | ≤ 1 | 0 | ~1e-2 |
| conv_t | (1, 24, 16, 112, 112) | ≤ 1 | 0 | ~1e-2 |
| conv_xy | (1, 24, 16, 112, 112) | ≤ 1 | 0 | ~1e-2 |

---

## 22. FPGA Int8 C Backend (scratch/ops/conv3d_fpga_c/ and conv3d_fpga.py)

### 22.1 Purpose

The FPGA int8 C backend (`libconv3d_fpga.so`) is a CPU-executable library that implements the same datapath the real FPGA accelerator will use: int8 × int8 → int32 accumulation followed by fixed-point (M0, n) requantization to int8. When the real FPGA fabric is wired up, the body of this function gets replaced by a DMA wrapper -- the signature stays identical.

### 22.2 C Implementation (conv3d_fpga.c)

The C function `conv3d_fpga_int8()` takes:
- `int8_t *input`, `int8_t *weight`, `int8_t *output` (input/output tensors)
- `int64_t *M0`, `int32_t *n` (per-channel fixed-point requantization params)
- Shape parameters: B, C_in, T, H, W, C_out, kT, kH, kW, strides, padding, groups

It handles zero-padding internally, performs true int8×int8→int32 native multiplication (no float casting), and applies the (M0, n) requantization identically to the Python `apply_requantize_fixed_point()` function. Uses pthreads for parallelism (4 threads matching the 4 U54 cores).

### 22.3 Python ctypes Wrapper (conv3d_fpga.py)

`scratch/ops/conv3d_fpga.py` loads `libconv3d_fpga.so` at import time using ctypes, registers the C function signature, and exposes `fpga_hw_int8_conv3d()` with the same calling convention as `fpga_sim_int8_conv3d` in `fpga_tests/kernels.py`:

```python
def fpga_hw_int8_conv3d(x_q, W_q, M0, n, stride, padding, groups) -> int8 output
```

The function ensures all arrays are contiguous with the correct dtype (int8, int64, int32), computes output dimensions, allocates the output buffer, and marshals everything through ctypes.

### 22.4 Building

```bash
make -C scratch/ops/conv3d_fpga_c              # auto-detect architecture
make -C scratch/ops/conv3d_fpga_c riscv        # RISC-V target
make -C scratch/ops/conv3d_fpga_c native       # x86 native
```

---

## 23. Minimal Int8 C Test Harness (testing/ and scratch/ops/conv3d_simple_c/)

### 23.1 Purpose

Before building the full-featured FPGA C backend with requantization, a minimal "bare metal" int8 convolution was implemented to validate the most basic operation: `output[int32] = conv3d(input[int8], weight[int8])`. No quantization scales, no requantization, no threading, no bias. Just the raw multiply-accumulate.

This serves as the ground truth that the FPGA implementation must match at the accumulator level before any requantization logic is added.

### 23.2 Architecture

Two implementations share an identical C API (`conv3d_simple.h`):

**`conv3d_int8_sw`** (conv3d_sw.c): Pure-C CPU reference. Single-threaded deeply nested loops. The accumulator is int32. int8 × int8 is promoted to int32 for accumulation -- no overflow is possible for any X3D-M layer shape.

**`conv3d_int8_fpga`** (conv3d_fpga.c): FPGA-offload entry point. Today it is a **stub** that pretends to DMA data to an accelerator:
- STEP 1: `fpga_dma_to_device()` — placeholder, does nothing
- STEP 2: `fpga_launch_and_wait()` — placeholder; actually calls `conv3d_int8_sw()` as the "FPGA"
- STEP 3: `fpga_dma_from_device()` — placeholder, does nothing

The stub guarantees the two implementations agree bit-for-bit, so the comparison harness exercises the full plumbing without real hardware. When the real FPGA driver exists, only the three stub functions need to change.

### 23.3 Test Harness (main.c)

Two versions exist:

**`scratch/ops/conv3d_simple_c/main.c`**: Tests a single small depthwise 3x3x3 configuration (4 channels, 4×8×8 spatial). Prints side-by-side output values for visual inspection.

**`testing/main.c`**: Tests **all 28 unique Conv3d configurations** in X3D-M in a single run. This covers:
- 2 stem convolutions (conv_t, conv_xy)
- 4 branch1 skip connections (one per stage)
- 8 conv_a configurations (2 per stage: first block vs remaining blocks)
- 8 conv_b configurations (same split)
- 4 conv_c configurations (one per stage)
- 2 head convolutions (pre_conv, post_conv)

Each layer uses a deterministic LCG PRNG seeded per-layer for reproducibility. The test fills input and weight buffers with random int8 values, runs both implementations, and compares element-for-element:

```
[14] conv_b_s2_blk0
     input  (1, 54, 16, 112, 112)
     weight (54, 1, 3, 3, 3)  groups=54
     output (1, 54, 16, 56, 56)
     elements=2408448  mismatched=0  max_diff=0  PASS
```

### 23.4 Building and Running

```bash
# Build and run the simple single-config test
cd scratch/ops/conv3d_simple_c && make && make run

# Build and run the full 28-layer test
cd testing && make && make run
```

---

## 24. Quantization Validation Script (scripts/validate_quantization.py)

This script validates that the int8 `.npz` weights produced by `quantize_x3d_ptq.py` load correctly and produce reasonable outputs. It:

1. Builds the quantized model using `build_quantized_x3d_m()`.
2. Loads the int8 weights from the specified `.npz` file.
3. Runs a forward pass on random or calibration input.
4. Reports the output distribution, top-5 predictions, and any anomalies (NaN, inf, all-zero outputs).

This serves as a sanity check in the quantization pipeline: PTQ script → validate → deploy to SoC.

---

## 25. Profiling Dashboard (dashboard.py)

The `dashboard.py` file implements an interactive web-based profiling dashboard for visualizing X3D-M inference performance across different platforms and configurations. It uses Dash/Plotly to render:

- Real-time comparison of inference latency across platforms (macOS vs PolarFire SoC)
- Per-layer breakdown charts showing which operations are the bottlenecks
- Section-by-section timing (Stem, Stage 2, Stage 3, Stage 4, Stage 5, Head)
- Threading speedup analysis
- Memory bandwidth utilization estimates

The dashboard reads profiling JSON files from `run_stats/` and presents them in an interactive interface where you can filter by platform, convolution method, and model section.

---

## 26. Complete Convolution Layer Catalog

Every unique Conv3d configuration in X3D-M is listed below. Understanding this catalog is essential for FPGA design because each unique configuration represents a different hardware workload.

### 26.1 Stem Convolutions

| Name | In→Out | Kernel | Stride | Padding | Groups | Input Shape | Output Shape |
|------|--------|--------|--------|---------|--------|-------------|--------------|
| conv_t | 3→24 | (1,3,3) | (1,2,2) | (0,1,1) | 1 | (1,3,16,224,224) | (1,24,16,112,112) |
| conv_xy | 24→24 | (5,1,1) | (1,1,1) | (2,0,0) | 24 | (1,24,16,112,112) | (1,24,16,112,112) |

### 26.2 Branch1 (Skip Connection) Convolutions

These only appear in the first block of each stage, where dimensions change:

| Stage | In→Out | Stride | Input Shape | Output Shape |
|-------|--------|--------|-------------|--------------|
| 2 | 24→24 | (1,2,2) | (1,24,16,112,112) | (1,24,16,56,56) |
| 3 | 24→48 | (1,2,2) | (1,24,16,56,56) | (1,48,16,28,28) |
| 4 | 48→96 | (1,2,2) | (1,48,16,28,28) | (1,96,16,14,14) |
| 5 | 96→192 | (1,2,2) | (1,96,16,14,14) | (1,192,16,7,7) |

### 26.3 Bottleneck Convolutions (conv_a, conv_b, conv_c)

**conv_a** (1×1×1 pointwise expand):

| Stage | Block | In→Out | Input Shape | Output Shape |
|-------|-------|--------|-------------|--------------|
| 2 | 0 | 24→54 | (1,24,16,112,112) | (1,54,16,112,112) |
| 2 | 1-2 | 24→54 | (1,24,16,56,56) | (1,54,16,56,56) |
| 3 | 0 | 24→108 | (1,24,16,56,56) | (1,108,16,56,56) |
| 3 | 1-4 | 48→108 | (1,48,16,28,28) | (1,108,16,28,28) |
| 4 | 0 | 48→216 | (1,48,16,28,28) | (1,216,16,28,28) |
| 4 | 1-10 | 96→216 | (1,96,16,14,14) | (1,216,16,14,14) |
| 5 | 0 | 96→432 | (1,96,16,14,14) | (1,432,16,14,14) |
| 5 | 1-6 | 192→432 | (1,192,16,7,7) | (1,432,16,7,7) |

**conv_b** (3×3×3 depthwise):

| Stage | Block | Channels | Stride | Input Shape | Output Shape |
|-------|-------|----------|--------|-------------|--------------|
| 2 | 0 | 54 | (1,2,2) | (1,54,16,112,112) | (1,54,16,56,56) |
| 2 | 1-2 | 54 | (1,1,1) | (1,54,16,56,56) | (1,54,16,56,56) |
| 3 | 0 | 108 | (1,2,2) | (1,108,16,56,56) | (1,108,16,28,28) |
| 3 | 1-4 | 108 | (1,1,1) | (1,108,16,28,28) | (1,108,16,28,28) |
| 4 | 0 | 216 | (1,2,2) | (1,216,16,28,28) | (1,216,16,14,14) |
| 4 | 1-10 | 216 | (1,1,1) | (1,216,16,14,14) | (1,216,16,14,14) |
| 5 | 0 | 432 | (1,2,2) | (1,432,16,14,14) | (1,432,16,7,7) |
| 5 | 1-6 | 432 | (1,1,1) | (1,432,16,7,7) | (1,432,16,7,7) |

**conv_c** (1×1×1 pointwise project):

| Stage | In→Out | Input Shape | Output Shape |
|-------|--------|-------------|--------------|
| 2 | 54→24 | (1,54,16,56,56) | (1,24,16,56,56) |
| 3 | 108→48 | (1,108,16,28,28) | (1,48,16,28,28) |
| 4 | 216→96 | (1,216,16,14,14) | (1,96,16,14,14) |
| 5 | 432→192 | (1,432,16,7,7) | (1,192,16,7,7) |

### 26.4 Head Convolutions

| Name | In→Out | Input Shape | Output Shape |
|------|--------|-------------|--------------|
| pre_conv | 192→432 | (1,192,16,7,7) | (1,432,16,7,7) |
| post_conv | 432→2048 | (1,432,1,1,1) | (1,2048,1,1,1) |

### 26.5 SE Convolutions (inside even-indexed blocks)

| SE Layer | Mid Channels | Example (Stage 2, inner=54) |
|----------|-------------|----------------------------|
| conv1 | `_round_width(inner, 0.0625)` | 54 × 0.0625 = 3.375 → rounded to 8 |
| conv2 | Same mid → inner | 8 → 54 |

SE conv1/conv2 are 1×1×1 pointwise with bias=True, operating on `(B, C, 1, 1, 1)` tensors (after global average pooling).

---

## 27. FPGA Integration Plan (fpga_flow.md)

The project includes a comprehensive FPGA integration planning document (`fpga_flow.md`) that lays out the full roadmap for moving from software-only execution to hardware-accelerated inference on the PolarFire SoC. Key sections include:

### 27.1 Three-Phase Deployment

**Phase 1 (Current):** Float32 software inference on RISC-V CPU. Multi-threaded Python/OpenCV or C backend. Validates correctness against PyTorch reference.

**Phase 2 (In Progress):** Int8 quantized inference with software reference kernel. BN folding, PTQ calibration, per-layer FPGA bring-up testing. The quantized model runs end-to-end on the CPU using the int8 reference kernel.

**Phase 3 (Future):** FPGA-accelerated int8 convolution. The C backend's function body is replaced with DMA transfers to the FPGA fabric. The rest of the pipeline (quantize/dequantize at boundaries, activations, pooling, linear) stays on the CPU.

### 27.2 CPU/FPGA Handshake Protocol

For each convolution layer during inference:

1. **CPU** quantizes the float32 activation to int8 using `input_scale`
2. **CPU** DMAs the int8 input tensor and int8 weights to FPGA-visible memory
3. **CPU** writes shape registers and the (M0, n) requantization table to the FPGA
4. **CPU** triggers the FPGA accelerator and waits for completion
5. **FPGA** performs int8×int8→int32 convolution with spatial tiling
6. **FPGA** applies (M0, n) requantization to produce int8 output
7. **CPU** DMAs the int8 output back
8. **CPU** dequantizes to float32 for the next non-conv operation (activation, pooling)

### 27.3 Freeze-One-Layer-At-A-Time Discipline

The bring-up strategy validates each layer independently before combining:

1. Test `conv_b` (depthwise 3×3×3) — the most common and FPGA-friendly conv
2. Test `conv_a` (pointwise 1×1×1) — simple but high-throughput
3. Test `conv_c` (pointwise 1×1×1) — same as conv_a but different channel dims
4. Test `conv_t` (standard 1×3×3) — the stem spatial conv
5. Test `conv_xy` (depthwise 5×1×1) — the stem temporal conv

Only after all five pass with 0 LSB HW-vs-Sim tolerance do we integrate into the full model.

---

## 28. Updated File Layout

The complete project file layout, including all components documented in this file:

```
x3d/
├── main.py                       # Float32 inference entry point with profiling (CLI)
├── main_int8.py                  # Int8 hybrid inference entry point (CLI)
├── dashboard.py                  # Interactive web profiling dashboard (Dash/Plotly)
├── visualize_stats.py            # Cross-platform profiling comparison & charts
├── x3d_layers.py                 # PyTorch reference implementation (verification only)
├── fpga_flow.md                  # Comprehensive FPGA integration plan
├── DOCUMENTATION.md              # This file
├── CLAUDE.md                     # Project conventions for AI assistants
├── README.md                     # Quick-start README
│
├── scratch/                      # The PyTorch-free neural network library
│   ├── __init__.py               # Public API: X3D_M, set_conv3d_method, etc.
│   ├── ops/                      # Stateless mathematical operations
│   │   ├── __init__.py           # Re-exports all ops
│   │   ├── conv3d.py             # 4 conv methods (slow/fast/threaded/native)
│   │   ├── conv3d_c/             # Float32 C backend (libconv3d.so)
│   │   │   ├── conv3d.c          # Pthreads, spatial tiling, 3 fast paths
│   │   │   ├── conv3d.h          # C API header
│   │   │   └── Makefile          # Build for RISC-V or native x86
│   │   ├── conv3d_fpga.py        # Python ctypes wrapper for FPGA int8 C backend
│   │   ├── conv3d_fpga_c/        # Int8 FPGA-offload C backend (libconv3d_fpga.so)
│   │   │   ├── conv3d_fpga.c     # int8×int8→int32 + (M0,n) requant, pthreads
│   │   │   ├── conv3d_fpga.h     # C API header
│   │   │   └── Makefile          # Build for RISC-V or native x86
│   │   ├── conv3d_simple_c/      # Minimal int8 conv: SW + FPGA stub + test
│   │   │   ├── conv3d_simple.h   # Shared API header
│   │   │   ├── conv3d_sw.c       # CPU reference (nested loops)
│   │   │   ├── conv3d_fpga.c     # FPGA stub (DMA placeholders)
│   │   │   ├── main.c            # Small single-config compare test
│   │   │   └── Makefile
│   │   ├── batchnorm3d.py        # 3D batch normalization
│   │   ├── activations.py        # relu, silu, sigmoid
│   │   ├── pooling.py            # avg_pool3d, adaptive_avg_pool3d
│   │   ├── linear.py             # Dense/fully-connected layer
│   │   └── dropout.py            # Dropout regularization
│   ├── nn/                       # Stateful neural network modules
│   │   ├── __init__.py           # Re-exports all nn modules
│   │   ├── module.py             # Base Module class
│   │   ├── sequential.py         # Sequential + ModuleList
│   │   ├── conv3d.py             # Conv3d layer
│   │   ├── batchnorm3d.py        # BatchNorm3d layer
│   │   ├── squeeze_excitation.py # SE attention block
│   │   ├── bottleneck.py         # conv_a → conv_b → conv_c pipeline
│   │   ├── resblock.py           # Bottleneck + skip connection
│   │   ├── resstage.py           # Stage of ResBlocks
│   │   ├── stem.py               # (2+1)D factorized stem
│   │   └── head.py               # Classification head
│   ├── quantized/                # Int8 quantized runtime (FPGA-oriented)
│   │   ├── __init__.py           # Public API
│   │   ├── conv3d_int8.py        # Software reference int8 conv kernel
│   │   ├── layers.py             # QuantizedConv3d + QuantizedLinear modules
│   │   ├── load_int8_weights.py  # Int8 .npz weight loader
│   │   └── model.py              # QuantizedX3D_M builder
│   ├── models/
│   │   ├── __init__.py
│   │   └── x3d_m.py              # Full X3D-M model assembly
│   ├── load_weights.py           # Float32 .npz weight loader
│   └── stats.py                  # StatsCollector, FLOPs estimation
│
├── scripts/
│   ├── convert_pytorch_weights_to_numpy.py  # PyTorch → float32 .npz
│   ├── quantize_x3d_ptq.py                 # Float32 → int8 .npz (PTQ)
│   └── validate_quantization.py             # Validate int8 weights
│
├── fpga_tests/                   # Per-layer FPGA offload validation
│   ├── __init__.py
│   ├── README.md                 # Detailed harness documentation
│   ├── test_layer.py             # CLI: float, SW-int8, FPGA-sim, FPGA-HW
│   ├── kernels.py                # Three int8 conv implementations
│   ├── quant.py                  # Int8 quantization / requantization primitives
│   ├── layer_configs.py          # Conv layer specs for all X3D-M conv types
│   └── runs/                     # Output tensors + JSON summaries (gitignored)
│
├── testing/                      # Minimal int8 C conv: full 28-layer test
│   ├── README.md
│   ├── conv3d_simple.h           # Shared C API
│   ├── conv3d_sw.c               # CPU reference
│   ├── conv3d_fpga.c             # FPGA stub
│   ├── main.c                    # 28-layer compare harness
│   ├── layers.md                 # Layer configuration documentation
│   └── Makefile
│
├── docs/                         # Supplementary documentation
│   ├── In Depth.md               # Deep-dive: memory layout, conv math, data flow
│   ├── Math.md                   # Mathematical foundations
│   ├── Quantization.md           # Quantization theory and practice
│   └── SCRATCH_IMPLEMENTATION.md # Implementation notes
│
├── weights/                      # .npz weight files (gitignored)
├── run_stats/                    # Profiling output (gitignored)
├── archive/                      # Legacy C++/PyTorch code (not used)
└── build/                        # CMake build artifacts (legacy)
```

---

## 29. Suggested Diagrams and Figures

The following diagrams would enhance this documentation if included in the official report. Each is described textually here for reference:

### Figure 1: X3D-M Architecture Block Diagram
A top-down flowchart showing: Input (3×16×224×224) → Stem → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Head → Output (400). Each stage box shows depth, channel counts, and spatial resolution.

### Figure 2: ResBlock Internal Structure
Two parallel paths: Branch 2 (conv_a → BN → ReLU → conv_b → BN → [SE] → SiLU → conv_c → BN) and Branch 1 (Identity or 1×1 Conv+BN). Merge with element-wise addition → ReLU.

### Figure 3: Bottleneck Channel Flow (Inverted Bottleneck)
A width diagram showing: narrow (in_channels) → wide (inner_channels via conv_a) → wide (inner_channels via conv_b depthwise) → narrow (out_channels via conv_c). Shows the "expand → filter → project" pattern.

### Figure 4: Squeeze-and-Excitation Block
Diagram showing: Input (B,C,T,H,W) → Global Avg Pool → (B,C,1,1,1) → Conv1×1 → ReLU → Conv1×1 → Sigmoid → (B,C,1,1,1) attention weights → Multiply with Input.

### Figure 5: (2+1)D Factorized Stem
Two sequential boxes: conv_t (1×3×3, spatial filtering, 224→112) → conv_xy (5×1×1, temporal filtering, T preserved) → BN → ReLU.

### Figure 6: ops/ vs nn/ Two-Layer Architecture
Side-by-side diagram: Left column (ops/) shows pure functions with arrows in/out. Right column (nn/) shows Module classes containing _parameters dict and calling ops/ functions in forward().

### Figure 7: Convolution Method Selection Flowchart
Decision tree: Is method "native"? → C backend. Is method "threaded"? → Is depthwise? → Temporal parallelism / Output-channel parallelism. Is method "fast"? → Single-threaded OpenCV. Else → Pure NumPy loops.

### Figure 8: Int8 Quantization Pipeline
Linear flow: Float32 Model → BN Folding → Calibration (collect activation ranges) → Per-channel weight quantization → Per-tensor activation quantization → Int8 .npz export.

### Figure 9: Quantized Conv3d Forward Pass
Three-step diagram: float32 input → [Quantize: ÷ s_in, round, clip] → int8 → [Int8 Conv: int8×int8→int32, +bias_q, ×M, round, clip] → int8 → [Dequantize: × s_out] → float32 output.

### Figure 10: FPGA Test Harness Four-Path Comparison
Four parallel lanes showing Float Ref, SW Int8, FPGA Sim, FPGA HW, all starting from the same input and converging at a comparison node that reports diff statistics.

### Figure 11: PolarFire SoC Block Diagram
Shows: 1× E51 Monitor Core + 4× U54 Application Cores → L2 Cache → DDR4 Controller → 2GB LPDDR4. Side connection: AXI bus → FPGA Fabric (254K LEs, 784 Math Blocks).

### Figure 12: Spatial Resolution Through the Network
A stepped bar chart showing H×W at each stage: 224×224 → 112×112 (Stem) → 56×56 (Stage 2) → 28×28 (Stage 3) → 14×14 (Stage 4) → 7×7 (Stage 5) → 1×1 (Head pool).

### Figure 13: Channel Count Through the Network
Bar chart: 3 (input) → 24 (Stem) → 24/54 (Stage 2) → 48/108 (Stage 3) → 96/216 (Stage 4) → 192/432 (Stage 5) → 432→2048→400 (Head). Shows outer/inner channel pairs.

### Figure 14: C Backend Spatial Tiling
Diagram of a 2D spatial plane divided into TILE_H×TILE_W (8×16) tiles, with the kernel's receptive field overlapping tile boundaries, showing how each tile fits in the 32KB L1 cache.

### Figure 15: Threading Strategy Decision
Two-column comparison: Left (Output-Channel Parallelism): many small tasks spread across 4 threads, used for pointwise/standard convs. Right (Temporal Parallelism): few large tasks, each thread processes T/4 temporal positions, used for depthwise convs.

---

## 30. Summary of All Implemented Components

| Component | Status | Files | Description |
|-----------|--------|-------|-------------|
| Float32 scratch library | Complete | scratch/ops/, scratch/nn/ | PyTorch-free NumPy/OpenCV inference |
| X3D-M model assembly | Complete | scratch/models/x3d_m.py | Full model with pretrained weight loading |
| Pure NumPy conv (slow) | Complete | scratch/ops/conv3d.py | Correctness reference, 6-deep nested loops |
| OpenCV conv (fast) | Complete | scratch/ops/conv3d.py | Single-threaded cv2.filter2D |
| Multi-threaded conv (threaded) | Complete | scratch/ops/conv3d.py | Adaptive hybrid parallelism, 4 threads |
| C backend conv (native) | Complete | scratch/ops/conv3d_c/ | Pthreads, spatial tiling, 3 fast paths |
| Float32 weight converter | Complete | scripts/convert_pytorch_weights_to_numpy.py | PyTorch → .npz |
| Float32 weight loader | Complete | scratch/load_weights.py | .npz → scratch model |
| Profiling system | Complete | scratch/stats.py, main.py | Per-layer timing, FLOPs, platform detection |
| Visualization | Complete | visualize_stats.py | Charts, HTML reports, cross-platform comparison |
| PyTorch reference | Complete | x3d_layers.py | Verification against official PyTorchVideo |
| Int8 PTQ script | Complete | scripts/quantize_x3d_ptq.py | BN folding, calibration, export int8 .npz |
| Int8 quantized runtime | Complete | scratch/quantized/ | QuantizedConv3d, QuantizedX3D_M, int8 loader |
| Int8 reference conv kernel | Complete | scratch/quantized/conv3d_int8.py | Pure NumPy int8 conv for validation |
| Int8 inference entry point | Complete | main_int8.py | CLI for hybrid int8 model |
| FPGA int8 C backend | Complete | scratch/ops/conv3d_fpga_c/ | int8×int8→int32 + (M0,n) requant in C |
| FPGA Python wrapper | Complete | scratch/ops/conv3d_fpga.py | ctypes bindings for libconv3d_fpga.so |
| FPGA per-layer test harness | Complete | fpga_tests/ | 4-path comparison, 5 conv types |
| Minimal int8 C test | Complete | testing/, conv3d_simple_c/ | 28-layer SW-vs-FPGA-stub comparison |
| Quantization validation | Complete | scripts/validate_quantization.py | Sanity check int8 weights |
| Dashboard | Complete | dashboard.py | Interactive web profiling dashboard |
| Real FPGA fabric driver | Not started | — | Replace C backend with DMA calls |
| BN folding in scratch runtime | Not started | — | Fold BN into conv weights at load time |
| Int8 lookup table for SiLU | Not started | — | 256-entry table replacing dequant/SiLU/requant |


## 18. Glossary

**Action Recognition**: the computer vision task of classifying the human action being performed in a video clip (e.g., "playing basketball", "cooking").

**Activation Function**: a non-linear function applied element-wise to a layer's output to introduce non-linearity into the network.

**Batch Normalization (BatchNorm, BN)**: a normalization technique that standardizes activations within a mini-batch to have zero mean and unit variance, then applies learned scale and shift parameters.

**Bottleneck**: a network block that first expands the channel dimension, performs spatial/temporal computation in the expanded space, then compresses back. Named for its hourglass-like shape.

**Channel**: a dimension of a tensor representing different feature maps. RGB images have 3 channels; intermediate CNN layers may have dozens or hundreds.

**Convolution (Conv)**: a mathematical operation that slides a learnable filter over input data, computing dot products to produce feature maps.

**Depthwise Convolution**: a convolution where each input channel is filtered independently by its own kernel. No cross-channel mixing occurs.

**DMA (Direct Memory Access)**: a hardware mechanism for transferring data between memory and peripherals (like FPGA fabric) without CPU involvement.

**Dropout**: a regularization technique that randomly sets activations to zero during training to prevent overfitting.

**FPGA (Field-Programmable Gate Array)**: reconfigurable hardware that can be programmed to implement custom digital circuits, including neural network accelerators.

**FLOPs (Floating-Point Operations)**: a measure of computational complexity, counting the number of floating-point multiply and add operations.

**Forward Pass**: computing the output of a neural network given an input, by passing data sequentially through all layers.

**GIL (Global Interpreter Lock)**: a Python mechanism that prevents multiple threads from executing Python bytecode simultaneously.

**Global Average Pooling**: averaging all spatial (and temporal) positions in a feature map into a single value per channel.

**Inference**: using a trained neural network to make predictions on new data (as opposed to training, which adjusts the network's weights).

**Inverted Bottleneck**: a block where channels are expanded before the main computation and compressed after, opposite to the original ResNet bottleneck which compresses first.

**Kernel (Filter)**: a small, learnable tensor that is convolved with the input to extract features.

**Kinetics-400**: a large-scale video dataset with 400 human action categories, commonly used for action recognition benchmarks.

**LPDDR4**: Low-Power Double Data Rate 4 memory, used on the PolarFire SoC Icicle Kit.

**Module**: in this project, a base class for neural network layers that manages parameters and child modules.

**NumPy**: a Python library for numerical computation with multi-dimensional arrays. The backbone of the scratch library.

**OpenCV**: an open-source computer vision library. Used in this project for its optimized 2D convolution implementation (`cv2.filter2D`).

**Padding**: adding zeros (or other values) around the border of a tensor to control the output size of a convolution.

**Pointwise Convolution**: a convolution with kernel size 1x1x1 that mixes information across channels without any spatial/temporal filtering.

**Pooling**: an operation that reduces spatial dimensions by summarizing local regions (e.g., average pooling computes the mean of each region).

**PyTorchVideo**: Facebook's video understanding library built on PyTorch. Provides the official X3D implementation and pretrained weights.

**ReLU (Rectified Linear Unit)**: an activation function: `f(x) = max(0, x)`.

**Residual Connection (Skip Connection)**: a shortcut that adds the input of a block directly to its output, enabling gradient flow through very deep networks.

**RISC-V**: an open-source instruction set architecture. The PolarFire SoC uses SiFive's U54 and E51 RISC-V cores.

**SE (Squeeze-and-Excitation)**: an attention mechanism that learns per-channel importance weights via global pooling and two small fully-connected layers.

**SiLU (Sigmoid Linear Unit, Swish)**: an activation function: `f(x) = x * sigmoid(x)`.

**Sigmoid**: an activation function: `f(x) = 1 / (1 + e^(-x))`, mapping values to the range (0, 1).

**SoC (System-on-Chip)**: a single chip that integrates multiple components (processor, memory controller, peripherals, and in this case, FPGA fabric).

**Stride**: the step size of a convolution kernel as it slides across the input. A stride of 2 halves the spatial dimensions.

**Tensor**: a multi-dimensional array of numbers, the fundamental data structure in neural network computation.

**(2+1)D Factorization**: decomposing a 3D convolution into a 2D spatial convolution followed by a 1D temporal convolution, reducing parameters while maintaining expressiveness.

**X3D (eXpand 3D)**: a family of efficient 3D CNN architectures for video understanding, designed by Facebook AI Research through progressive network expansion.

---

