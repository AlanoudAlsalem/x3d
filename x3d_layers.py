"""
X3D-M Layer-by-Layer Recreation in PyTorch
===========================================

This file reconstructs the X3D-M (eXpand-3D, Medium variant) video classification
model layer by layer. X3D is a family of efficient 3D CNNs for video understanding,
designed by Facebook AI Research. It processes short video clips (16 frames) and
classifies them into one of 400 action categories (Kinetics-400 dataset).

The goal of this reconstruction is to make every layer explicit and inspectable,
so that individual convolutional layers can later be offloaded to an FPGA fabric
(e.g. PolarFire SoC) while the RISC-V core handles control flow, batch norm,
activation functions, and data orchestration.

Architecture overview (X3D-M):
    Input:  [B, 3, 16, 224, 224]  -- B=batch, 3=RGB, 16=frames, 224x224 spatial
    Stem:   Conv2plus1d -> BatchNorm -> ReLU             => [B, 24, 16, 112, 112]
    Stage2: 3  ResBlocks (24  -> 54  inner -> 24  out)   => [B, 24, 16, 56, 56]
    Stage3: 5  ResBlocks (48  -> 108 inner -> 48  out)   => [B, 48, 16, 28, 28]
    Stage4: 11 ResBlocks (96  -> 216 inner -> 96  out)   => [B, 96, 16, 14, 14]
    Stage5: 7  ResBlocks (192 -> 432 inner -> 192 out)   => [B, 192, 16, 7, 7]
    Head:   Conv5 -> AvgPool -> FC(2048) -> FC(400)      => [B, 400]

Total parameters: ~3.79M (very lightweight for a video model)

PyTorch API Reference (for the reader):
========================================
- torch.nn.Module: Base class for all neural network modules. You subclass it,
  define layers in __init__, and implement forward() to describe the computation.

- torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias, groups):
  3D convolution over a 5D input [B, C, D, H, W].
    - kernel_size: can be int (same for all 3 dims) or tuple (t, h, w)
    - stride: spatial/temporal step size
    - padding: zero-padding added to each side of each dimension
    - groups=1: standard convolution
    - groups=in_channels: depthwise convolution (each channel convolved separately)
      This is the key to X3D's efficiency -- depthwise separable convolutions
      dramatically reduce parameter count and FLOPs.
    - bias=False: typically used before BatchNorm (BN absorbs the bias)

- torch.nn.BatchNorm3d(num_features, eps, momentum):
  Normalizes activations over [B, D, H, W] for each channel independently.
    - Learns two parameters per channel: gamma (scale) and beta (shift)
    - eps: small constant for numerical stability in division
    - momentum: controls running mean/variance update speed

- torch.nn.ReLU(inplace): max(0, x). inplace=True saves memory by modifying
  the tensor in place instead of allocating a new one.

- torch.nn.SiLU() (aka Swish): x * sigmoid(x). Smoother than ReLU, used in
  the bottleneck blocks after the depthwise convolution.

- torch.nn.Sigmoid(): Squashes values to [0, 1]. Used in Squeeze-and-Excitation
  to produce per-channel attention weights.

- torch.nn.AdaptiveAvgPool3d(output_size): Pools to a fixed output size regardless
  of input size. output_size=(1,1,1) means global average pooling.

- torch.nn.AvgPool3d(kernel_size, stride): Fixed-size average pooling window.

- torch.nn.Linear(in_features, out_features, bias): Fully connected / dense layer.

- torch.nn.Dropout(p): During training, randomly zeros elements with probability p.
  During eval, acts as identity (no-op).

- torch.nn.Identity(): A no-op module that just passes input through. Used as a
  placeholder when Squeeze-and-Excitation is skipped on odd-indexed blocks.

- torch.nn.Sequential(*modules): Chains modules sequentially. Calling it on an
  input runs each module in order, passing output of one as input to the next.

- torch.nn.ModuleList(modules): A list of modules properly registered with PyTorch
  so that .parameters() and .to(device) work correctly. Unlike a plain Python list,
  ModuleList ensures PyTorch tracks contained modules for gradient computation.
"""

import torch
import torch.nn as nn


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise attention.

    SE recalibrates channel responses by:
      1. Squeeze:  Global average pool -> [B, C, 1, 1, 1]  (one value per channel)
      2. Excite:   Two FC layers (implemented as 1x1x1 convolutions):
                   Conv3d(C -> C*ratio) -> ReLU -> Conv3d(C*ratio -> C) -> Sigmoid
      3. Scale:    Element-wise multiply original features by the attention weights

    The bottleneck width is computed using round_width(), which rounds to the nearest
    multiple of 8 (with a minimum of 8). For example:
      - 54  channels * 0.0625 = 3.375 -> rounded to 8
      - 108 channels * 0.0625 = 6.75  -> rounded to 8
      - 216 channels * 0.0625 = 13.5  -> rounded to 16
      - 432 channels * 0.0625 = 27.0  -> rounded to 32

    For FPGA offloading: SE is cheap (two tiny 1x1x1 convolutions on a [B,C,1,1,1]
    tensor), so it's best to keep it on the RISC-V core rather than offloading.
    The global average pool that feeds it is also simple to compute on CPU.

    PyTorch note: We use Conv3d with kernel_size=1 instead of nn.Linear because
    the tensor is 5D [B,C,1,1,1]. Conv3d(1x1x1) on a 5D tensor is mathematically
    identical to a fully-connected layer but avoids reshape operations.
    """
    @staticmethod
    def _round_width(width: int, multiplier: float, min_width: int = 8,
                     divisor: int = 8) -> int:
        """
        Rounds width*multiplier to the nearest multiple of divisor, with a minimum.
        This is the same rounding used throughout X3D for channel calculations.
        """
        width = width * multiplier
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def __init__(self, channels: int, se_ratio: float = 0.0625):
        super().__init__()
        # Bottleneck width: compress channel dimension by se_ratio, rounded
        # to the nearest multiple of 8 (min 8). This matches PyTorchVideo's
        # round_width() function which ensures hardware-friendly channel counts.
        mid_channels = self._round_width(channels, se_ratio)

        self.block = nn.Sequential(
            # FC1: compress channels -> mid_channels (with bias, since no BN follows)
            nn.Conv3d(channels, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(),
            # FC2: expand mid_channels -> channels, sigmoid produces [0,1] weights
            nn.Conv3d(mid_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] -- feature map after depthwise conv + BN

        Returns:
            [B, C, T, H, W] -- re-weighted feature map
        """
        # Squeeze: global average pool across T, H, W -> [B, C, 1, 1, 1]
        # adaptive_avg_pool3d(input, output_size) pools to a fixed spatial size.
        # Using (1,1,1) means we average ALL temporal and spatial positions into
        # a single value per channel -- this captures "how active is this channel
        # across the entire clip?"
        scale = torch.nn.functional.adaptive_avg_pool3d(x, 1)

        # Excite: learn channel attention weights
        scale = self.block(scale)  # [B, C, 1, 1, 1] with values in [0, 1]

        # Scale: broadcast-multiply attention weights across all spatial positions.
        # Broadcasting: [B,C,1,1,1] * [B,C,T,H,W] -> [B,C,T,H,W]
        # Each channel gets uniformly scaled by its attention weight.
        return x * scale


class BottleneckBlock(nn.Module):
    """
    X3D Bottleneck block: the core computational unit.

    Architecture:
        conv_a:  1x1x1 pointwise conv  (expand channels:  in_ch -> inner_ch)
        norm_a:  BatchNorm3d
        act_a:   ReLU

        conv_b:  3x3x3 depthwise conv  (spatial+temporal filtering, groups=inner_ch)
        norm_b:  BatchNorm3d + optional SqueezeExcitation
        act_b:   Swish (SiLU)

        conv_c:  1x1x1 pointwise conv  (project channels: inner_ch -> out_ch)
        norm_c:  BatchNorm3d

    This follows the "inverted bottleneck" pattern:
      - conv_a EXPANDS channels (e.g., 24 -> 54) to create a richer representation
      - conv_b applies depthwise spatial+temporal filtering in this expanded space
        (depthwise = groups equals the number of channels, so each channel gets
        its own 3x3x3 filter -- no cross-channel mixing here)
      - conv_c PROJECTS back down to the output channel count (e.g., 54 -> 24)

    Why depthwise? A standard 3D conv with 54 input and 54 output channels and a
    3x3x3 kernel has 54*54*27 = 78,732 parameters. A depthwise conv has only
    54*27 = 1,458 parameters -- a 54x reduction! The pointwise (1x1x1) convolutions
    before and after handle the cross-channel mixing that depthwise conv skips.

    SE (Squeeze-and-Excitation) is applied on even-indexed blocks (0-indexed from
    the ResBlock perspective -- blocks at positions 1, 3, 5, ... within a stage get SE,
    while blocks at 0, 2, 4, ... get Identity).

    For FPGA offloading: conv_a, conv_b, conv_c are the convolution-heavy ops.
    conv_b (depthwise 3x3x3) is the most computationally expensive per element.
    conv_a and conv_c are 1x1x1 pointwise (essentially matrix multiplies).
    BN, ReLU, Swish can stay on the RISC-V core.
    """
    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = False,
        se_ratio: float = 0.0625,
    ):
        super().__init__()

        # --- Branch A: expand channels with 1x1x1 pointwise convolution ---
        # kernel_size=1 means no spatial filtering, just a learned linear combination
        # across channels at each spatial position. Think of it as a per-pixel
        # fully-connected layer that maps in_channels -> inner_channels.
        self.conv_a = nn.Conv3d(
            in_channels, inner_channels,
            kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.norm_a = nn.BatchNorm3d(inner_channels, eps=1e-5, momentum=0.1)
        self.act_a = nn.ReLU(inplace=True)

        # --- Branch B: depthwise 3D convolution ---
        # groups=inner_channels makes this DEPTHWISE: each channel gets its own
        # independent 3x3x3 filter. No mixing between channels.
        # stride=(1, stride, stride): temporal stride is always 1 (we keep all 16
        # frames), spatial stride is 2 for the first block of each stage (downsamples
        # H and W by 2x) and 1 for subsequent blocks.
        # padding=(1,1,1): "same" padding for 3x3x3 kernel so output size matches
        # input size (before stride).
        self.conv_b = nn.Conv3d(
            inner_channels, inner_channels,
            kernel_size=3, stride=(1, stride, stride), padding=1,
            groups=inner_channels, bias=False,
        )

        # norm_b is an nn.Sequential to match the pretrained checkpoint structure:
        #   norm_b.0 = BatchNorm3d
        #   norm_b.1 = SqueezeExcitation (or Identity on odd-indexed blocks)
        # This way, state_dict keys like "norm_b.0.weight" and "norm_b.1.block.0.weight"
        # map directly to the pretrained weights.
        #
        # Squeeze-and-Excitation adds channel attention AFTER BN, BEFORE Swish.
        # On blocks where SE is not used, Identity passes the tensor through unchanged.
        if use_se:
            se_module = SqueezeExcitation(inner_channels, se_ratio)
        else:
            se_module = nn.Identity()

        self.norm_b = nn.Sequential(
            nn.BatchNorm3d(inner_channels, eps=1e-5, momentum=0.1),
            se_module,
        )

        # Swish (SiLU): f(x) = x * sigmoid(x)
        # Smoother gradient than ReLU near zero, which helps training convergence.
        # Used after depthwise conv in X3D (after SE recalibration).
        self.act_b = nn.SiLU(inplace=True)

        # --- Branch C: project channels with 1x1x1 pointwise convolution ---
        # Maps inner_channels -> out_channels (compression step).
        # No activation after this -- the residual addition + ReLU happens in ResBlock.
        self.conv_c = nn.Conv3d(
            inner_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.norm_c = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, T, H, W]
        Returns:
            [B, out_channels, T, H', W']  where H'=H/stride, W'=W/stride
        """
        # Expand channels
        x = self.act_a(self.norm_a(self.conv_a(x)))

        # Depthwise spatial+temporal filtering + SE attention
        # norm_b is Sequential(BatchNorm3d, SE_or_Identity)
        x = self.norm_b(self.conv_b(x))
        x = self.act_b(x)    # Swish activation

        # Project channels back down
        x = self.norm_c(self.conv_c(x))
        return x


class ResBlock(nn.Module):
    """
    Residual Block: wraps a BottleneckBlock with a skip connection.

    The skip (residual) connection is the key idea from ResNet: instead of learning
    the output directly, the network learns the RESIDUAL (the difference from the
    input). This makes it much easier to train very deep networks because gradients
    can flow directly through the skip connection.

        output = ReLU(bottleneck(x) + shortcut(x))

    If the input and output have the same shape, the shortcut is just identity (pass
    the input through unchanged). If shapes differ (channel count changes or spatial
    downsampling), a 1x1x1 conv + BN adapts the shortcut to match.

    For FPGA offloading: The skip connection itself is just tensor addition (very
    cheap). The branch1_conv (when present) is a small 1x1x1 convolution.
    """
    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = False,
        se_ratio: float = 0.0625,
    ):
        super().__init__()

        # Main path: the bottleneck computation
        self.branch2 = BottleneckBlock(
            in_channels, inner_channels, out_channels,
            stride=stride, use_se=use_se, se_ratio=se_ratio,
        )

        # Skip connection path: adapt dimensions if needed
        # branch1_conv is needed when spatial dimensions change (stride=2) OR
        # channel count changes. branch1_norm (BatchNorm) is only added when
        # the channel count actually changes (not for same-channel downsampling).
        # This matches PyTorchVideo's behavior where Stage 2 block 0 has
        # branch1_conv (for stride-2) but no branch1_norm (channels stay at 24).
        self.has_branch1 = (in_channels != out_channels) or (stride != 1)
        self.has_branch1_norm = (in_channels != out_channels)

        if self.has_branch1:
            self.branch1_conv = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, stride=(1, stride, stride), padding=0, bias=False,
            )
        if self.has_branch1_norm:
            self.branch1_norm = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)

        # Final activation after residual addition
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, T, H, W]
        Returns:
            [B, out_channels, T, H', W']
        """
        # Compute the residual (main path)
        residual = self.branch2(x)

        # Compute the shortcut (skip connection)
        if self.has_branch1:
            shortcut = self.branch1_conv(x)
            if self.has_branch1_norm:
                shortcut = self.branch1_norm(shortcut)
        else:
            shortcut = x

        # Add and activate: this is the core residual learning equation
        return self.activation(residual + shortcut)


class ResStage(nn.Module):
    """
    A stage is a sequence of ResBlocks with the same output channel count.

    X3D-M has 4 residual stages (stages 2-5 in the paper's notation):
      Stage 2: 3  blocks, channels 24  -> 54  -> 24,  stride=2 on first block
      Stage 3: 5  blocks, channels 48  -> 108 -> 48,  stride=2 on first block
      Stage 4: 11 blocks, channels 96  -> 216 -> 96,  stride=2 on first block
      Stage 5: 7  blocks, channels 192 -> 432 -> 192, stride=2 on first block

    Only the FIRST block in each stage does spatial downsampling (stride=2).
    All subsequent blocks maintain the same spatial resolution (stride=1).
    The first block also handles the channel dimension change from the previous
    stage's output channels to this stage's output channels.

    SE (Squeeze-and-Excitation) is applied on even-indexed blocks within each stage
    (i.e., blocks 0, 2, 4, ... get SE; blocks 1, 3, 5, ... get Identity).
    """
    def __init__(
        self,
        depth: int,
        in_channels: int,
        inner_channels: int,
        out_channels: int,
        stride: int = 2,
        se_ratio: float = 0.0625,
    ):
        super().__init__()

        blocks = []
        for i in range(depth):
            blocks.append(ResBlock(
                # First block takes input from previous stage; rest take this stage's output
                in_channels=in_channels if i == 0 else out_channels,
                inner_channels=inner_channels,
                out_channels=out_channels,
                # Only the first block downsamples spatially
                stride=stride if i == 0 else 1,
                # SE on even-indexed blocks (0, 2, 4, ...)
                use_se=(i % 2 == 0),
                se_ratio=se_ratio,
            ))

        # nn.ModuleList: stores sub-modules in an ordered list.
        # Unlike nn.Sequential, we iterate manually in forward() -- this gives us
        # the option to insert hooks, logging, or FPGA offloading calls between blocks.
        self.res_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x)
        return x


# =============================================================================
# STEM (Input processing)
# =============================================================================

class Conv2plus1dStem(nn.Module):
    """
    X3D Stem: the entry point that processes raw video frames.

    Instead of a single 3D convolution, X3D uses a (2+1)D factorized convolution:
      1. conv_t: Spatial conv (1x3x3) -- standard convolution that filters each
                 frame spatially, with stride=(1,2,2) to halve spatial resolution.
                 Maps 3 RGB channels to 24 output channels.
      2. conv_xy: Temporal conv (5x1x1) -- depthwise convolution (groups=24) that
                  filters across the time dimension. Each of the 24 channels gets
                  its own independent 5-frame temporal filter.

    NOTE: The naming conv_t/conv_xy is PyTorchVideo's convention, which is
    counterintuitive -- conv_t actually does SPATIAL filtering and conv_xy
    does TEMPORAL filtering. We keep these names to match the pretrained
    checkpoint's key names for direct weight loading.

    Why (2+1)D? Factoring a 5x3x3 convolution into 1x3x3 + 5x1x1:
      - Standard 5x3x3 conv: 3*24*5*3*3 = 3,240 params
      - Factored (2+1)D: (3*24*1*3*3) + (24*1*5*1*1) = 648 + 120 = 768 params
      That's ~4x fewer parameters! The factorization also separates spatial
      and temporal processing, which helps the model learn each independently.

    Input:  [B, 3, 16, 224, 224]
    Output: [B, 24, 16, 112, 112]

    For FPGA offloading: conv_t and conv_xy are small convolutions that process
    the full-resolution input. They're good candidates for FPGA acceleration since
    the input tensor is large (3*16*224*224 = ~2.4M elements).
    """
    def __init__(self):
        super().__init__()

        # Spatial convolution (confusingly named conv_t by PyTorchVideo).
        # kernel_size=(1,3,3): 3x3 spatial filter per frame, no temporal mixing.
        # stride=(1,2,2): keep all 16 frames but halve spatial resolution 224->112.
        # padding=(0,1,1): pad 1 pixel on each side spatially for valid output size.
        # groups=1: standard (non-grouped) convolution. All 3 RGB channels are mixed
        #   together by each of the 24 output filters. Each filter sees the full RGB
        #   input and produces one output channel.
        self.conv_t = nn.Conv3d(
            3, 24,
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
            groups=1, bias=False,
        )

        # Temporal convolution (confusingly named conv_xy by PyTorchVideo).
        # kernel_size=(5,1,1): 5-frame temporal window, no spatial filtering.
        # stride=1: maintain all spatial and temporal dimensions.
        # padding=(2,0,0): pad 2 frames on each side of the temporal dimension
        #   so that output temporal size = input temporal size (16 frames in -> 16 out).
        # groups=24: depthwise -- each of the 24 channels from conv_t gets
        #   its own independent 5-tap temporal filter. No cross-channel mixing.
        self.conv_xy = nn.Conv3d(
            24, 24,
            kernel_size=(5, 1, 1), stride=1, padding=(2, 0, 0),
            groups=24, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 16, 224, 224] -- raw video clip
        Returns:
            [B, 24, 16, 112, 112]
        """
        x = self.conv_t(x)   # [B,3,16,224,224] -> [B,24,16,112,112] (spatial filter + downsample)
        x = self.conv_xy(x)  # [B,24,16,112,112] -> [B,24,16,112,112] (temporal filter, same shape)
        return x


class Stem(nn.Module):
    """
    Full stem block: Conv2plus1d + BatchNorm + ReLU.

    This wraps the factorized convolution with normalization and activation
    to produce the feature map that feeds into the first residual stage.
    """
    def __init__(self):
        super().__init__()
        self.conv = Conv2plus1dStem()
        self.norm = nn.BatchNorm3d(24, eps=1e-5, momentum=0.1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, 3, 16, 224, 224] -> [B, 24, 16, 112, 112]
        """
        return self.activation(self.norm(self.conv(x)))


# =============================================================================
# HEAD (Output classification)
# =============================================================================

class ProjectedPool(nn.Module):
    """
    Projected pooling: Conv5 expansion + global average pool + post-pool expansion.

    This sub-module is nested inside the Head as `self.pool` to match the
    pretrained checkpoint key names (blocks.5.pool.pre_conv, etc.).

    Architecture:
        1. pre_conv:  1x1x1 Conv3d (192 -> 432) -- expand channels before pooling
        2. pre_norm:  BatchNorm3d(432)
        3. pre_act:   ReLU
        4. pool:      AvgPool3d(16x7x7) -- collapse all spatial+temporal dimensions
        5. post_conv: 1x1x1 Conv3d (432 -> 2048) -- expand to classification width
        6. post_act:  ReLU

    For FPGA offloading: pre_conv is a 1x1x1 conv on a [B,192,16,7,7] tensor --
    moderately sized. post_conv is 1x1x1 on [B,432,1,1,1] -- very small.
    The AvgPool is trivial on CPU/RISC-V.
    """
    def __init__(self):
        super().__init__()

        # Conv5: channel expansion before pooling
        # 192 -> 432 channels via 1x1x1 pointwise convolution
        self.pre_conv = nn.Conv3d(192, 432, kernel_size=1, bias=False)
        self.pre_norm = nn.BatchNorm3d(432, eps=1e-5, momentum=0.1)
        self.pre_act = nn.ReLU(inplace=True)

        # Global spatio-temporal pooling
        # kernel_size=(16,7,7) matches the exact dimensions at this point.
        # After 4 stages of stride-2 downsampling: 224->112->56->28->14->7
        # Temporal dimension stays at 16 throughout (no temporal downsampling).
        self.pool = nn.AvgPool3d(kernel_size=(16, 7, 7))

        # Post-pool expansion: 432 -> 2048 channels
        # bias=False here to match the pretrained checkpoint (no bias key present).
        # This is effectively a learned projection to a high-dim representation.
        self.post_conv = nn.Conv3d(432, 2048, kernel_size=1, bias=False)
        self.post_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, 192, 16, 7, 7] -> [B, 2048, 1, 1, 1]
        """
        x = self.pre_act(self.pre_norm(self.pre_conv(x)))  # [B, 432, 16, 7, 7]
        x = self.pool(x)                                    # [B, 432, 1, 1, 1]
        x = self.post_act(self.post_conv(x))                # [B, 2048, 1, 1, 1]
        return x


class Head(nn.Module):
    """
    X3D classification head: converts spatial feature maps to class predictions.

    Architecture:
        1. pool (ProjectedPool):
           - pre_conv + BN + ReLU: expand 192 -> 432 channels
           - AvgPool3d: collapse [16,7,7] -> [1,1,1]
           - post_conv + ReLU: expand 432 -> 2048 channels
        2. dropout:    Dropout(p=0.5)
           - During training, randomly zeros 50% of activations to prevent
             overfitting. During eval (model.eval()), this is a no-op.
        3. proj:       Linear(2048 -> 400)
           - Final classification layer mapping to 400 Kinetics action classes.
        4. output_pool: AdaptiveAvgPool3d (for multi-crop inference averaging)

    The `pool` attribute wraps pre_conv/pre_norm/post_conv so that state_dict
    keys match the pretrained PyTorchVideo checkpoint (blocks.5.pool.pre_conv.*).
    """
    def __init__(self, num_classes: int = 400):
        super().__init__()

        # Pooling sub-module (contains conv5 + avgpool + post-projection)
        self.pool = ProjectedPool()

        # Classification layers
        self.dropout = nn.Dropout(p=0.5)
        # nn.Linear expects the features on the LAST dimension.
        # Since we'll permute from [B,2048,1,1,1] to [B,1,1,1,2048], Linear
        # operates on the last dim (2048) and maps to num_classes (400).
        self.proj = nn.Linear(2048, num_classes, bias=True)

        # For multi-crop evaluation: averages predictions across crops.
        # During single-crop inference this is a no-op (already [B,C,1,1,1]).
        self.output_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 192, 16, 7, 7] -- output of the last residual stage
        Returns:
            [B, 400] -- class logits (unnormalized log-probabilities)
        """
        # Pooling: conv5 + global avg pool + post-expansion
        x = self.pool(x)
        # x: [B, 2048, 1, 1, 1]

        # Dropout (active only during training)
        x = self.dropout(x)

        # Reshape for linear layer: [B, 2048, 1, 1, 1] -> [B, 1, 1, 1, 2048]
        # nn.Linear operates on the LAST dimension, so we permute channels to the end.
        x = x.permute(0, 2, 3, 4, 1)  # [B, 1, 1, 1, 2048]

        # Classification projection
        x = self.proj(x)               # [B, 1, 1, 1, 400]

        # Permute back to channels-first: [B, 400, 1, 1, 1]
        # This is needed so that output_pool operates on the spatial dims [1,1,1]
        # (a no-op for single-crop) rather than accidentally pooling the class dim.
        x = x.permute(0, 4, 1, 2, 3)  # [B, 400, 1, 1, 1]

        # Multi-crop averaging: pools spatial dims. For single-crop inference
        # the tensor is already [B, 400, 1, 1, 1] so this is a no-op.
        x = self.output_pool(x)        # [B, 400, 1, 1, 1]

        # Flatten to [B, 400]
        return x.view(x.shape[0], -1)


# =============================================================================
# FULL X3D-M MODEL
# =============================================================================

class X3D_M(nn.Module):
    """
    Complete X3D-M model: Stem + 4 Residual Stages + Head.

    The model is stored as a flat ModuleList called `blocks` (indices 0-5) to match
    the structure of the pretrained PyTorchVideo checkpoint, enabling direct weight
    loading via load_state_dict().

    blocks[0] = Stem          (Conv2plus1d -> BN -> ReLU)
    blocks[1] = Stage 2       (3 ResBlocks,  24ch,  stride=2: 112->56)
    blocks[2] = Stage 3       (5 ResBlocks,  48ch,  stride=2: 56->28)
    blocks[3] = Stage 4       (11 ResBlocks, 96ch,  stride=2: 28->14)
    blocks[4] = Stage 5       (7 ResBlocks,  192ch, stride=2: 14->7)
    blocks[5] = Head          (Conv5 + Pool + FC)

    Data flow with tensor shapes for batch_size=1:
        Input:       [1, 3,   16, 224, 224]   RGB video clip
        After Stem:  [1, 24,  16, 112, 112]   Initial features
        After S2:    [1, 24,  16, 56,  56]    Spatial /2
        After S3:    [1, 48,  16, 28,  28]    Spatial /2, channels x2
        After S4:    [1, 96,  16, 14,  14]    Spatial /2, channels x2
        After S5:    [1, 192, 16, 7,   7]     Spatial /2, channels x2
        Output:      [1, 400]                  Class logits

    Notice: temporal dimension stays at 16 throughout all stages. X3D-M does NOT
    do temporal downsampling -- it relies on the temporal receptive field growing
    through stacked 3x3x3 depthwise convolutions to capture motion patterns.
    """
    def __init__(self, num_classes: int = 400):
        super().__init__()

        self.blocks = nn.ModuleList([
            # blocks[0]: Stem
            Stem(),

            # blocks[1]: Stage 2 -- 3 residual blocks
            # in=24 (from stem), inner=54 (bottleneck expansion), out=24
            # First block has stride=2: 112x112 -> 56x56
            # Note: in_channels=24 matches stem output. The first block's skip
            # connection is just identity (24==24), but stride=2 requires a
            # branch1_conv to downsample the shortcut spatially.
            ResStage(depth=3, in_channels=24, inner_channels=54, out_channels=24),

            # blocks[2]: Stage 3 -- 5 residual blocks
            # in=24 (from stage 2), inner=108, out=48
            # First block: stride=2 (56x56 -> 28x28) + channel expansion (24->48)
            ResStage(depth=5, in_channels=24, inner_channels=108, out_channels=48),

            # blocks[3]: Stage 4 -- 11 residual blocks (the deepest stage)
            # in=48 (from stage 3), inner=216, out=96
            # First block: stride=2 (28x28 -> 14x14) + channel expansion (48->96)
            ResStage(depth=11, in_channels=48, inner_channels=216, out_channels=96),

            # blocks[4]: Stage 5 -- 7 residual blocks
            # in=96 (from stage 4), inner=432, out=192
            # First block: stride=2 (14x14 -> 7x7) + channel expansion (96->192)
            ResStage(depth=7, in_channels=96, inner_channels=432, out_channels=192),

            # blocks[5]: Head
            Head(num_classes=num_classes),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through the network.

        Args:
            x: [B, 3, 16, 224, 224] -- a batch of video clips
               B = batch size
               3 = RGB channels
               16 = number of frames (temporal dimension)
               224 = height and width (spatial dimensions)

        Returns:
            [B, 400] -- raw logits for each of the 400 Kinetics action classes.
                        Apply softmax to get probabilities:
                        probs = torch.softmax(logits, dim=1)
        """
        for block in self.blocks:
            x = block(x)
        return x


# =============================================================================
# WEIGHT LOADING FROM PRETRAINED PyTorchVideo MODEL
# =============================================================================

def load_pretrained_weights(model: X3D_M, device: str = "cpu") -> X3D_M:
    """
    Loads pretrained weights from the official PyTorchVideo X3D-M checkpoint
    into our custom model. The key names in our model are designed to match
    the PyTorchVideo naming convention exactly, so load_state_dict() works
    directly.

    torch.hub.load(): Downloads a model definition + weights from a GitHub repo.
    The first call downloads and caches; subsequent calls use the cache.

    state_dict(): Returns a dict mapping parameter names (strings) to tensors.
    For example: {"blocks.0.conv.conv_t.weight": tensor(...), ...}

    load_state_dict(strict=False): Loads parameters by matching key names.
    strict=False means it won't error on missing/extra keys -- useful when
    our architecture has slight naming differences (like 'pool.pool' vs 'pool').

    Args:
        model: Our X3D_M instance (randomly initialized)
        device: "cpu", "cuda", or "mps"

    Returns:
        The model with pretrained weights loaded
    """
    # Load the official pretrained model from PyTorchVideo
    pretrained = torch.hub.load(
        "facebookresearch/pytorchvideo", "x3d_m", pretrained=True
    )
    pretrained_sd = pretrained.state_dict()

    # Our model is designed so that state_dict key names match the pretrained
    # checkpoint exactly. This includes:
    #   - Stem:   blocks.0.conv.conv_t, blocks.0.conv.conv_xy, blocks.0.norm
    #   - Stages: blocks.N.res_blocks.M.branch2.conv_a/norm_a/conv_b/norm_b.0/norm_b.1.block.*
    #   - Head:   blocks.5.pool.pre_conv/pre_norm/post_conv, blocks.5.proj
    # So we can load directly without any key remapping.
    missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)

    if missing:
        print(f"  Missing keys ({len(missing)}):")
        for k in missing[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:10]:
            print(f"    {k}")

    return model.to(device)


# =============================================================================
# MAIN: Build, load weights, verify, and inspect
# =============================================================================

if __name__ == "__main__":
    device = "cpu"
    print(f"Device: {device}")
    print()

    # --- 1) Build our model from scratch ---
    print("Building X3D-M from custom layers...")
    model = X3D_M(num_classes=400)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print()

    # --- 2) Load pretrained weights ---
    print("Loading pretrained weights from PyTorchVideo...")
    model = load_pretrained_weights(model, device=device)
    model.eval()
    print()

    # --- 3) Verify against original model ---
    print("Loading original PyTorchVideo model for comparison...")
    original = torch.hub.load(
        "facebookresearch/pytorchvideo", "x3d_m", pretrained=True
    )
    original = original.eval().to(device)
    print()

    # Create a deterministic test input
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 16, 224, 224, device=device)

    with torch.no_grad():
        our_output = model(dummy_input)
        ref_output = original(dummy_input)

    print(f"Our model output shape:      {tuple(our_output.shape)}")
    print(f"Original model output shape:  {tuple(ref_output.shape)}")

    # Check if outputs match (they should be very close if weights loaded correctly)
    max_diff = (our_output - ref_output).abs().max().item()
    mean_diff = (our_output - ref_output).abs().mean().item()
    print(f"Max absolute difference:      {max_diff:.6e}")
    print(f"Mean absolute difference:     {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("PASS: Outputs match within tolerance!")
    else:
        print("WARN: Outputs differ -- check weight mapping.")

    # --- 4) Print layer-by-layer shapes ---
    print()
    print("=" * 70)
    print("LAYER-BY-LAYER SHAPE TRACE")
    print("=" * 70)

    torch.manual_seed(42)
    x = torch.randn(1, 3, 16, 224, 224, device=device)
    print(f"{'Input':<40s} {str(tuple(x.shape)):>30s}")

    with torch.no_grad():
        # Stem
        x = model.blocks[0](x)
        print(f"{'Stem (Conv2plus1d + BN + ReLU)':<40s} {str(tuple(x.shape)):>30s}")

        # Stages
        stage_names = ["Stage 2 (3 blocks)", "Stage 3 (5 blocks)",
                       "Stage 4 (11 blocks)", "Stage 5 (7 blocks)"]
        for i, name in enumerate(stage_names, start=1):
            x = model.blocks[i](x)
            print(f"{name:<40s} {str(tuple(x.shape)):>30s}")

        # Head
        x = model.blocks[5](x)
        print(f"{'Head (Conv5 + Pool + FC)':<40s} {str(tuple(x.shape)):>30s}")

    # --- 5) Top-5 predictions on dummy input (just to show the model works) ---
    print()
    print("Top-5 predictions (dummy random input, not meaningful):")
    probs = torch.softmax(our_output, dim=1)
    top5_probs, top5_indices = probs.topk(5, dim=1)
    for rank, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0]), 1):
        print(f"  {rank}. class {idx.item():3d}  probability {prob.item():.4f}")
