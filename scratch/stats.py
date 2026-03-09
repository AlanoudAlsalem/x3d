"""
Statistics collector for X3D inference profiling.

Collects timing, parameter counts, compute estimates (FLOPs), memory usage,
and tensor shapes for each layer. Exports to JSON for later analysis.
"""

from __future__ import annotations
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def get_platform_info() -> Dict[str, Any]:
    """Collect platform/system information."""
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "node": platform.node(),
    }
    
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        info["total_memory_gb"] = round(mem.total / (1024**3), 2)
        info["available_memory_gb"] = round(mem.available / (1024**3), 2)
    except ImportError:
        info["cpu_count_logical"] = os.cpu_count()
        info["psutil_available"] = False
    
    node_lower = platform.node().lower()
    machine_lower = platform.machine().lower()
    system_lower = platform.system().lower()
    
    if "polarfire" in node_lower or "icicle" in node_lower or "mpfs" in node_lower:
        info["device_type"] = "PolarFire SoC Icicle Kit"
    elif "riscv" in machine_lower or "rv64" in machine_lower:
        info["device_type"] = "RISC-V Linux"
    elif "darwin" in system_lower:
        info["device_type"] = "macOS"
    elif "linux" in system_lower:
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()
                if "sifive" in cpuinfo or "riscv" in cpuinfo or "polarfire" in cpuinfo:
                    info["device_type"] = "PolarFire SoC Icicle Kit"
                else:
                    info["device_type"] = "Linux"
        except (FileNotFoundError, PermissionError):
            info["device_type"] = "Linux"
    else:
        info["device_type"] = platform.system()
    
    return info


@dataclass
class LayerStats:
    """Statistics for a single layer/operation."""
    name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    latency_ms: float = 0.0
    params: int = 0
    flops: int = 0
    memory_bytes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        d["output_shape"] = list(self.output_shape)
        return d


@dataclass
class RunStats:
    """Complete statistics for a single inference run."""
    run_id: str
    timestamp: str
    platform_info: Dict[str, Any]
    model_name: str
    input_shape: Tuple[int, ...]
    total_latency_ms: float = 0.0
    total_params: int = 0
    total_flops: int = 0
    layers: List[LayerStats] = field(default_factory=list)
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "platform_info": self.platform_info,
            "model_name": self.model_name,
            "input_shape": list(self.input_shape),
            "total_latency_ms": self.total_latency_ms,
            "total_params": self.total_params,
            "total_flops": self.total_flops,
            "layers": [layer.to_dict() for layer in self.layers],
            "sections": self.sections,
            "notes": self.notes,
        }
        return d


class StatsCollector:
    """
    Collects and manages inference statistics.
    
    Usage:
        collector = StatsCollector("X3D-M")
        collector.start_run((1, 3, 16, 224, 224))
        
        with collector.time_layer("stem.conv_t", "Conv3d", input_shape, output_shape):
            output = conv_t.forward(input)
        
        collector.end_run()
        collector.save("run_stats/")
    """
    
    def __init__(self, model_name: str = "X3D-M", verbose_log: bool = False):
        self.model_name = model_name
        self.verbose_log = verbose_log
        self.current_run: Optional[RunStats] = None
        self._section_stack: List[str] = []
        self._section_start_times: Dict[str, float] = {}
        
    def start_run(self, input_shape: Tuple[int, ...], notes: str = "") -> None:
        """Start a new profiling run."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        timestamp = datetime.now().isoformat()
        platform_info = get_platform_info()
        
        self.current_run = RunStats(
            run_id=run_id,
            timestamp=timestamp,
            platform_info=platform_info,
            model_name=self.model_name,
            input_shape=input_shape,
            notes=notes,
        )
        self._run_start_time = time.perf_counter()
        
    def end_run(self) -> None:
        """End the current profiling run and compute totals."""
        if self.current_run is None:
            return
            
        self.current_run.total_latency_ms = (time.perf_counter() - self._run_start_time) * 1000
        self.current_run.total_params = sum(layer.params for layer in self.current_run.layers)
        self.current_run.total_flops = sum(layer.flops for layer in self.current_run.layers)
        
    def start_section(self, name: str) -> None:
        """Start timing a section (e.g., 'Stem', 'Stage2')."""
        self._section_stack.append(name)
        self._section_start_times[name] = time.perf_counter()
        
    def end_section(self, name: str) -> None:
        """End timing a section and record stats."""
        if self.current_run is None or name not in self._section_start_times:
            return
            
        elapsed_ms = (time.perf_counter() - self._section_start_times[name]) * 1000
        
        section_layers = [
            layer for layer in self.current_run.layers 
            if layer.name.startswith(name)
        ]
        section_params = sum(layer.params for layer in section_layers)
        section_flops = sum(layer.flops for layer in section_layers)
        
        self.current_run.sections[name] = {
            "latency_ms": elapsed_ms,
            "params": section_params,
            "flops": section_flops,
            "num_layers": len(section_layers),
        }
        
        if name in self._section_stack:
            self._section_stack.remove(name)
        del self._section_start_times[name]
        
    def add_layer(
        self,
        name: str,
        layer_type: str,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        latency_ms: float,
        params: int = 0,
        flops: int = 0,
        memory_bytes: int = 0,
        **extra: Any,
    ) -> None:
        """Add statistics for a layer."""
        if self.current_run is None:
            return
            
        layer_stats = LayerStats(
            name=name,
            layer_type=layer_type,
            input_shape=input_shape,
            output_shape=output_shape,
            latency_ms=latency_ms,
            params=params,
            flops=flops,
            memory_bytes=memory_bytes,
            extra=extra,
        )
        self.current_run.layers.append(layer_stats)
        
        if self.verbose_log:
            out_str = str(output_shape)
            if len(out_str) > 40:
                out_str = out_str[:37] + "..."
            print(f"    {name}: {layer_type} | {latency_ms:.3f} ms | out={out_str} | params={params:,} | flops={flops:,}")
        
    def time_layer(
        self,
        name: str,
        layer_type: str,
        input_shape: Tuple[int, ...],
        params: int = 0,
        flops: int = 0,
        memory_bytes: int = 0,
        **extra: Any,
    ) -> "LayerTimer":
        """Context manager for timing a layer."""
        return LayerTimer(self, name, layer_type, input_shape, params, flops, memory_bytes, extra)
        
    def save(self, output_dir: Union[str, Path], filename: Optional[str] = None) -> str:
        """Save run statistics to a JSON file."""
        if self.current_run is None:
            raise ValueError("No run to save. Call start_run() first.")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            device_type = self.current_run.platform_info.get("device_type", "unknown")
            device_type = device_type.replace(" ", "_").lower()
            filename = f"{self.current_run.run_id}_{device_type}.json"
            
        filepath = output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.current_run.to_dict(), f, indent=2)
            
        return str(filepath)
    
    def save_text_report(self, output_dir: Union[str, Path], filename: Optional[str] = None) -> str:
        """Save a human-readable text report."""
        if self.current_run is None:
            raise ValueError("No run to save. Call start_run() first.")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            device_type = self.current_run.platform_info.get("device_type", "unknown")
            device_type = device_type.replace(" ", "_").lower()
            filename = f"{self.current_run.run_id}_{device_type}.txt"
            
        filepath = output_dir / filename
        
        lines = self._generate_text_report()
        
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
            
        return str(filepath)
    
    def _generate_text_report(self) -> List[str]:
        """Generate text report lines."""
        run = self.current_run
        if run is None:
            return ["No run data available."]
            
        lines = []
        lines.append("=" * 80)
        lines.append(f"X3D INFERENCE STATISTICS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append("PLATFORM INFORMATION")
        lines.append("-" * 40)
        for key, value in run.platform_info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        lines.append("RUN SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Run ID: {run.run_id}")
        lines.append(f"  Timestamp: {run.timestamp}")
        lines.append(f"  Model: {run.model_name}")
        lines.append(f"  Input Shape: {run.input_shape}")
        lines.append(f"  Total Latency: {run.total_latency_ms:.2f} ms")
        lines.append(f"  Total Parameters: {run.total_params:,}")
        lines.append(f"  Total FLOPs: {run.total_flops:,}")
        if run.notes:
            lines.append(f"  Notes: {run.notes}")
        lines.append("")
        
        if run.sections:
            lines.append("SECTION BREAKDOWN")
            lines.append("-" * 40)
            lines.append(f"  {'Section':<20} {'Latency (ms)':>12} {'Params':>12} {'FLOPs':>15} {'% Time':>8}")
            lines.append(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*15} {'-'*8}")
            for name, stats in run.sections.items():
                pct = (stats["latency_ms"] / run.total_latency_ms * 100) if run.total_latency_ms > 0 else 0
                lines.append(
                    f"  {name:<20} {stats['latency_ms']:>12.2f} {stats['params']:>12,} "
                    f"{stats['flops']:>15,} {pct:>7.1f}%"
                )
            lines.append("")
        
        lines.append("LAYER-BY-LAYER BREAKDOWN")
        lines.append("-" * 40)
        lines.append(f"  {'Layer':<40} {'Type':<15} {'Latency (ms)':>12} {'Params':>10} {'Output Shape':<20}")
        lines.append(f"  {'-'*40} {'-'*15} {'-'*12} {'-'*10} {'-'*20}")
        
        for layer in run.layers:
            output_str = str(layer.output_shape)
            if len(output_str) > 20:
                output_str = output_str[:17] + "..."
            lines.append(
                f"  {layer.name:<40} {layer.layer_type:<15} {layer.latency_ms:>12.3f} "
                f"{layer.params:>10,} {output_str:<20}"
            )
        lines.append("")
        
        top_layers = sorted(run.layers, key=lambda x: x.latency_ms, reverse=True)[:10]
        lines.append("TOP 10 SLOWEST LAYERS")
        lines.append("-" * 40)
        for i, layer in enumerate(top_layers, 1):
            pct = (layer.latency_ms / run.total_latency_ms * 100) if run.total_latency_ms > 0 else 0
            lines.append(f"  {i:2}. {layer.name:<40} {layer.latency_ms:>10.3f} ms ({pct:>5.1f}%)")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return lines
    
    def print_summary(self) -> None:
        """Print a summary to stdout."""
        lines = self._generate_text_report()
        print("\n".join(lines))


class LayerTimer:
    """Context manager for timing a layer execution."""
    
    def __init__(
        self,
        collector: StatsCollector,
        name: str,
        layer_type: str,
        input_shape: Tuple[int, ...],
        params: int,
        flops: int,
        memory_bytes: int,
        extra: Dict[str, Any],
    ):
        self.collector = collector
        self.name = name
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.params = params
        self.flops = flops
        self.memory_bytes = memory_bytes
        self.extra = extra
        self.output_shape: Optional[Tuple[int, ...]] = None
        
    def __enter__(self) -> "LayerTimer":
        self._start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        output_shape = self.output_shape if self.output_shape else self.input_shape
        self.collector.add_layer(
            name=self.name,
            layer_type=self.layer_type,
            input_shape=self.input_shape,
            output_shape=output_shape,
            latency_ms=elapsed_ms,
            params=self.params,
            flops=self.flops,
            memory_bytes=self.memory_bytes,
            **self.extra,
        )
        
    def set_output_shape(self, shape: Tuple[int, ...]) -> None:
        """Set the output shape after computation."""
        self.output_shape = shape


def estimate_conv3d_flops(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    kernel_size: Tuple[int, int, int],
    groups: int = 1,
) -> int:
    """
    Estimate FLOPs for a 3D convolution.
    FLOPs = 2 * output_elements * kernel_elements * (in_channels / groups)
    """
    B, C_out, T_out, H_out, W_out = output_shape
    B, C_in, T_in, H_in, W_in = input_shape
    kT, kH, kW = kernel_size
    
    output_elements = B * C_out * T_out * H_out * W_out
    kernel_elements = kT * kH * kW
    channels_per_group = C_in // groups
    
    return 2 * output_elements * kernel_elements * channels_per_group


def estimate_linear_flops(in_features: int, out_features: int, batch_size: int = 1) -> int:
    """Estimate FLOPs for a linear layer. FLOPs = 2 * batch * in * out."""
    return 2 * batch_size * in_features * out_features


def count_parameters(weight: np.ndarray, bias: Optional[np.ndarray] = None) -> int:
    """Count parameters in weight and optional bias."""
    total = weight.size
    if bias is not None:
        total += bias.size
    return total
