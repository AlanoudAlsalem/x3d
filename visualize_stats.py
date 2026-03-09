"""
Visualization script for X3D inference statistics.

Loads statistics from run_stats/ directory and generates:
- Comparison tables across platforms
- Latency breakdown charts
- Bottleneck analysis
- Layer-by-layer comparisons

Usage:
    python visualize_stats.py                    # Analyze all runs in run_stats/
    python visualize_stats.py --dir custom_dir   # Use custom directory
    python visualize_stats.py --output report    # Save charts to report/ directory
    python visualize_stats.py --format png       # Output format (png, svg, pdf)
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Charts will not be generated.")
    print("Install with: pip install matplotlib")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class RunData:
    """Parsed run statistics."""
    run_id: str
    timestamp: str
    platform: str
    device_type: str
    model_name: str
    input_shape: Tuple[int, ...]
    total_latency_ms: float
    total_params: int
    total_flops: int
    sections: Dict[str, Dict[str, Any]]
    layers: List[Dict[str, Any]]
    notes: str
    raw_data: Dict[str, Any]


def load_run(filepath: Path) -> Optional[RunData]:
    """Load a single run statistics file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        platform_info = data.get("platform_info", {})
        
        return RunData(
            run_id=data.get("run_id", "unknown"),
            timestamp=data.get("timestamp", ""),
            platform=platform_info.get("platform", "unknown"),
            device_type=platform_info.get("device_type", "unknown"),
            model_name=data.get("model_name", "X3D-M"),
            input_shape=tuple(data.get("input_shape", [])),
            total_latency_ms=data.get("total_latency_ms", 0),
            total_params=data.get("total_params", 0),
            total_flops=data.get("total_flops", 0),
            sections=data.get("sections", {}),
            layers=data.get("layers", []),
            notes=data.get("notes", ""),
            raw_data=data,
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_all_runs(stats_dir: Path) -> List[RunData]:
    """Load all run statistics from a directory."""
    runs = []
    
    if not stats_dir.exists():
        print(f"Directory not found: {stats_dir}")
        return runs
    
    for filepath in sorted(stats_dir.glob("*.json")):
        run = load_run(filepath)
        if run:
            runs.append(run)
    
    return runs


def print_comparison_table(runs: List[RunData]) -> None:
    """Print a comparison table of all runs."""
    if not runs:
        print("No runs to compare.")
        return
    
    print("\n" + "=" * 100)
    print("PLATFORM COMPARISON TABLE")
    print("=" * 100)
    
    header = f"{'Run ID':<20} {'Device Type':<25} {'Total Latency (ms)':>18} {'Params':>12} {'FLOPs':>15}"
    print(header)
    print("-" * 100)
    
    for run in runs:
        print(
            f"{run.run_id[:20]:<20} {run.device_type[:25]:<25} "
            f"{run.total_latency_ms:>18.2f} {run.total_params:>12,} {run.total_flops:>15,}"
        )
    
    print("-" * 100)
    
    if len(runs) > 1:
        baseline = runs[0]
        print(f"\nSpeedup relative to {baseline.device_type}:")
        for run in runs[1:]:
            if run.total_latency_ms > 0:
                speedup = baseline.total_latency_ms / run.total_latency_ms
                print(f"  {run.device_type}: {speedup:.2f}x")


def print_section_breakdown(runs: List[RunData]) -> None:
    """Print section-by-section breakdown for each run."""
    if not runs:
        return
    
    print("\n" + "=" * 100)
    print("SECTION BREAKDOWN BY PLATFORM")
    print("=" * 100)
    
    all_sections = set()
    for run in runs:
        all_sections.update(run.sections.keys())
    all_sections = sorted(all_sections)
    
    for section in all_sections:
        print(f"\n{section}:")
        print(f"  {'Device Type':<25} {'Latency (ms)':>15} {'% of Total':>12} {'Params':>12}")
        print(f"  {'-'*25} {'-'*15} {'-'*12} {'-'*12}")
        
        for run in runs:
            if section in run.sections:
                stats = run.sections[section]
                pct = (stats["latency_ms"] / run.total_latency_ms * 100) if run.total_latency_ms > 0 else 0
                print(
                    f"  {run.device_type[:25]:<25} {stats['latency_ms']:>15.2f} "
                    f"{pct:>11.1f}% {stats['params']:>12,}"
                )


def print_bottleneck_analysis(runs: List[RunData], top_n: int = 10) -> None:
    """Print bottleneck analysis showing slowest layers."""
    if not runs:
        return
    
    print("\n" + "=" * 100)
    print("BOTTLENECK ANALYSIS (Top Slowest Layers)")
    print("=" * 100)
    
    for run in runs:
        print(f"\n{run.device_type} ({run.run_id}):")
        print(f"  {'Rank':<5} {'Layer':<45} {'Latency (ms)':>12} {'% of Total':>10}")
        print(f"  {'-'*5} {'-'*45} {'-'*12} {'-'*10}")
        
        sorted_layers = sorted(run.layers, key=lambda x: x.get("latency_ms", 0), reverse=True)
        
        for i, layer in enumerate(sorted_layers[:top_n], 1):
            latency = layer.get("latency_ms", 0)
            pct = (latency / run.total_latency_ms * 100) if run.total_latency_ms > 0 else 0
            name = layer.get("name", "unknown")[:45]
            print(f"  {i:<5} {name:<45} {latency:>12.3f} {pct:>9.1f}%")


def print_layer_comparison(runs: List[RunData], layer_pattern: str = "") -> None:
    """Print layer-by-layer comparison across platforms."""
    if len(runs) < 2:
        print("\nNeed at least 2 runs for layer comparison.")
        return
    
    print("\n" + "=" * 100)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 100)
    
    layer_names = set()
    for run in runs:
        for layer in run.layers:
            name = layer.get("name", "")
            if layer_pattern.lower() in name.lower():
                layer_names.add(name)
    
    layer_names = sorted(layer_names)
    
    if not layer_names:
        print("No matching layers found.")
        return
    
    header = f"{'Layer':<40}"
    for run in runs:
        header += f" {run.device_type[:15]:>15}"
    print(header)
    print("-" * (40 + 16 * len(runs)))
    
    for layer_name in layer_names[:30]:
        row = f"{layer_name[:40]:<40}"
        for run in runs:
            latency = 0
            for layer in run.layers:
                if layer.get("name") == layer_name:
                    latency = layer.get("latency_ms", 0)
                    break
            row += f" {latency:>15.3f}"
        print(row)


def generate_latency_bar_chart(runs: List[RunData], output_path: Optional[Path] = None) -> None:
    """Generate bar chart comparing total latencies."""
    if not HAS_MATPLOTLIB or not runs:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    devices = [run.device_type for run in runs]
    latencies = [run.total_latency_ms for run in runs]
    
    colors = plt.cm.Set2(range(len(runs)))
    bars = ax.bar(devices, latencies, color=colors)
    
    ax.set_ylabel("Total Latency (ms)")
    ax.set_xlabel("Platform")
    ax.set_title("X3D-M Inference Latency Comparison")
    
    for bar, latency in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(latencies) * 0.01,
            f"{latency:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_section_breakdown_chart(runs: List[RunData], output_path: Optional[Path] = None) -> None:
    """Generate stacked bar chart showing section breakdown."""
    if not HAS_MATPLOTLIB or not runs:
        return
    
    all_sections = set()
    for run in runs:
        all_sections.update(run.sections.keys())
    all_sections = sorted(all_sections)
    
    if not all_sections:
        print("No section data available for chart.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    devices = [run.device_type for run in runs]
    x = range(len(devices))
    width = 0.6
    
    colors = plt.cm.tab10(range(len(all_sections)))
    
    bottom = [0] * len(runs)
    
    for section_idx, section in enumerate(all_sections):
        latencies = []
        for run in runs:
            if section in run.sections:
                latencies.append(run.sections[section]["latency_ms"])
            else:
                latencies.append(0)
        
        ax.bar(x, latencies, width, bottom=bottom, label=section, color=colors[section_idx])
        bottom = [b + l for b, l in zip(bottom, latencies)]
    
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Platform")
    ax.set_title("X3D-M Section Latency Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(devices, rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_section_pie_chart(run: RunData, output_path: Optional[Path] = None) -> None:
    """Generate pie chart showing section breakdown for a single run."""
    if not HAS_MATPLOTLIB or not run.sections:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sections = list(run.sections.keys())
    latencies = [run.sections[s]["latency_ms"] for s in sections]
    
    colors = plt.cm.Set3(range(len(sections)))
    
    def autopct_func(pct):
        return f"{pct:.1f}%" if pct > 3 else ""
    
    wedges, texts, autotexts = ax.pie(
        latencies,
        labels=sections,
        autopct=autopct_func,
        colors=colors,
        startangle=90,
    )
    
    ax.set_title(f"Latency Distribution: {run.device_type}\nTotal: {run.total_latency_ms:.1f} ms")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_bottleneck_chart(run: RunData, top_n: int = 15, output_path: Optional[Path] = None) -> None:
    """Generate horizontal bar chart showing top bottleneck layers."""
    if not HAS_MATPLOTLIB or not run.layers:
        return
    
    sorted_layers = sorted(run.layers, key=lambda x: x.get("latency_ms", 0), reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    names = [layer.get("name", "unknown")[-40:] for layer in sorted_layers]
    latencies = [layer.get("latency_ms", 0) for layer in sorted_layers]
    
    colors = plt.cm.Reds([(l / max(latencies)) * 0.7 + 0.3 for l in latencies])
    
    y_pos = range(len(names))
    bars = ax.barh(y_pos, latencies, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Top {top_n} Bottleneck Layers: {run.device_type}")
    
    for bar, latency in zip(bars, latencies):
        pct = (latency / run.total_latency_ms * 100) if run.total_latency_ms > 0 else 0
        ax.text(
            bar.get_width() + max(latencies) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{latency:.1f}ms ({pct:.1f}%)",
            va="center",
            fontsize=8,
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_layer_type_breakdown(run: RunData, output_path: Optional[Path] = None) -> None:
    """Generate chart showing latency by layer type."""
    if not HAS_MATPLOTLIB or not run.layers:
        return
    
    type_latencies: Dict[str, float] = {}
    for layer in run.layers:
        layer_type = layer.get("layer_type", "Unknown")
        latency = layer.get("latency_ms", 0)
        type_latencies[layer_type] = type_latencies.get(layer_type, 0) + latency
    
    sorted_types = sorted(type_latencies.items(), key=lambda x: x[1], reverse=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    types = [t[0] for t in sorted_types]
    latencies = [t[1] for t in sorted_types]
    
    colors = plt.cm.Set2(range(len(types)))
    bars = ax.bar(types, latencies, color=colors)
    
    ax.set_ylabel("Total Latency (ms)")
    ax.set_xlabel("Layer Type")
    ax.set_title(f"Latency by Layer Type: {run.device_type}")
    
    for bar, latency in zip(bars, latencies):
        pct = (latency / run.total_latency_ms * 100) if run.total_latency_ms > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(latencies) * 0.01,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_speedup_chart(runs: List[RunData], output_path: Optional[Path] = None) -> None:
    """Generate chart showing speedup relative to slowest platform."""
    if not HAS_MATPLOTLIB or len(runs) < 2:
        return
    
    slowest = max(runs, key=lambda r: r.total_latency_ms)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    devices = [run.device_type for run in runs]
    speedups = [slowest.total_latency_ms / run.total_latency_ms if run.total_latency_ms > 0 else 0 for run in runs]
    
    colors = ["green" if s > 1 else "red" for s in speedups]
    bars = ax.bar(devices, speedups, color=colors, alpha=0.7)
    
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1, label="Baseline (1x)")
    
    ax.set_ylabel("Speedup (x)")
    ax.set_xlabel("Platform")
    ax.set_title(f"Speedup Relative to {slowest.device_type}")
    
    for bar, speedup in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_charts(
    runs: List[RunData],
    output_dir: Path,
    fmt: str = "png",
) -> None:
    """Generate all visualization charts."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping chart generation.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating charts in {output_dir}/...")
    
    generate_latency_bar_chart(runs, output_dir / f"latency_comparison.{fmt}")
    
    generate_section_breakdown_chart(runs, output_dir / f"section_breakdown.{fmt}")
    
    if len(runs) >= 2:
        generate_speedup_chart(runs, output_dir / f"speedup_comparison.{fmt}")
    
    for run in runs:
        device_name = run.device_type.replace(" ", "_").lower()
        
        generate_section_pie_chart(run, output_dir / f"pie_{device_name}.{fmt}")
        generate_bottleneck_chart(run, output_path=output_dir / f"bottlenecks_{device_name}.{fmt}")
        generate_layer_type_breakdown(run, output_dir / f"layer_types_{device_name}.{fmt}")
    
    print(f"\nAll charts saved to {output_dir}/")


def generate_html_report(runs: List[RunData], output_path: Path) -> None:
    """Generate an HTML report with embedded data."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>X3D Inference Statistics Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        .metric { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .card { background: #f9f9f9; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
        .highlight { background-color: #fff3cd; }
        .timestamp { color: #888; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>X3D-M Inference Statistics Report</h1>
        <p class="timestamp">Generated: """ + str(runs[0].timestamp if runs else "N/A") + """</p>
        
        <h2>Platform Comparison</h2>
        <table>
            <tr>
                <th>Platform</th>
                <th>Total Latency (ms)</th>
                <th>Parameters</th>
                <th>FLOPs</th>
                <th>Notes</th>
            </tr>
"""
    
    for run in runs:
        html_content += f"""            <tr>
                <td>{run.device_type}</td>
                <td>{run.total_latency_ms:.2f}</td>
                <td>{run.total_params:,}</td>
                <td>{run.total_flops:,}</td>
                <td>{run.notes}</td>
            </tr>
"""
    
    html_content += """        </table>
        
        <h2>Section Breakdown</h2>
"""
    
    for run in runs:
        html_content += f"""        <h3>{run.device_type}</h3>
        <table>
            <tr>
                <th>Section</th>
                <th>Latency (ms)</th>
                <th>% of Total</th>
                <th>Parameters</th>
            </tr>
"""
        for section, stats in run.sections.items():
            pct = (stats["latency_ms"] / run.total_latency_ms * 100) if run.total_latency_ms > 0 else 0
            html_content += f"""            <tr>
                <td>{section}</td>
                <td>{stats["latency_ms"]:.2f}</td>
                <td>{pct:.1f}%</td>
                <td>{stats["params"]:,}</td>
            </tr>
"""
        html_content += """        </table>
"""
    
    html_content += """    </div>
</body>
</html>
"""
    
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"HTML report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize X3D inference statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_stats.py                        # Analyze all runs
    python visualize_stats.py --dir custom_stats     # Custom directory
    python visualize_stats.py --output charts        # Save charts to charts/
    python visualize_stats.py --no-charts            # Text output only
    python visualize_stats.py --html                 # Generate HTML report
        """,
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default="run_stats",
        help="Directory containing statistics JSON files (default: run_stats)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for charts (default: display interactively)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="png",
        choices=["png", "svg", "pdf"],
        help="Output format for charts (default: png)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation, text output only",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top bottleneck layers to show (default: 10)",
    )
    parser.add_argument(
        "--layer-filter",
        type=str,
        default="",
        help="Filter layers by name pattern for comparison",
    )
    
    args = parser.parse_args()
    
    stats_dir = Path(args.dir)
    runs = load_all_runs(stats_dir)
    
    if not runs:
        print(f"\nNo statistics files found in {stats_dir}/")
        print("Run 'python main.py --profile' first to generate statistics.")
        return
    
    print(f"\nLoaded {len(runs)} run(s) from {stats_dir}/")
    
    print_comparison_table(runs)
    print_section_breakdown(runs)
    print_bottleneck_analysis(runs, top_n=args.top_n)
    
    if len(runs) >= 2:
        print_layer_comparison(runs, layer_pattern=args.layer_filter)
    
    if not args.no_charts and HAS_MATPLOTLIB:
        if args.output:
            output_dir = Path(args.output)
            generate_all_charts(runs, output_dir, fmt=args.format)
        else:
            print("\nGenerating interactive charts...")
            generate_latency_bar_chart(runs)
            generate_section_breakdown_chart(runs)
            for run in runs:
                generate_bottleneck_chart(run, top_n=args.top_n)
    
    if args.html:
        html_path = Path(args.output or "run_stats") / "report.html"
        generate_html_report(runs, html_path)


if __name__ == "__main__":
    main()
