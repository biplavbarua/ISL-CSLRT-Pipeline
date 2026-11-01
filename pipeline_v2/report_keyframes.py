"""
pipeline_v2/report_keyframes.py

Keypoint / Keyframe Report Generator.

Reads the keyframe_report.csv produced by step01 and the gap_stats.json files
from step02 to produce a rich summary HTML + CSV report showing:
  - Per-signer detection rates for Pose, Left Hand, Right Hand
  - Gap statistics per clip
  - Overall corpus statistics table
  - Heatmap: detection rate by (sentence, signer) for each landmark group

Usage:
    python3 pipeline_v2/report_keyframes.py
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
import config

REPORT_CSV  = os.path.join(config.LOGS_DIR, "keyframe_report.csv")
REPORT_HTML = os.path.join(config.REPORTS_DIR, "keypoint_report.html")
HEATMAP_PNG = os.path.join(config.REPORTS_DIR, "detection_heatmap.png")


def load_report() -> list[dict]:
    if not os.path.exists(REPORT_CSV):
        raise FileNotFoundError(f"Run step01 first. Expected: {REPORT_CSV}")
    with open(REPORT_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_gap_stats() -> dict:
    """Returns {(slug, signer_id): gap_stats_dict}"""
    stats = {}
    norm_root = config.KP_NORM_DIR
    if not os.path.isdir(norm_root):
        return stats
    for slug in os.listdir(norm_root):
        slug_dir = os.path.join(norm_root, slug)
        if not os.path.isdir(slug_dir):
            continue
        for signer in os.listdir(slug_dir):
            gp = os.path.join(slug_dir, signer, "gap_stats.json")
            if os.path.exists(gp):
                with open(gp) as f:
                    stats[(slug, signer)] = json.load(f)
    return stats


def build_summary_stats(rows: list[dict], gap_stats: dict) -> dict:
    total = len(rows)
    ok    = sum(1 for r in rows if r.get("status") == "ok")
    cached = sum(1 for r in rows if r.get("status") == "cached")

    pose_rates  = [float(r["pose_det_%"])  for r in rows if r.get("pose_det_%",  "") not in ("", "-1")]
    lhand_rates = [float(r["lhand_det_%"]) for r in rows if r.get("lhand_det_%", "") not in ("", "-1")]
    rhand_rates = [float(r["rhand_det_%"]) for r in rows if r.get("rhand_det_%", "") not in ("", "-1")]

    long_gap_clips = sum(
        1 for gs in gap_stats.values() if gs.get("has_long_gap", False)
    )

    return {
        "total_clips":      total,
        "extracted":        ok,
        "cached":           cached,
        "mean_pose_det":    round(np.mean(pose_rates),  2) if pose_rates  else 0,
        "mean_lhand_det":   round(np.mean(lhand_rates), 2) if lhand_rates else 0,
        "mean_rhand_det":   round(np.mean(rhand_rates), 2) if rhand_rates else 0,
        "long_gap_clips":   long_gap_clips,
        "long_gap_pct":     round(long_gap_clips / max(total, 1) * 100, 1),
    }


def make_heatmap(rows: list[dict]):
    """
    Heatmap: rows=sentences (sorted), columns=signers (1–7)
    Three panels: Pose, L-Hand, R-Hand detection %.
    """
    sentences = sorted(set(r["slug"] for r in rows))
    signers   = sorted(set(int(r["signer_id"]) for r in rows))

    def build_matrix(key):
        mat = np.full((len(sentences), len(signers)), np.nan)
        for r in rows:
            si = sentences.index(r["slug"])
            sj = signers.index(int(r["signer_id"]))
            try:
                mat[si, sj] = float(r[key])
            except (ValueError, KeyError):
                pass
        return mat

    fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(sentences) * 0.18)))
    panels = [
        ("pose_det_%",  "Pose Detection %"),
        ("lhand_det_%", "Left Hand Detection %"),
        ("rhand_det_%", "Right Hand Detection %"),
    ]
    for ax, (key, title) in zip(axes, panels):
        mat = build_matrix(key)
        im  = ax.imshow(mat, aspect="auto", vmin=0, vmax=100,
                        cmap="RdYlGn", interpolation="nearest")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(signers)))
        ax.set_xticklabels([f"S{s}" for s in signers], fontsize=7)
        if ax is axes[0]:
            ax.set_yticks(range(len(sentences)))
            ax.set_yticklabels(sentences, fontsize=5)
        else:
            ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.02)

    plt.suptitle("MediaPipe Detection Rates by Sentence × Signer", fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(HEATMAP_PNG), exist_ok=True)
    plt.savefig(HEATMAP_PNG, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [HEATMAP] {HEATMAP_PNG}")


def write_html_report(rows: list[dict], summary: dict, gap_stats: dict):
    os.makedirs(os.path.dirname(REPORT_HTML), exist_ok=True)

    table_rows = ""
    for r in rows:
        key = (r["slug"], r["signer_id"])
        gs  = gap_stats.get(key, {})
        long_flag = "⚠" if gs.get("has_long_gap") else ""
        table_rows += (
            f"<tr>"
            f"<td>{r['sentence']}</td>"
            f"<td>{r['signer_id']}</td>"
            f"<td>{r.get('total_frames', '—')}</td>"
            f"<td>{r.get('fps', '—')}</td>"
            f"<td>{r.get('pose_det_%', '—')}%</td>"
            f"<td>{r.get('lhand_det_%', '—')}%</td>"
            f"<td>{r.get('rhand_det_%', '—')}%</td>"
            f"<td>{gs.get('n_gaps', '—')}</td>"
            f"<td>{gs.get('max_gap', '—')} {long_flag}</td>"
            f"<td>{r.get('status', '—')}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ISL-CSLRT Keypoint Report</title>
<style>
  body {{ font-family: monospace; font-size: 12px; margin: 20px; }}
  h1   {{ font-size: 18px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th,td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
  th    {{ background: #222; color: #fff; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .stat {{ display: inline-block; margin: 8px 16px; padding: 8px 16px;
           background: #eef; border-radius: 6px; }}
</style>
</head><body>
<h1>ISL-CSLRT v2 — Keypoint / Keyframe Report</h1>

<div>
  <span class="stat"><b>Total clips:</b> {summary["total_clips"]}</span>
  <span class="stat"><b>Pose det:</b> {summary["mean_pose_det"]}%</span>
  <span class="stat"><b>L-Hand det:</b> {summary["mean_lhand_det"]}%</span>
  <span class="stat"><b>R-Hand det:</b> {summary["mean_rhand_det"]}%</span>
  <span class="stat"><b>Long-gap clips:</b> {summary["long_gap_clips"]} ({summary["long_gap_pct"]}%)</span>
</div>

<p><img src="detection_heatmap.png" style="max-width:100%;"/></p>

<table>
<thead>
<tr>
  <th>Sentence</th><th>Signer</th><th>Frames</th><th>FPS</th>
  <th>Pose%</th><th>L-Hand%</th><th>R-Hand%</th>
  <th>Gaps</th><th>Max Gap</th><th>Status</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</body></html>"""

    with open(REPORT_HTML, "w") as f:
        f.write(html)
    print(f"  [HTML] {REPORT_HTML}")


def main():
    print("Generating keypoint/keyframe report...")
    rows      = load_report()
    gap_stats = load_gap_stats()
    summary   = build_summary_stats(rows, gap_stats)

    print(f"\n  Corpus summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    print("\n  Building detection heatmap...")
    make_heatmap(rows)

    print("  Writing HTML report...")
    write_html_report(rows, summary, gap_stats)

    print("\nReport complete.")


if __name__ == "__main__":
    main()

