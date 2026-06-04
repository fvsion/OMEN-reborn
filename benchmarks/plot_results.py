#!/usr/bin/env python3
"""Render benchmark ``results.json`` into publication-quality PNGs.

Produces two views of cumulative held-out crack rate vs. number of guesses:

* ``crackrate_vs_guesses.png``  — log-x, linear-y (the headline chart)
* ``crackrate_loglog.png``      — log-x, log-y (shows early-guess behaviour)

matplotlib is a benchmark-only dependency; the ``omen`` runtime stays
stdlib-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results.json"
DEFAULT_OUT = HERE.parent / "docs" / "img"

# Stable, readable colours; OMEN curves warm, competitors cool/grey.
# Keys are matched by longest-prefix first, so specific labels win over generic.
STYLE = {
    "OMEN n=4": {"color": "#d62728", "lw": 2.4, "zorder": 5},
    "OMEN n=3": {"color": "#ff7f0e", "lw": 2.2, "zorder": 4},
    "PRINCE (train words)": {"color": "#1f77b4", "lw": 2.0, "zorder": 3},
    "PRINCE (top-100k)": {"color": "#17becf", "lw": 2.0, "ls": ":", "zorder": 3},
    "PRINCE": {"color": "#1f77b4", "lw": 2.0, "zorder": 3},
    "rockyou wordlist": {"color": "#7f7f7f", "lw": 1.8, "ls": "--", "zorder": 2},
}


def _style(label: str) -> dict[str, object]:
    for key in sorted(STYLE, key=len, reverse=True):
        if label.startswith(key):
            return dict(STYLE[key])
    return {"lw": 2.0}


def _plot(results: dict, *, loglog: bool, out_path: Path) -> None:
    meta = results["meta"]
    total = meta["test_unique"]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)

    # Order legend by final crack rate, best first.
    methods = sorted(
        results["methods"].items(),
        key=lambda kv: kv[1].get("final_crackrate", 0.0),
        reverse=True,
    )
    for label, data in methods:
        xs = data["guesses"]
        ys = [c / total * 100.0 for c in data["cracked"]]
        ax.plot(xs, ys, label=f"{label}  ({data.get('final_crackrate', 0):.1%})", **_style(label))

    ax.set_xscale("log")
    if loglog:
        ax.set_yscale("log")
        ax.set_ylim(0.01, 100)
    else:
        ax.set_ylim(0, None)
    ax.set_xlabel("Guesses (log scale)")
    ax.set_ylabel("Held-out passwords cracked (%)")
    title = "Password guessing efficiency on rockyou (disjoint train/test)"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="method (final crack rate)", loc="upper left", framealpha=0.9)
    fig.text(
        0.5,
        0.005,
        f"train: {meta.get('split', '')}  |  held-out unique: {total:,}",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    out_dir = Path(args.out_dir)
    _plot(results, loglog=False, out_path=out_dir / "crackrate_vs_guesses.png")
    _plot(results, loglog=True, out_path=out_dir / "crackrate_loglog.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
