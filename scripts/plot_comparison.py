"""Compare ACT vs Diffusion Policy on PushT.

Parses the LeRobot training logs (which use tqdm ``\\r`` progress bars) and plots
success rate and coverage vs. training steps for the two policies side by side.

Usage:
    python scripts/plot_comparison.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# A training run may span multiple logs when it was resumed after a crash.
LOGS = {
    "ACT": ["train_act_pusht_200k.log"],
    "Diffusion Policy": [
        "train_diffusion_pusht_200k.log",
        "train_diffusion_pusht_resume.log",
    ],
}


def parse_eval_log(log_path: Path) -> list[tuple[int, float, float]]:
    """Return ``[(step, coverage, success_pct), ...]`` parsed from a training log.

    ``coverage`` is the pusht ``avg_max_reward`` metric and ``success_pct`` the
    percentage of successful rollouts (``pc_success``), both from the
    ``Suite overall aggregated`` line that LeRobot prints after each eval.
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    results: list[tuple[int, float, float]] = []
    step: int | None = None
    for line in text.splitlines():
        if m := re.search(r"Eval policy at step (\d+)", line):
            step = int(m.group(1))
        if m := re.search(
            r"Suite overall aggregated: \{.*?'avg_max_reward': ([\d.eE+-]+).*?'pc_success': ([\d.eE+-]+)",
            line,
        ):
            if step is not None:
                results.append((step, float(m.group(1)), float(m.group(2))))
                step = None
    return results


def load_results(policy: str) -> list[tuple[int, float, float]]:
    results: list[tuple[int, float, float]] = []
    for name in LOGS[policy]:
        results.extend(parse_eval_log(REPO_ROOT / name))
    results.sort()
    return results


def main() -> None:
    out_dir = REPO_ROOT / "assets"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    print("Final results:")
    for policy in LOGS:
        results = load_results(policy)
        steps = [r[0] for r in results]
        coverage = [r[1] for r in results]
        success = [r[2] for r in results]

        axes[0].plot(steps, success, marker="o", label=policy)
        axes[1].plot(steps, coverage, marker="o", label=policy)

        s, c, p = results[-1]
        print(f"  {policy:16s} step={s:6d}  success={p:.0f}%  coverage={c:.3f}")

    axes[0].set_xlabel("Training steps")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Success rate vs. training steps")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Training steps")
    axes[1].set_ylabel("Coverage (avg max reward)")
    axes[1].set_title("Coverage vs. training steps")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
