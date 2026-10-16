from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False


CURVES = Path("results/phase3_fairness_curves.csv")
FIG_DIR = Path("results/figures")
PNG = FIG_DIR / "phase3_fairness_curves_publication.png"
PDF = FIG_DIR / "phase3_fairness_curves_publication.pdf"
SVG = FIG_DIR / "phase3_fairness_curves_publication.svg"
LOG = Path("reports/phase3_fairness_figure_publication_log.txt")

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def num(series):
    return pd.to_numeric(series, errors="coerce")


def safe_top(value, pad=1.18):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return 1.0
    return value * pad


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    if not CURVES.exists():
        raise SystemExit(f"Curves CSV not found: {CURVES.resolve()}")

    p("=" * 80)
    p("PHASE 3 PUBLICATION FAIRNESS FIGURE")
    p("=" * 80)
    p(f"INPUT: {CURVES.resolve()}")

    c = pd.read_csv(CURVES, low_memory=False)

    need = [
        "curve",
        "x_floor",
        "x_share",
        "young_floor",
        "girls_floor",
        "obj_unweighted",
        "obj_drop_vs_raw",
        "observed_oos_captured",
        "capture_drop_vs_raw",
        "girls_share",
    ]

    missing = [k for k in need if k not in c.columns]
    p(f"rows={len(c)}; missing columns: {missing if missing else 'none'}")

    if missing:
        raise SystemExit(f"Curves CSV missing columns: {missing}")

    for k in need[1:]:
        c[k] = num(c[k])

    young = c[c["curve"] == "young_floor_at_girls_0"].sort_values("x_floor")
    g0 = c[c["curve"] == "girls_floor_at_young_0"].sort_values("girls_floor")
    g423 = c[c["curve"] == "girls_floor_at_young_423"].sort_values("girls_floor")

    if len(young) == 0 or len(g0) == 0 or len(g423) == 0:
        raise SystemExit("One of the curve groups is empty.")

    obj_raw_y0 = float(young.loc[young["x_floor"] == 0, "obj_unweighted"].iloc[0])
    cap_raw_y0 = int(young.loc[young["x_floor"] == 0, "observed_oos_captured"].iloc[0])

    obj_raw_y423 = float(young.loc[young["x_floor"] == 423, "obj_unweighted"].iloc[0])
    cap_raw_y423 = int(young.loc[young["x_floor"] == 423, "observed_oos_captured"].iloc[0])

    p("")
    p("BASELINES")
    p(f"  y=0:   obj={obj_raw_y0:.6f}, captured OOS={cap_raw_y0}")
    p(f"  y=423: obj={obj_raw_y423:.6f}, captured OOS={cap_raw_y423}")

    checks = [
        (
            "g0 first obj links to y=0 baseline",
            abs(float(g0["obj_unweighted"].iloc[0]) - obj_raw_y0) <= 1e-6,
        ),
        (
            "g0 first capture links to y=0 baseline",
            int(g0["observed_oos_captured"].iloc[0]) == cap_raw_y0,
        ),
        (
            "g423 first obj links to y=423 baseline",
            abs(float(g423["obj_unweighted"].iloc[0]) - obj_raw_y423) <= 1e-6,
        ),
        (
            "g423 first capture links to y=423 baseline",
            int(g423["observed_oos_captured"].iloc[0]) == cap_raw_y423,
        ),
    ]

    p("")
    p("CROSS-LINK CHECKS")
    for name, ok in checks:
        p(f"  {name}: {'OK' if ok else 'FAIL'}")

    if not all(ok for _, ok in checks):
        raise SystemExit("Cross-link failed. Do not use the figure.")

    young_x = young["x_share"].to_numpy()
    young_obj_drop = obj_raw_y0 - young["obj_unweighted"].to_numpy()
    young_cap_drop = cap_raw_y0 - young["observed_oos_captured"].to_numpy()

    g0_x = g0["girls_share"].to_numpy()
    g0_obj_drop = obj_raw_y0 - g0["obj_unweighted"].to_numpy()
    g0_cap_drop = cap_raw_y0 - g0["observed_oos_captured"].to_numpy()

    g423_x = g423["girls_share"].to_numpy()
    g423_obj_drop = obj_raw_y423 - g423["obj_unweighted"].to_numpy()
    g423_cap_drop = cap_raw_y423 - g423["observed_oos_captured"].to_numpy()

    p("")
    p("OBJECTIVE MONOTONICITY CHECKS")

    mono_fail = False

    for name, arr in [
        ("young objective drop", young_obj_drop),
        ("girls objective drop at y=0", g0_obj_drop),
        ("girls objective drop at y=423", g423_obj_drop),
    ]:
        bad = int((np.diff(arr) < -1e-9).sum())
        p(f"  {name}: {'PASS' if bad == 0 else f'FAIL ({bad} decreases)'}")
        if bad > 0:
            mono_fail = True

    if mono_fail:
        raise SystemExit("Objective drop is not monotone. Stop before plotting.")

    p("")
    p("CAPTURE WOBBLE DIAGNOSTICS")

    for name, arr in [
        ("young capture drop", young_cap_drop),
        ("girls capture drop at y=0", g0_cap_drop),
        ("girls capture drop at y=423", g423_cap_drop),
    ]:
        bad = int((np.diff(arr) < -1e-9).sum())
        p(
            f"  {name}: {bad} decrease(s) -> "
            f"{'smooth' if bad == 0 else 'non-monotone; evaluation metric only'}"
        )

    if HAVE_SEABORN:
        sns.set_context("paper", font_scale=1.05)
        p("")
        p("Seaborn context applied: paper.")
    else:
        p("")
        p("Seaborn not installed; using matplotlib rcParams only.")

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    GRAY = "#555555"

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)

    # Panel A: young floor, objective cost.
    ax = axs[0, 0]
    ax.plot(young_x, young_obj_drop, marker="o", color=BLUE)

    prop_idx = int(np.argmin(np.abs(young_x - 0.579452)))

    ax.annotate(
        (
            f"proportional young floor\n"
            f"{young_x[prop_idx] * 100:.1f}%, cost {young_obj_drop[prop_idx]:.1f}"
        ),
        xy=(young_x[prop_idx], young_obj_drop[prop_idx]),
        xytext=(0.04, 0.78),
        textcoords="axes fraction",
        fontsize=7,
        color=GRAY,
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=GRAY,
            linewidth=0.8,
            connectionstyle="arc3,rad=0.12",
        ),
    )

    ax.set_xlabel("Selected share aged < 13")
    ax.set_ylabel("Objective drop vs raw")
    ax.set_title("A. Cost of age coverage")
    ax.set_ylim(0, safe_top(np.nanmax(young_obj_drop)))

    # Panel B: young floor, observed capture cost.
    ax = axs[0, 1]
    ax.plot(young_x, young_cap_drop, marker="s", color=ORANGE)

    ax.annotate(
        (
            f"proportional young floor\n"
            f"loss {young_cap_drop[prop_idx]:.0f} observed OOS"
        ),
        xy=(young_x[prop_idx], young_cap_drop[prop_idx]),
        xytext=(0.04, 0.78),
        textcoords="axes fraction",
        fontsize=7,
        color=GRAY,
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=GRAY,
            linewidth=0.8,
            connectionstyle="arc3,rad=0.12",
        ),
    )

    ax.set_xlabel("Selected share aged < 13")
    ax.set_ylabel("Observed OOS not captured")
    ax.set_title("B. Reach cost of age coverage")
    ax.set_ylim(0, safe_top(np.nanmax(young_cap_drop)))

    # Panel C: girls floor, incremental objective cost.
    ax = axs[1, 0]

    ln0_c, = ax.plot(g0_x, g0_obj_drop, marker="o", color=BLUE)
    ln423_c, = ax.plot(
        g423_x,
        g423_obj_drop,
        marker="^",
        linestyle="--",
        color=GREEN,
    )

    ax.axvline(0.374415, color=GRAY, linestyle=":", linewidth=0.8)
    ax.axvline(0.502740, color=GRAY, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Selected share girls")
    ax.set_ylabel("Incremental objective drop")
    ax.set_title("C. Cost of gender coverage")

    ax.legend(
        handles=[ln0_c, ln423_c],
        labels=["No young floor", "Proportional young floor"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=7.0,
    )

    ymax_c = max(np.nanmax(g0_obj_drop), np.nanmax(g423_obj_drop))
    ax.set_ylim(0, safe_top(ymax_c))

    # Panel D: girls floor, incremental observed capture cost.
    ax = axs[1, 1]

    ln0_d, = ax.plot(g0_x, g0_cap_drop, marker="o", color=BLUE)
    ln423_d, = ax.plot(
        g423_x,
        g423_cap_drop,
        marker="^",
        linestyle="--",
        color=GREEN,
    )

    ax.axvline(0.374415, color=GRAY, linestyle=":", linewidth=0.8)
    ax.axvline(0.502740, color=GRAY, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Selected share girls")
    ax.set_ylabel("Incremental observed OOS not captured")
    ax.set_title("D. Reach cost of gender coverage")

    ax.legend(
        handles=[ln0_d, ln423_d],
        labels=["No young floor", "Proportional young floor"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=7.0,
    )

    ymax_d = max(np.nanmax(g0_cap_drop), np.nanmax(g423_cap_drop))
    ax.set_ylim(0, safe_top(ymax_d))

    fig.suptitle(
        "Price of fairness constraints, rural Chattogram ages 6-17",
        fontsize=11,
        fontweight="bold",
    )

    fig.savefig(PNG, dpi=600, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(SVG, bbox_inches="tight")
    plt.close(fig)

    p("")
    p("SAVED FILES")
    for f in [PNG, PDF, SVG]:
        p(f"  {f.resolve()} ({os.path.getsize(f)} bytes)")

    p("")
    p("NEXT: open the PNG locally and check the two red-marked problems.")
    p("The gray vertical-line explanation is intentionally not inside the PNG.")
    p("It will be placed as a caption under the image on the website/report.")

    LOG.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"LOG SAVED: {LOG.resolve()}")


if __name__ == "__main__":
    main()