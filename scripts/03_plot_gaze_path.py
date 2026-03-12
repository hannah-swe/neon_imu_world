from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

from neon_imu.plot_config import setup_plot_style
from neon_imu.gaze_utils import (
    load_gaze_csv,
    add_relative_time,
    subset_gaze_by_time,
    filter_valid_gaze,
)

setup_plot_style()

# ----------------------------
# Configuration
# ----------------------------
RAW_ROOT = Path("data/raw")
OUTPUT_ROOT = Path("data/processed")
SUBJECT_GLOB = "sub-*"
GAZE_FILENAME = "gaze.csv"

SHOW_PLOTS = True
SAVE_PLOTS = False

# Zeitfenster pro Subject in Sekunden relativ zum Start der gaze.csv
TIME_WINDOWS_S = {
    "sub-997": (196.0, 210.0),
    "sub-998": (3.5, 15.5),
    "sub-999": (1.0, 13.0),
}
# ----------------------------

subject_dirs = sorted([p for p in RAW_ROOT.glob(SUBJECT_GLOB) if p.is_dir()])
print(f"Found {len(subject_dirs)} subject folder(s) under {RAW_ROOT}")

for subject_dir in subject_dirs:
    print(f"Processing {subject_dir.name}")

    gaze_path = subject_dir / GAZE_FILENAME
    if not gaze_path.exists():
        print(f"  SKIP: Gaze file not found: {gaze_path}")
        continue

    df = load_gaze_csv(gaze_path)
    df = add_relative_time(df)
    df = filter_valid_gaze(df, require_worn=True)

    t_start_s, t_end_s = TIME_WINDOWS_S[subject_dir.name]
    df_win = subset_gaze_by_time(df, t_start_s, t_end_s)

    print(f"  Samples in interval [{t_start_s}, {t_end_s}] s: {len(df_win)}")

    if len(df_win) == 0:
        print("  SKIP: No gaze samples in selected interval")
        continue

    x = df_win["gaze x [px]"].to_numpy()
    y = df_win["gaze y [px]"].to_numpy()
    t = df_win["time [s]"].to_numpy()

    if SHOW_PLOTS:
        fig, ax = plt.subplots()
        cmap = sns.color_palette("crest_r", as_cmap=True)
        norm = mcolors.Normalize(vmin=t.min(), vmax=t.max())
        # gaze path as connected line
        for i in range(len(x) - 1):
            ax.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                color=cmap(norm(t[i])),
                alpha=1,
                linewidth=2.5,
            )
        ax.set_xlabel("gaze x [px]")
        ax.set_ylabel("gaze y [px]")
        ax.set_title(f"{subject_dir.name} – Gaze path")
        # Pixel coordinates usually have origin top-left -> invert y for intuitive image-like view
        ax.invert_yaxis()
        # equal aspect is optional; for pixel coordinates it can be helpful
        ax.set_aspect("equal", adjustable="box")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label("time [s]")
        sns.despine(ax=ax)
        if SAVE_PLOTS:
            outdir = OUTPUT_ROOT / "plots" / subject_dir.name
            outdir.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir / "gaze_path.png", dpi=400, transparent=True)
        plt.show()