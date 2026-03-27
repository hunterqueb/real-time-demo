import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


COLORS = {
    "true":  "#2ecc71",
    "mamba": "#e74c3c",
    "lstm":  "#3498db",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", type=str, default="lorenz",
        help="system to visualize: lorenz | 2bp | 3bp",
    )
    parser.add_argument(
        "--trail", type=int, default=300,
        help="number of past frames shown as fading trail (default: 300)",
    )
    parser.add_argument(
        "--frames", type=int, default=500,
        help="target number of animation frames (data is downsampled to this, default: 500)",
    )
    parser.add_argument(
        "--interval", type=int, default=50,
        help="milliseconds between frames (default: 20)",
    )
    parser.add_argument(
        "--save", action='store_true',
        help="save animation to file, e.g. out.mp4 (requires ffmpeg)",
    )

    return parser.parse_args()


def load_data(system):
    paths = {
        "lorenz": "scripts/data/lorenz.npz",
        "2bp":    "scripts/data/2bp.npz",
        "3bp":    "scripts/data/CR3BPRetrograde.npz",
    }
    if system not in paths:
        raise ValueError(f"Unknown system '{system}'. Choose from: lorenz, 2bp, 3bp")
    return np.load(paths[system], allow_pickle=True)


def extract(data, system):
    """Return (true, mamba, lstm) position arrays and plot dimensionality."""
    true  = data["trueTraj"]
    mamba = data["networkPredictionMamba"]
    lstm  = data["networkPredictionLSTM"]
    t     = data["t"].ravel()
    d_units = data["d_units"]
    t_units = data["t_units"]

    print(type(str(d_units)))

    if system == "2bp":
        # state = [x, y, z, vx, vy, vz] — plot position only
        return true[:, :3], mamba[:, :3], lstm[:, :3], t, d_units, t_units, "3d"
    elif system == "3bp":
        # state = [x, y, xdot, ydot] — planar orbit
        return true[:, :2], mamba[:, :2], lstm[:, :2], t, d_units, t_units, "2d"
    else:  # lorenz: already (N, 3)
        return true, mamba, lstm, t, d_units, t_units, "3d"



def animate_3d(true_pos, mamba_pos, lstm_pos, t, system, trail, n_frames, interval, save,d_units,t_units):
    fig = plt.figure(figsize=(10, 8))
    ax      = fig.add_axes([0.05, 0.08, 0.90, 0.88], projection="3d")
    prog_ax = fig.add_axes([0.05, 0.02, 0.90, 0.03])
    prog_ax.set_xlim(0, 1); prog_ax.set_ylim(0, 1)
    prog_ax.set_xticks([]); prog_ax.set_yticks([])
    prog_bar = prog_ax.barh(0.5, 0, height=1, color=COLORS["true"], align="center")[0]

    # Fixed limits from all trajectories
    all_pos = np.concatenate([true_pos, mamba_pos, lstm_pos], axis=0)
    pad = 0.05
    for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        col = all_pos[:, dim]
        col = col[np.isfinite(col)]
        lo_v, hi_v = col.min(), col.max()
        margin = (hi_v - lo_v) * pad
        setter(lo_v - margin, hi_v + margin)

    n    = len(true_pos)
    step = max(1, n // n_frames)
    idx  = np.arange(0, n, step)

    # Full true trajectory ghost (static)
    ax.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2],
            color='k', lw=0.6, alpha=0.2, ls="--", zorder=1)

    if system == "2bp":
        ax.scatter(0,0,0,
                color = 'k',label = "Earth")

    # Trail lines
    line_true,  = ax.plot([], [], [], color=COLORS["true"],  lw=1.2, alpha=0.9, label="True",  zorder=3)
    line_mamba, = ax.plot([], [], [], color=COLORS["mamba"], lw=1.0, alpha=0.8, label="Mamba", zorder=3)
    line_lstm,  = ax.plot([], [], [], color=COLORS["lstm"],  lw=1.0, alpha=0.8, label="LSTM",  zorder=3)
    # Current-position dots
    dot_true,  = ax.plot([], [], [], "o", color=COLORS["true"],  ms=6, zorder=5)
    dot_mamba, = ax.plot([], [], [], "o", color=COLORS["mamba"], ms=5, zorder=5)
    dot_lstm,  = ax.plot([], [], [], "o", color=COLORS["lstm"],  ms=5, zorder=5)

    ax.set_xlabel("X " + str(d_units)); ax.set_ylabel("Y " + str(d_units)); ax.set_zlabel("Z " + str(d_units))
    ax.legend(loc="upper left")
    title = ax.set_title("", pad=12)

    artists = (line_true, line_mamba, line_lstm, dot_true, dot_mamba, dot_lstm, title)

    def init():
        for a in artists[:-1]:
            a.set_data([], [])
            if hasattr(a, "set_3d_properties"):
                a.set_3d_properties([])
        return artists

    def update(frame_idx):
        i  = idx[frame_idx]
        lo = max(0, i - trail)

        for line, pos in [(line_true, true_pos), (line_mamba, mamba_pos), (line_lstm, lstm_pos)]:
            line.set_data(pos[lo:i+1, 0], pos[lo:i+1, 1])
            line.set_3d_properties(pos[lo:i+1, 2])
        for dot, pos in [(dot_true, true_pos), (dot_mamba, mamba_pos), (dot_lstm, lstm_pos)]:
            dot.set_data([pos[i, 0]], [pos[i, 1]])
            dot.set_3d_properties([pos[i, 2]])

        prog_bar.set_width(i / (n - 1))
        title.set_text(f"{system.upper()}   t = {t[i]:.3f} "+str(t_units))
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=len(idx), init_func=init,
        interval=interval, blit=False,
    )

    _show_or_save_plot(ani, save, system)



def animate_2d(true_pos, mamba_pos, lstm_pos, t, system, trail, n_frames, interval, save,d_units,t_units):
    fig = plt.figure(figsize=(9, 8))
    ax      = fig.add_axes([0.10, 0.08, 0.87, 0.88])
    prog_ax = fig.add_axes([0.10, 0.02, 0.87, 0.03])
    prog_ax.set_xlim(0, 1); prog_ax.set_ylim(0, 1)
    prog_ax.set_xticks([]); prog_ax.set_yticks([])
    prog_bar = prog_ax.barh(0.5, 0, height=1, color=COLORS["true"], align="center")[0]

    # Limits from true trajectory only — network predictions can diverge wildly
    pad = 0.15
    for dim, setter in enumerate([ax.set_xlim, ax.set_ylim]):
        lo_v, hi_v = true_pos[:, dim].min(), true_pos[:, dim].max()
        margin = max((hi_v - lo_v) * pad, 1e-6)
        setter(lo_v - margin, hi_v + margin)

    n    = len(true_pos)
    step = max(1, n // n_frames)
    idx  = np.arange(0, n, step)

    if system == "3bp":
        m_1 = 5.974E24  # Mass of Earth in kg
        m_2 = 7.348E22  # Mass of Moon in kg
        mu = m_2 / (m_1 + m_2)

        DU = 389703
        G = 6.67430e-11
        TU = 382981

        earthLocation = -mu
        moonLocation = (1 - mu)

        earthLocation = earthLocation * DU
        moonLocation = moonLocation * DU

        L1 = [0.8369154703225321* DU,0] 
        L2 = [1.1556818961296604* DU,0]
        L3 = [-1.0050626166357435* DU,0]
        L4 = [0.48784941* DU,0.86602540* DU] 
        L5 = [0.48784941* DU,-0.86602540* DU]

        print(L1[0])

        ax.scatter(earthLocation, 0, color = 'k',marker = 'o', label='Earth')
        ax.scatter(moonLocation, 0, color = 'g', marker = 'o', label='Moon')

        ax.scatter(L1[0], L1[1], color = 'grey',marker = '*',alpha = 0.3)
        ax.scatter(L2[0], L2[1], color = 'grey',marker = '*',alpha = 0.3)
        ax.scatter(L3[0], L3[1], color = 'grey',marker = '*',alpha = 0.3)
        ax.scatter(L4[0], L4[1], color = 'grey',marker = '*',alpha = 0.3)
        ax.scatter(L5[0], L5[1], color = 'grey',marker = '*',alpha = 0.3)


    # Full true trajectory ghost (static)
    ax.plot(true_pos[:, 0], true_pos[:, 1],
            color="k", lw=0.8, alpha=0.2, ls="--", zorder=1)

    line_true,  = ax.plot([], [], color=COLORS["true"],  lw=1.4, alpha=0.9, label="True")
    line_mamba, = ax.plot([], [], color=COLORS["mamba"], lw=1.1, alpha=0.8, label="Mamba")
    line_lstm,  = ax.plot([], [], color=COLORS["lstm"],  lw=1.1, alpha=0.8, label="LSTM")
    dot_true,   = ax.plot([], [], "o", color=COLORS["true"],  ms=7, zorder=5)
    dot_mamba,  = ax.plot([], [], "o", color=COLORS["mamba"], ms=6, zorder=5)
    dot_lstm,   = ax.plot([], [], "o", color=COLORS["lstm"],  ms=6, zorder=5)

    ax.set_xlabel("X " + str(d_units)); ax.set_ylabel("Y " + str(d_units))
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")
    ax.grid(True, lw=0.5)
    title = ax.set_title("", pad=10)

    artists = (line_true, line_mamba, line_lstm, dot_true, dot_mamba, dot_lstm, title)

    def init():
        for a in artists[:-1]:
            a.set_data([], [])
        return artists

    def update(frame_idx):
        i  = idx[frame_idx]
        lo = max(0, i - trail)

        for line, pos in [(line_true, true_pos), (line_mamba, mamba_pos), (line_lstm, lstm_pos)]:
            line.set_data(pos[lo:i+1, 0], pos[lo:i+1, 1])
        for dot, pos in [(dot_true, true_pos), (dot_mamba, mamba_pos), (dot_lstm, lstm_pos)]:
            dot.set_data([pos[i, 0]], [pos[i, 1]])

        prog_bar.set_width(i / (n - 1))
        title.set_text(f"{system.upper()}   t = {t[i]:.4f} "+str(t_units))
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=len(idx), init_func=init,
        interval=interval, blit=False,
    )

    _show_or_save_plot(ani, save, system)



def plot_timing(data, system,save):
    timing_keys = ("timeToTrainMamba", "timeToTrainLSTM", "timeToTestMamba", "timeToTestLSTM")
    if not all(k in data.files for k in timing_keys):
        return

    train_mamba  = float(data["timeToTrainMamba"])
    train_lstm   = float(data["timeToTrainLSTM"])
    test_mamba   = float(data["timeToTestMamba"])
    test_lstm    = float(data["timeToTestLSTM"])
    has_params   = "paramsMamba" in data.files and "paramsLSTM" in data.files
    params_mamba = int(data["paramsMamba"]) if has_params else None
    params_lstm  = int(data["paramsLSTM"])  if has_params else None

    ncols = 3 if has_params else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4))
    fig.suptitle(f"{system.upper()} — Model Comparison", fontsize=13)

    def fmt_bytes(n):
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.3g} {unit}"
            n /= 1024
        return f"{n:.3g} TB"

    panels = [
        ("Train Time (s)", train_mamba,  train_lstm,  "{:.4g}s", None, None),
        ("Test Time (s)",  test_mamba,   test_lstm,   "{:.4g}s", None, None),
    ]
    if has_params:
        panels.append((
            "Learnable Parameters",
            params_mamba, params_lstm, "{:,}",
            fmt_bytes(params_mamba * 8), fmt_bytes(params_lstm * 8),
        ))

    for ax, (title, mamba_val, lstm_val, fmt, mem_mamba, mem_lstm) in zip(axes, panels):
        bars = ax.bar(
            ["Mamba", "LSTM"],
            [mamba_val, lstm_val],
            color=[COLORS["mamba"], COLORS["lstm"]],
            width=0.5,
        )
        for bar, val, mem in zip(bars, [mamba_val, lstm_val], [mem_mamba, mem_lstm]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                fmt.format(val),
                ha="center", va="bottom", fontsize=10,
            )
            if mem is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    mem,
                    ha="center", va="center", fontsize=9, alpha=0.75,
                )
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(mamba_val, lstm_val) * 1.25)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Mamba", "LSTM"])

    plt.tight_layout()

    if save:
        plt.savefig("plots/"+str(system)+"/parameters.png")

def _show_or_save_plot(ani, save, system):
    plt.tight_layout()
    if save:
        ani.save("plots/"+str(system)+"/phase_space.mp4", writer="ffmpeg", fps=30, dpi=150)
        print(f"Saved → {save}")
    else:
        plt.show()


def main():
    args = parse_args()
    data = load_data(args.system)
    true_pos, mamba_pos, lstm_pos, t, d_units, t_units ,plot_type = extract(data, args.system)
    print(d_units)
    kwargs = dict(
        true_pos=true_pos, mamba_pos=mamba_pos, lstm_pos=lstm_pos, t=t,
        system=args.system, trail=args.trail,
        n_frames=args.frames, interval=args.interval, save=args.save,
        d_units = d_units, t_units = t_units,
    )

    plot_timing(data, args.system,args.save)

    if plot_type == "3d":
        animate_3d(**kwargs)
    else:
        animate_2d(**kwargs)



if __name__ == "__main__":
    main()
