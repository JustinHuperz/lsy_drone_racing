"""Viele Läufe ohne Rendering simulieren und die Ergebnisse von oben zeichnen.

Aufruf im Projektordner (nach `pixi shell`):

    python scripts/plot_runs.py --config level2.toml --controller my_controller.py --n_runs 20

Optional: `--seed 7` macht die Läufe wiederholbar, `--out auswertung` wählt den Ausgabeordner.

Ergebnis im Ausgabeordner:
    uebersicht.png  alle Läufe übereinander, mit Zusammenfassung
    laeufe.png      ein Bild pro Lauf (Bahn, Gates dieses Laufs, Absturzstelle, Zeit)
"""

from __future__ import annotations

from pathlib import Path

# Muss vor scipy geladen werden (crazyflow setzt dabei eine nötige Einstellung für scipy)
from lsy_drone_racing.utils import load_config, load_controller  # isort: skip

import fire  # noqa: E402
import gymnasium  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Farben: Gates (feste Reihenfolge, geprüft auf Farbfehlsichtigkeit; zusätzlich Beschriftung
# und eigene Markerform je Gate), Rest neutral. Rot ist nur für Abstürze reserviert.
GATE_COLORS = ["#2a78d6", "#e87ba4", "#1baf7a", "#eda100"]
GATE_MARKERS = ["o", "s", "^", "D"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CRASH = "#d03b3b"
SURFACE = "#fcfcfb"

# Maße aus gate.xml und obstacle.xml
GATE_OUTER = 0.36  # halbe Außenbreite des Rahmens in m
GATE_INNER = 0.20  # halbe Breite der Öffnung in m
VIEW_X, VIEW_Y = (-2.65, 2.65), (-1.68, 1.68)  # Übersicht: ganzes Lab
PANEL_X, PANEL_Y = (-1.95, 1.75), (-1.58, 1.3)  # Einzelbilder: nur die Strecke


def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def _yaw(quat: np.ndarray) -> float:
    return float(R.from_quat(quat).as_euler("xyz")[2])


def _crash_cause(pos, gates_pos, gates_quat, obstacles_pos, low, high) -> str:
    """Wahrscheinliche Absturzursache aus der letzten gültigen Position."""
    d_obst = np.linalg.norm(obstacles_pos[:, :2] - pos[:2], axis=1)
    j = int(np.argmin(d_obst))
    if d_obst[j] < 0.17:
        return f"Pfosten {j}"
    if pos[2] < 0.08:
        return "Boden"
    if np.any(pos < low + 0.05) or np.any(pos > high - 0.05):
        return "Spielfeldrand"
    for g in range(len(gates_pos)):
        rel = R.from_quat(gates_quat[g]).inv().apply(pos - gates_pos[g])
        if abs(rel[0]) < 0.2 and max(abs(rel[1]), abs(rel[2])) < 0.5:
            where = "seitlich" if abs(rel[1]) > abs(rel[2]) else ("oben" if rel[2] > 0 else "unten")
            return f"Rahmen G{g} {where}"
    return "unklar"


def run_episodes(config: str, controller: str, n_runs: int, seed: int | None) -> tuple[list, dict]:
    """Läufe wie in scripts/sim.py simulieren und alles Nötige mitschreiben."""
    root = Path(__file__).parents[1]
    cfg = load_config(root / "config" / config)
    cfg.sim.render = False
    if seed is not None:
        cfg.env.seed = seed
    controller_cls = load_controller(root / "lsy_drone_racing/control" / controller)
    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)
    track = cfg.env.track
    info_track = {
        "nom_gates": np.array([g["pos"] for g in track.gates], dtype=float),
        "nom_gate_yaw": np.array([g["rpy"][2] for g in track.gates], dtype=float),
        "nom_obst": np.array([o["pos"] for o in track.obstacles], dtype=float),
        "gate_rand": _rand_range(cfg, "gate_pos"),
        "obst_rand": _rand_range(cfg, "obstacle_pos"),
        "lab_low": np.array(track.safety_limits.pos_limit_low, dtype=float),
        "lab_high": np.array(track.safety_limits.pos_limit_high, dtype=float),
        "config": config,
        "controller": controller,
    }
    core = env.unwrapped
    sim_low = np.asarray(core.data.pos_limit_low).reshape(-1)[:3]
    sim_high = np.asarray(core.data.pos_limit_high).reshape(-1)[:3]

    runs = []
    for k in range(n_runs):
        obs, info = env.reset()
        ctrl = controller_cls(obs, info, cfg)
        gates_pos = np.array(core.data.gates_pos)[0]
        gates_quat = np.array(core.data.gates_quat)[0]
        obst_pos = np.array(core.data.obstacles_pos)[0]
        n_total = int(obs["gate_sequence"].shape[0])
        path, times = [np.array(obs["pos"], dtype=float)], [0.0]
        visited = np.array(obs["gates_visited"], dtype=bool)
        reveals = [(g, path[0].copy()) for g in np.flatnonzero(visited)]
        passes = []
        i, end = 0, "Zeit abgelaufen"
        while True:
            t = i / cfg.env.freq
            last_pos = np.array(obs["pos"], dtype=float)
            action = ctrl.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            finished_ctrl = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
            n_passed = int(obs["n_gates_passed"])
            if terminated or truncated or finished_ctrl:
                if n_passed >= n_total:
                    end = "geschafft"
                elif terminated:
                    end = _crash_cause(last_pos, gates_pos, gates_quat, obst_pos, sim_low, sim_high)
                elif finished_ctrl:
                    end = "Bahn zu Ende, Gate verfehlt"
                break
            pos = np.array(obs["pos"], dtype=float)
            path.append(pos)
            times.append(t + 1 / cfg.env.freq)
            new = np.array(obs["gates_visited"], dtype=bool) & ~visited
            reveals += [(g, pos.copy()) for g in np.flatnonzero(new)]
            visited |= new
            if n_passed > len(passes):
                passes.append(pos.copy())
            i += 1
        ctrl.episode_callback()
        ctrl.episode_reset()
        ok = end == "geschafft"
        runs.append({
            "path": np.array(path), "times": np.array(times), "gates_pos": gates_pos,
            "gates_quat": gates_quat, "obst_pos": obst_pos, "reveals": reveals,
            "passes": passes, "ok": ok, "time": t, "n_passed": n_passed,
            "end": end, "crash_pos": None if ok or end.startswith("Zeit") else last_pos,
        })
        status = f"{de(t)} s" if ok else f"{end} nach {n_passed} Gates (t = {de(t)} s)"
        print(f"Lauf {k + 1:2d}: {status}")
    env.close()
    return runs, info_track


def _rand_range(cfg, key: str) -> np.ndarray:
    """Halbe Breite der zufälligen Verschiebung in x und y (0, wenn nicht randomisiert)."""
    rnd = cfg.env.get("randomizations") or {}
    if key not in rnd:
        return np.zeros(2)
    kw = rnd[key]["kwargs"]
    return np.maximum(np.abs(kw["minval"][:2]), np.abs(kw["maxval"][:2])).astype(float)


def _draw_track_base(ax, tr: dict, view=(VIEW_X, VIEW_Y)):
    """Spielfeld, Lab-Grenze, mögliche Gate- und Pfostenpositionen, Gate-Namen."""
    ax.set_facecolor(SURFACE)
    lo, hi = tr["lab_low"], tr["lab_high"]
    ax.add_patch(plt.Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False,
                               ec=INK2, lw=1.0, ls=(0, (6, 4)), zorder=1))
    for g, (c, yaw) in enumerate(zip(tr["nom_gates"], tr["nom_gate_yaw"])):
        rx, ry = tr["gate_rand"]
        if rx > 0:
            ax.add_patch(plt.Rectangle((c[0] - rx, c[1] - ry), 2 * rx, 2 * ry, fill=False,
                                       ec=GATE_COLORS[g], lw=1.0, ls=(0, (3, 3)), zorder=2))
        ax.plot(c[0], c[1], "+", color=GATE_COLORS[g], ms=7, mew=1.2, zorder=2)
    for o in tr["nom_obst"]:
        rx, ry = tr["obst_rand"]
        if rx > 0:
            ax.add_patch(plt.Rectangle((o[0] - rx, o[1] - ry), 2 * rx, 2 * ry, fill=False,
                                       ec=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=2))
    ax.set_xlim(*view[0])
    ax.set_ylim(*view[1])
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def _gate_label(ax, tr: dict, size: float = 9):
    """Gate-Namen neben den nominalen Positionen (Farbe nie allein)."""
    for g, (c, yaw) in enumerate(zip(tr["nom_gates"], tr["nom_gate_yaw"])):
        off = np.array([np.cos(yaw), np.sin(yaw)]) * 0.33  # in Flugrichtung versetzt
        ax.text(c[0] - off[0], c[1] - off[1], f"G{g}", color=INK, fontsize=size,
                fontweight="bold", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))


def _draw_gate(ax, pos, quat, g: int, alpha: float = 1.0, lw: float = 3.0):
    """Gate von oben: Rahmenbalken dick, Öffnung dünn, kleiner Pfeil in Flugrichtung."""
    yaw = _yaw(quat)
    d = np.array([np.cos(yaw), np.sin(yaw)])  # Durchflugrichtung
    n = np.array([-d[1], d[0]])  # quer dazu
    p = pos[:2]
    col = GATE_COLORS[g]
    for s in (1, -1):
        a, b = p + s * n * GATE_INNER, p + s * n * GATE_OUTER
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw, alpha=alpha,
                solid_capstyle="butt", zorder=4)
    a, b = p - n * GATE_INNER, p + n * GATE_INNER
    ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=0.8, alpha=alpha, zorder=4)
    ax.annotate("", xy=p + d * 0.16, xytext=p, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, alpha=alpha,
                                mutation_scale=7))


def _summary(runs: list) -> tuple[str, str]:
    ok = [r for r in runs if r["ok"]]
    n = len(runs)
    line1 = f"{len(ok)} von {n} Läufen geschafft ({100 * len(ok) / n:.0f} %)"
    if ok:
        ts = np.array([r["time"] for r in ok])
        line1 += (f"   ·   Zeit: Median {de(np.median(ts))} s, Mittel {de(ts.mean())} s, "
                  f"schnellster {de(ts.min())} s, langsamster {de(ts.max())} s")
    causes = {}
    for r in runs:
        if not r["ok"]:
            key = r["end"].split(" seitlich")[0].split(" oben")[0].split(" unten")[0]
            causes[key] = causes.get(key, 0) + 1
    line2 = ("Abstürze: " + ", ".join(f"{k} ({v}x)" for k, v in
                                      sorted(causes.items(), key=lambda kv: -kv[1]))
             if causes else "keine Abstürze")
    return line1, line2


def plot_overview(runs: list, tr: dict, out: Path):
    fig, ax = plt.subplots(figsize=(13, 7.9), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    _draw_track_base(ax, tr)
    for r in runs:
        for g in range(len(r["gates_pos"])):
            _draw_gate(ax, r["gates_pos"][g], r["gates_quat"][g], g, alpha=0.55, lw=2.2)
        ax.scatter(r["obst_pos"][:, 0], r["obst_pos"][:, 1], s=14, color=INK2, zorder=3,
                   linewidths=0)
    for r in runs:  # Bahnen über den Gates
        col, a = (INK2, 0.45) if r["ok"] else (CRASH, 0.75)
        ax.plot(r["path"][:, 0], r["path"][:, 1], color=col, lw=0.9, alpha=a, zorder=5)
    groups = []  # nahe beieinander liegende Abstürze teilen sich eine Beschriftung
    for k, r in enumerate(runs):
        if r["crash_pos"] is None:
            continue
        c = r["crash_pos"]
        ax.plot(c[0], c[1], "x", color=CRASH, ms=10, mew=2.4, zorder=7)
        for grp in groups:
            if np.linalg.norm(grp[0][:2] - c[:2]) < 0.2:
                grp[1].append(k + 1)
                break
        else:
            groups.append((c, [k + 1]))
    for c, ks in groups:
        ax.text(c[0] + 0.07, c[1] + 0.07, ", ".join(map(str, ks)), color=CRASH, fontsize=8.5,
                fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none", alpha=0.8))
    start = runs[0]["path"][0]
    ax.plot(start[0], start[1], "o", ms=9, mfc=SURFACE, mec=INK, mew=1.6, zorder=6)
    ax.text(start[0], start[1] + 0.1, "Start", ha="center", va="bottom", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none", alpha=0.8), zorder=6)
    _gate_label(ax, tr, 10)
    lo, hi = tr["lab_low"], tr["lab_high"]
    ax.text(lo[0] + 0.04, hi[1] + 0.05, f"Lab-Grenze: x ±{de(hi[0], 1)} m, y ±{de(hi[1], 1)} m "
            "(die Simulation bricht erst bei ±3 m ab)", fontsize=8.5, color=INK2, va="bottom")
    l1, l2 = _summary(runs)
    fig.text(0.06, 0.97, f"{len(runs)} Läufe von oben: {tr['controller']} auf {tr['config']}",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.928, l1, fontsize=10.5, color=INK, va="top")
    fig.text(0.06, 0.898, l2, fontsize=10.5, color=INK2, va="top")
    handles = [
        Line2D([], [], color=INK2, lw=1.2, alpha=0.7, label="Bahn, Lauf geschafft"),
        Line2D([], [], color=CRASH, lw=1.2, label="Bahn, Lauf abgestürzt"),
        Line2D([], [], color=CRASH, marker="x", ls="", ms=8, mew=2, label="Absturzstelle (Laufnummer)"),
        Line2D([], [], color=INK2, marker="o", ls="", ms=5, label="Pfosten (echte Lage je Lauf)"),
        Line2D([], [], color=MUTED, ls=(0, (3, 3)), lw=1, label="möglicher Bereich (±15 cm in x und y)"),
        Line2D([], [], color=INK2, marker="+", ls="", ms=8, mew=1.2, label="nominale Gate-Mitte"),
    ]
    handles += [Line2D([], [], color=c, lw=3, label=f"Gate {g} (echte Lage je Lauf)")
                for g, c in enumerate(GATE_COLORS)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False,
              fontsize=9, labelcolor=INK)
    ax.set_xlabel("x in m", color=INK2, fontsize=9)
    ax.set_ylabel("y in m", color=INK2, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.78, top=0.87, bottom=0.07)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_runs(runs: list, tr: dict, out: Path, dt_mark: float = 0.5):
    n = len(runs)
    ncol = 4 if n > 6 else max(1, min(n, 3))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.55 * nrow + 1.0), dpi=110,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        r = runs[k]
        _draw_track_base(ax, tr, (PANEL_X, PANEL_Y))
        for g in range(len(r["gates_pos"])):
            _draw_gate(ax, r["gates_pos"][g], r["gates_quat"][g], g, lw=3.0)
        ax.scatter(r["obst_pos"][:, 0], r["obst_pos"][:, 1], s=16, color=INK2, zorder=3,
                   linewidths=0)
        ax.plot(r["path"][:, 0], r["path"][:, 1], color=INK, lw=1.0, zorder=5)
        step = max(1, int(round(dt_mark / (r["times"][1] - r["times"][0])))) if len(r["times"]) > 1 else 1
        m = r["path"][::step]
        ax.scatter(m[:, 0], m[:, 1], s=7, color=INK, zorder=5, linewidths=0)
        for g, p in r["reveals"]:
            ax.scatter(p[0], p[1], s=40, marker=GATE_MARKERS[g], facecolor=SURFACE,
                       edgecolor=GATE_COLORS[g], linewidths=1.6, zorder=6)
        if r["crash_pos"] is not None:
            c = r["crash_pos"]
            ax.plot(c[0], c[1], "x", color=CRASH, ms=11, mew=2.6, zorder=7)
        _gate_label(ax, tr, 7.5)
        if r["ok"]:
            title, col = f"Lauf {k + 1}: {de(r['time'])} s", INK
        else:
            title = f"Lauf {k + 1}: {r['end']}, {r['n_passed']} Gates, t = {de(r['time'])} s"
            col = CRASH
        ax.set_title(title, fontsize=9.5, color=col, loc="left", fontweight="bold")
        ax.tick_params(labelsize=6.5)
    l1, l2 = _summary(runs)
    fig.text(0.01, 0.995, f"Jeder Lauf einzeln: {tr['controller']} auf {tr['config']}",
             fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.995 - 0.35 / (3.55 * nrow + 1.0), l1 + "   ·   " + l2, fontsize=10, color=INK2,
             va="top")
    fig.text(0.01, 0.995 - 0.62 / (3.55 * nrow + 1.0),
             f"Punkte auf der Bahn alle {de(dt_mark, 1)} s (weiter Abstand = schnell). "
             "Offene Symbole in Gate-Farbe: hier wurde das Gate sichtbar (0,7 m Sensorreichweite). "
             "Rotes x: letzte Position vor dem Absturz.", fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / (3.55 * nrow + 1.0)))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(config: str = "level2.toml", controller: str = "my_controller.py", n_runs: int = 20,
         seed: int | None = None, out: str = "auswertung"):
    """Läufe simulieren und zwei Bilder speichern."""
    runs, tr = run_episodes(config, controller, n_runs, seed)
    out_dir = Path(__file__).parents[1] / out
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(runs, tr, out_dir / "uebersicht.png")
    plot_runs(runs, tr, out_dir / "laeufe.png")
    l1, l2 = _summary(runs)
    print(f"\n{l1}\n{l2}\nBilder gespeichert in {out_dir}/")


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
