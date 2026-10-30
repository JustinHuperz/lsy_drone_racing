"""Level-2-Controller mit Reinforcement Learning (neuronale Netze, PPO), Version 5.

Zwei Sätze Netze in einer Datei, die passende Variante wird automatisch gewählt:

* Simulation (scripts/sim.py, Leaderboard): rl_policy.npz, neu in Version 5. Die Netze sehen
  zusätzlich alle 4 Gates (nicht nur die nächsten zwei), die Lage der Drohne im Koordinatensystem
  des nächsten Gates und ob die Position eines Gates/Pfostens schon exakt bekannt ist.
  Ergebnis (600 neue Zufallsstrecken, scripts/sim.py): 94 % Erfolg (563/600), im Mittel 4,09 s,
  schnellste Runde 3,74 s. Version 4 in derselben Messung: 94 % (563/600), im Mittel 4,21 s.
* Echte Drohne (scripts/deploy.py): rl_policy_real.npz, unverändert zu Version 4. Diese Netze sind
  mit allen simulierten Labor-Störungen trainiert: 0–40 ms Verzögerung, Schub ±12 % plus
  Akku-Abfall, träge Motoren bzw. Lageregelung, Trimmfehler, Funkverluste, Vicon-Aussetzer,
  Messrauschen und Filterverzug, Messfehler der Gates/Pfosten, Böen, Bodeneffekt und gehaltener
  Gierwinkel (real wird Gierrate 0 gesendet). Sie sehen zusätzlich einen kurzen Verlauf (Befehle
  der letzten 3 Schritte und die letzte Geschwindigkeit). Erkannt wird der echte Betrieb daran,
  dass deploy.py ROS2 (rclpy) geladen hat. Mit MODE = "sim" / "real" lässt sich das erzwingen.
  Details und Zahlen: LABOR.md.

Jedes Netz bekommt 50-mal pro Sekunde den Zustand der Drohne, die Gates und die Pfosten und gibt
direkt den Befehl [roll, pitch, yaw, Schub] aus. Es gibt keine Bahnplanung und kein MPC. Geflogen
wird der Mittelwert der Befehle mehrerer Netze (Ensemble).

Für das Labor: THRUST_SCALE multipliziert den Schubbefehl (nur im echten Betrieb). Steigt die
Drohne beim Start deutlich zu schnell oder kommt sie kaum hoch, hier nachjustieren (0,95 / 1,05).

Start: python scripts/sim.py --config level2.toml --controller rl_controller.py
       python scripts/deploy.py --config level2.toml --controller rl_controller.py
Benötigt in config/level2.toml: control_mode = "attitude".
Gewichte: rl_policy.npz und rl_policy_real.npz im selben Ordner wie diese Datei.
Training: rl_training/README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_TILT = 1.0  # rad, Bereich für roll/pitch
T_MID, T_AMP = 0.475, 0.325  # Schub = T_MID + T_AMP * a  (0.15 ... 0.80 N)
# "auto": echte Drohne, wenn ROS2 (rclpy) geladen ist, sonst Simulation. Oder "sim" / "real".
MODE = "auto"
THRUST_SCALE = 1.0  # nur echter Betrieb: Faktor auf den Schubbefehl (Kalibrierung im Labor)


def quat_to_mat(q: NDArray) -> NDArray:
    """Quaternion(en) xyzw (..., 4) -> Rotationsmatrix (..., 3, 3)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def features(obs: dict, last_act: NDArray, seq: NDArray) -> NDArray:
    """Eingang des Netzes (54 Werte), genau wie im Training."""
    pos, vel = obs["pos"], obs["vel"]
    n_pass = len(seq)
    n = int(obs["n_gates_passed"])
    parts = [pos / 2.0, vel / 4.0, quat_to_mat(obs["quat"]).reshape(9), obs["ang_vel"] / 10.0]
    for k in (np.clip(n, 0, n_pass - 1), np.clip(n + 1, 0, n_pass - 1)):
        g = seq[k]
        gm = quat_to_mat(obs["gates_quat"][g])
        parts += [(obs["gates_pos"][g] - pos) / 2.0, gm[:, 0], gm[:, 1]]
    parts.append(((obs["obstacles_pos"][:, :2] - pos[:2]) / 2.0).reshape(8))
    parts += [(np.arange(n_pass + 1) == np.clip(n, 0, n_pass)).astype(float), last_act]
    return np.concatenate(parts)


def features_ext(obs: dict, seq: NDArray, nom_gates: NDArray) -> NDArray:
    """Zusatzeingänge (21 Werte) der Sim-Netze ab Version 5, genau wie im Training.

    Lage und Geschwindigkeit der Drohne im Koordinatensystem des nächsten Gates (Lage auch für das
    übernächste), wie weit das Gate von seiner Nennposition verschoben ist, und ob die genaue
    Position des Gates bzw. der Pfosten schon bekannt ist (Sensorreichweite 0,7 m).
    """
    pos, vel = obs["pos"], obs["vel"]
    n_pass = len(seq)
    n = int(obs["n_gates_passed"])
    gv = np.asarray(obs.get("gates_visited", np.zeros(len(nom_gates))), dtype=float)
    ov = np.asarray(obs.get("obstacles_visited", np.zeros(4)), dtype=float)
    parts = []
    for j, k in enumerate((np.clip(n, 0, n_pass - 1), np.clip(n + 1, 0, n_pass - 1))):
        g = seq[k]
        gp = obs["gates_pos"][g]
        gm = quat_to_mat(obs["gates_quat"][g])
        parts.append(gm.T @ (pos - gp) / (1.0 if j == 0 else 2.0))
        if j == 0:
            parts.append(gm.T @ vel / 4.0)
        parts.append((gp - nom_gates[g]) / 0.15)
        parts.append(gv[g : g + 1])
    parts.append(ov)
    return np.concatenate(parts)


def features_gates(obs: dict) -> NDArray:
    """Alle 4 Gates (20 Werte): Lage relativ zur Drohne und Ausrichtung. So sieht das Netz auch Gates,
    die gerade nicht Ziel sind, aber im Weg liegen (z. B. Gate 1 auf dem Weg von Gate 2 zu Gate 3)."""
    gm = quat_to_mat(np.asarray(obs["gates_quat"]))
    return np.concatenate([((obs["gates_pos"] - obs["pos"]) / 2.0).reshape(-1), gm[:, :2, 0].reshape(-1)])


class RLController(Controller):
    """Mehrere neuronale Netze (je 2 x 128 tanh), Befehl = Mittelwert ihrer Ausgaben."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Gewichte laden (erste Achse = Nummer des Netzes)."""
        super().__init__(obs, info, config)
        self.real = MODE == "real" or (MODE == "auto" and "rclpy" in sys.modules)
        name = "rl_policy_real.npz" if self.real else "rl_policy.npz"
        w = np.load(Path(__file__).parent / name)
        add_axis = (lambda x: x[None]) if w["W0"].ndim == 2 else (lambda x: x)  # auch Einzelnetz
        self.layers = [(add_axis(w[f"W{i}"]), add_axis(w[f"b{i}"])) for i in range(3)]
        self.obs_mean, self.obs_var = add_axis(w["obs_mean"]), add_axis(w["obs_var"])
        self.seq = np.abs(np.asarray(config.env.track.gate_order)) - 1
        self.n_in = self.layers[0][0].shape[1]
        # Eingänge: 54 Grundwerte, +21 (+20) Zusatzwerte (Sim-Netze ab V5, "feat" = 2 bzw. 3), +11 Verlauf
        # (Real-Netze: Befehle t-2, t-3 und Geschwindigkeit t-1)
        self.feat = int(w["feat"]) if "feat" in w.files else 1  # 2: +21 Werte, 3: +21 +20 Werte
        self.nom_gates = np.array([g["pos"] for g in config.env.track.gates], dtype=float)
        self.episode_callback()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Ein Schritt: Zustand -> Netz -> [roll, pitch, yaw, Schub]."""
        x = features(obs, self.last, self.seq)
        if self.feat >= 2:
            x = np.concatenate([x, features_ext(obs, self.seq, self.nom_gates)])
        if self.feat == 3:
            x = np.concatenate([x, features_gates(obs)])
        if self.n_in > len(
            x
        ):  # kurzer Verlauf, damit das Netz Verzögerung/Schub der Drohne erkennt
            v_prev = obs["vel"] if self.v_prev is None else self.v_prev
            x = np.concatenate([x, self.a2, self.a3, v_prev / 4.0])
        if not np.all(np.isfinite(x)):  # kaputte Messung: letzten Befehl halten
            return self.cmd_prev
        x = np.clip((x[None] - self.obs_mean) / np.sqrt(self.obs_var + 1e-8), -10, 10)  # (M, n_in)
        for i, (W, b) in enumerate(self.layers):
            x = np.einsum("mi,mij->mj", x, W) + b
            if i < 2:
                x = np.tanh(x)
        a = np.clip(x, -1.0, 1.0).mean(axis=0)  # Mittelwert der Netze
        a[2] = 0.0
        self.a3, self.a2, self.last = self.a2, self.last, a
        self.v_prev = np.asarray(obs["vel"], dtype=float)
        thrust = T_MID + T_AMP * a[3]
        if self.real:
            thrust *= THRUST_SCALE
        self.cmd_prev = np.array([a[0] * MAX_TILT, a[1] * MAX_TILT, 0.0, thrust], dtype=np.float32)
        return self.cmd_prev

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Nach jedem Schritt: fertig, wenn alle Gates durchflogen sind."""
        n = int(obs["n_gates_passed"])
        self.finished = n >= len(self.seq) or n == -1
        return self.finished

    def episode_callback(self):
        """Neue Runde: Verlauf zurücksetzen."""
        self.last, self.a2, self.a3 = np.zeros(4), np.zeros(4), np.zeros(4)
        self.v_prev = None
        self.cmd_prev = np.array([0.0, 0.0, 0.0, T_MID - T_AMP], dtype=np.float32)
        self.finished = False
