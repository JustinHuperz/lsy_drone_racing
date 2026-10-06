"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class StateController(Controller):
    """State controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq



        #####neu
        # Wegpunkte aus den Gate-Positionen berechnen
        start_pos = obs["pos"]
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        d = 0.3  # Abstand vor und hinter dem Gate in m
        waypoints = [start_pos]
        for gate_idx, direction in zip(obs["gate_sequence"], obs["gate_sequence_direction"]):
            p = gates_pos[gate_idx]
            normal = R.from_quat(gates_quat[gate_idx]).apply([1.0, 0.0, 0.0]) * direction
            waypoints.append(p - d * normal)
            waypoints.append(p)
            waypoints.append(p + d * normal)
        waypoints = np.array(waypoints)

        self._des_pos_spline, self._t_total = self._safe_spline(waypoints, 0.0, "not-a-knot", obs)


        self._tick = 0
        self._finished = False
        self._z_offset = 0.0  # aufsummierte Höhenkorrektur in m
        self._ki = 1.5  # Stärke des Integralanteils
        self._verbose = False  # auf True setzen, um die Ausgaben wieder zu sehen

        self._known_gates_pos = np.array(obs["gates_pos"])

        self._known_obst_pos = np.array(obs["obstacles_pos"])

    def _safe_spline(self, waypoints, t0, bc_type, obs):
        """Spline bauen und Ausweichpunkte einfügen, bis Pfosten und Gate-Rahmen frei sind."""
        speed = 0.8
        poles = np.array([(o[0], o[1], 0.15) for o in obs["obstacles_pos"]])  # x, y, Radius
        gates = []
        for c, q in zip(obs["gates_pos"], obs["gates_quat"]):
            rot = R.from_quat(q)
            n = rot.apply([1.0, 0.0, 0.0])  # Durchflugrichtung
            s = rot.apply([0.0, 1.0, 0.0])  # quer zum Gate
            gates.append((np.array(c), n, s))
        waypoints = [np.array(w, dtype=float) for w in waypoints]
        movable = [False] * len(waypoints)  # nur eingefügte Ausweichpunkte dürfen verschoben werden
        for _ in range(40):
            wp = np.array(waypoints)
            seg_len = np.maximum(np.linalg.norm(np.diff(wp, axis=0), axis=1), 1e-3)
            t = t0 + np.concatenate([[0.0], np.cumsum(seg_len / speed)])
            spline = CubicSpline(t, wp, bc_type=bc_type)
            ts = np.arange(t[0] + 0.1, t[-1], 0.02)  # Bahn in kleinen Schritten ablaufen
            path = spline(ts)
            first_k, new = None, None

            # 1) Pfosten: senkrechte Stangen
            diffs = path[:, None, :2] - poles[None, :, :2]
            dists = np.linalg.norm(diffs, axis=2)
            inside = dists < poles[None, :, 2]
            if inside.any():
                k, j = np.argwhere(inside)[0]
                d = diffs[k, j] if dists[k, j] > 1e-6 else np.array([1.0, 0.0])
                first_k = k
                new = path[k].copy()
                new[:2] = poles[j, :2] + d / np.linalg.norm(d) * (poles[j, 2] + 0.15)

            # 2) Gate-Rahmen: die ganze Gate-Ebene ist gesperrt, nur die Öffnung ist frei
            for c, n, s in gates:
                rel = path - c
                a = rel @ n  # Abstand vor/hinter dem Gate
                l = rel @ s  # Abstand seitlich von der Mitte
                dz = rel[:, 2]  # Abstand nach oben/unten
                in_opening = (np.abs(l) < 0.14) & (np.abs(dz) < 0.14)
                hit = (np.abs(a) < 0.08) & (np.abs(l) < 0.46) & ~in_opening
                if hit.any():
                    k = int(np.argmax(hit))
                    if first_k is None or k < first_k:
                        first_k = k
                        if abs(l[k]) < 0.2 and abs(dz[k]) < 0.2:
                            new = c + a[k] * n  # knapp am Rand der Öffnung: zur Mitte schieben
                        else:
                            # außen um das Gate herum, auf der Seite mit mehr Abstand zu den Pfosten
                            best = None
                            for side in (1.0, -1.0):
                                cand = path[k] + (side * 0.65 - l[k]) * s
                                clear = np.min(np.linalg.norm(poles[:, :2] - cand[:2], axis=1))
                                if side == np.sign(l[k] + 1e-9):
                                    clear += 0.05  # bei Gleichstand die nähere Seite nehmen
                                if best is None or clear > best[0]:
                                    best = (clear, cand)
                            new = best[1]

            if first_k is None:  # Bahn ist frei
                break
            # Gibt es schon einen Ausweichpunkt in der Nähe, wird der weiter rausgeschoben,
            # statt immer neue Punkte dicht nebeneinander einzufügen.
            near = [i for i in range(len(waypoints)) if movable[i] and np.linalg.norm(waypoints[i] - new) < 0.25]
            if near:
                waypoints[near[0]] = new + 0.5 * (new - path[first_k])
            else:
                idx = int(np.searchsorted(t, ts[first_k]))
                waypoints.insert(idx, new)
                movable.insert(idx, True)
        return spline, t[-1]

    def _replan(self, obs, t_now):
        """Restliche Bahn ab dem aktuellen Sollpunkt neu planen."""
        t_eval = min(t_now, self._t_total)
        p0 = self._des_pos_spline(t_eval)  # aktueller Sollpunkt
        v0 = self._des_pos_spline(t_eval, 1)  # aktuelle Sollgeschwindigkeit
        d = 0.3
        speed = 0.8
        waypoints = [p0]
        first = int(obs["n_gates_passed"])  # erstes noch offenes Gate
        for i in range(first, len(obs["gate_sequence"])):
            g = obs["gate_sequence"][i]
            p = obs["gates_pos"][g]
            normal = R.from_quat(obs["gates_quat"][g]).apply([1.0, 0.0, 0.0])
            normal = normal * obs["gate_sequence_direction"][i]
            pre = p - d * normal
            # Punkt vor dem aktuellen Gate weglassen, wenn wir schon daran vorbei sind
            if i > first or np.dot(pre - p0, normal) > 0:
                waypoints.append(pre)
            waypoints.append(p)
            waypoints.append(p + d * normal)
        waypoints = np.array(waypoints)
        bc = ((1, v0), "not-a-knot")
        self._des_pos_spline, self._t_total = self._safe_spline(waypoints, t_now, bc, obs)


    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz] as a
            numpy array.
        """
        self._last_obs = obs
        if self._verbose and self._tick % self._freq == 0:  # einmal pro Sekunde
            print(f"\nt = {self._tick / self._freq:.0f} s")
            print("Drohne:", np.round(obs["pos"], 2))
            print("Soll:  ", np.round(self._des_pos_spline(min(self._tick / self._freq, self._t_total)), 2))
            print("Gates passiert:", obs["n_gates_passed"], "| Reihenfolge:", obs["gate_sequence"])
            print("Gate-Positionen:\n", np.round(obs["gates_pos"], 2))
        
        gates_now = np.array(obs["gates_pos"])
        obst_now = np.array(obs["obstacles_pos"])
        gates_moved = np.any(np.abs(gates_now - self._known_gates_pos) > 1e-3)
        obst_moved = np.any(np.abs(obst_now - self._known_obst_pos) > 1e-3)
        if gates_moved or obst_moved:
            self._known_gates_pos = gates_now
            self._known_obst_pos = obst_now

            if obs["n_gates_passed"] < len(obs["gate_sequence"]):
                self._replan(obs, self._tick / self._freq)
                print(f"t = {self._tick / self._freq:.2f} s: neu geplant")
        
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        des_pos = self._des_pos_spline(t)
        des_vel = self._des_pos_spline(t, 1)  # 1. Ableitung = Geschwindigkeit
        des_acc = self._des_pos_spline(t, 2)  # 2. Ableitung = Beschleunigung
        if t >= self._t_total:  # am Ende stillstehen
            des_vel = np.zeros(3)
            des_acc = np.zeros(3)

        z_error = obs["pos"][2] - des_pos[2]
        self._z_offset += self._ki * z_error / self._freq
        self._z_offset = np.clip(self._z_offset, -0.2, 0.2)  # Sicherheitsgrenze
        des_pos = des_pos.copy()
        des_pos[2] -= self._z_offset

        des_yaw = 0.0
        des_quat = R.from_euler("z", des_yaw).as_quat()
        action = np.concatenate((des_pos, des_vel, des_acc, des_quat, np.zeros(3)), dtype=np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(self._tick / self._freq, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        for p, q in zip(self._last_obs["gates_pos"], self._last_obs["gates_quat"]):
            normal = R.from_quat(q).apply([1.0, 0.0, 0.0])
            draw_line(sim, np.array([p, p + 0.5 * normal]), rgba=(1.0, 0.0, 0.0, 1.0))
