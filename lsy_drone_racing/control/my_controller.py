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
        waypoints = [start_pos, start_pos + np.array([0.0, 0.0, 0.3])]  # erst hochsteigen
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

        self._known_gates_pos = np.array(obs["gates_pos"])

        self._known_obst_pos = np.array(obs["obstacles_pos"])

    def _safe_spline(self, waypoints, t0, bc_type, obs):
        """Spline bauen und Ausweichpunkte einfügen, bis Pfosten und Rahmen frei sind."""
        speed = 0.8
        # Hindernisse als senkrechte Stangen: (x, y, Sicherheitsradius)
        poles = [(o[0], o[1], 0.2) for o in obs["obstacles_pos"]]
        for p, q in zip(obs["gates_pos"], obs["gates_quat"]):
            side = R.from_quat(q).apply([0.0, 1.0, 0.0])  # Richtung quer zum Gate
            for s in (-0.28, 0.28):  # linker und rechter Rahmen
                poles.append((p[0] + s * side[0], p[1] + s * side[1], 0.18))
        waypoints = [np.array(w, dtype=float) for w in waypoints]
        for _ in range(40):
            wp = np.array(waypoints)
            seg_len = np.maximum(np.linalg.norm(np.diff(wp, axis=0), axis=1), 1e-3)
            t = t0 + np.concatenate([[0.0], np.cumsum(seg_len / speed)])
            spline = CubicSpline(t, wp, bc_type=bc_type)
            ts = np.arange(t[0] + 0.1, t[-1], 0.02)  # Bahn in kleinen Schritten ablaufen
            path = spline(ts)
            hit = None
            for k, pt in enumerate(path):
                for px, py, r in poles:
                    diff = pt[:2] - np.array([px, py])
                    dist = np.linalg.norm(diff)
                    if dist < r:
                        hit = (k, px, py, r, diff, dist)
                        break
                if hit:
                    break
            if hit is None:  # Bahn ist frei
                break
            k, px, py, r, diff, dist = hit
            if dist < 1e-6:  # genau auf der Stange: quer zur Flugrichtung ausweichen
                vel = spline(ts[k], 1)
                diff = np.array([-vel[1], vel[0]])
                dist = np.linalg.norm(diff) + 1e-9
            new = path[k].copy()
            new[:2] = np.array([px, py]) + diff / dist * (r + 0.1)  # nach außen schieben
            idx = int(np.searchsorted(t, ts[k]))
            waypoints.insert(idx, new)
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
        if self._tick % self._freq == 0:  # einmal pro Sekunde
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
