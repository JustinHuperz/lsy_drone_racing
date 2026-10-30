"""Level-2-Controller: feste, optimierte Rennlinie mit weicher Korrektur.

Idee:
1. Die Level-2-Strecke ist bekannt (nominale Lage aller Gates und Pfosten in level2.toml). Deshalb
   wird vorab eine feste Rennlinie durch alle Gates gelegt (RACING_LINE). Die Wegpunkte wurden
   offline mit einem Optimierer (CMA-ES) bestimmt: kürzeste geplante Flugzeit unter den
   Nebenbedingungen Mindestabstand zu den Pfosten (0,36 m), kein Kontakt mit den Gate-Rahmen
   (Drohne als 14 x 14 x 4 cm Box, plus Reserve für die zufällige Verschiebung der Gates) und
   Abstand zum Rand des Spielfelds.
2. Entlang dieser Linie wird ein zeitoptimales Geschwindigkeitsprofil berechnet: Höchst-
   geschwindigkeit, Kurvenlimit über die Querbeschleunigung, begrenztes Beschleunigen und Bremsen.
   Vor Gates, deren echte Lage noch unbekannt ist, wird etwas langsamer geflogen.
3. Sobald ein Gate in Sensorreichweite kommt und seine echte Lage bekannt ist, wird das Bahnstück
   um dieses Gate starr mit verschoben und gedreht, weich ein- und ausgeblendet. Es wird nicht neu
   geplant, deshalb springen Geschwindigkeit und Beschleunigung nicht.
4. Der eingebaute Positionsregler wird verstärkt, indem zum Sollwert ein Korrekturterm addiert
   wird. Die Sollbeschleunigung wird leicht vorausgeschickt, weil die Drohne Zeit zum Neigen
   braucht, und um 10 % erhöht, weil der Regler in der Simulation mit einer zu kleinen Masse
   rechnet (39,3 g statt 43,4 g). Eine Integralkorrektur gleicht eine bleibende Höhenabweichung aus.

Die Rennlinie gilt nur für die Level-2-Strecke.
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


############Planer


# Rennlinie für die Level-2-Strecke, Punkt für Punkt in Flugreihenfolge (Start = Drohnenposition):
#   ("frei", [x, y, z])          Wegpunkt in Weltkoordinaten
#   ("gate", k, [dx, dy, dz])    Punkt relativ zu dem Gate, das beim k-ten Durchflug dran ist
#                                (dx = Durchflugrichtung, dy = seitlich, dz = nach oben).
#                                [0, 0, 0] ist die Gate-Mitte und markiert den Durchflug.
RACING_LINE = [
    ("frei", [-0.246, 0.395, 0.604]),
    ("gate", 0, [-0.307, -0.134, -0.01]),
    ("gate", 0, [0.0, 0.0, 0.0]),
    ("gate", 0, [0.432, -0.021, 0.014]),
    ("frei", [1.067, -0.239, 0.815]),
    ("gate", 1, [-0.532, 0.08, -0.094]),
    ("gate", 1, [0.0, 0.0, 0.0]),
    ("gate", 1, [0.211, 0.084, 0.1]),
    ("frei", [0.267, 0.471, 1.291]),  # über die obere Ecke von Gate 0 hinweg
    ("gate", 2, [-0.126, -0.065, 0.006]),
    ("gate", 2, [0.0, 0.0, 0.0]),
    ("gate", 2, [0.151, 0.129, 0.009]),
    ("frei", [-1.232, -0.909, 0.849]),
    ("frei", [-0.997, -1.149, 0.932]),
    ("frei", [-0.471, -1.249, 1.084]),  # unter Pfosten 3 durch
    ("gate", 3, [-0.141, -0.124, 0.008]),
    ("gate", 3, [0.0, 0.0, 0.0]),
    ("gate", 3, [0.449, 0.15, -0.044]),
    ("frei", [1.299, 0.011, 1.186]),
    ("gate", 4, [-0.513, 0.064, 0.013]),
    ("gate", 4, [0.0, 0.0, 0.0]),
    ("gate", 4, [0.519, -0.009, 0.082]),
    ("frei", [1.096, 1.192, 1.438]),  # Auslauf nach dem letzten Gate
]


class Level2Controller(Controller):
    """Folgt einer festen Rennlinie und korrigiert sie lokal, sobald Gates sichtbar werden."""

    # Geschwindigkeitsprofil (die Rennlinie ist für genau diese Werte optimiert)
    V_TOP = 2.5  # Höchstgeschwindigkeit in m/s
    A_LAT = 4.0  # maximale Querbeschleunigung in Kurven in m/s²
    A_LONG = 4.0  # maximale Beschleunigung und Bremsung in m/s²
    V_GATE = 1.8  # Tempolimit im Anflug auf ein Gate, dessen echte Lage noch unbekannt ist, in m/s
    D_SLOW = 0.7  # Länge dieser Anflugzone vor dem Gate in m

    # Korrektur, wenn ein Gate sichtbar wird
    D_HOLD = 0.5  # bis zu diesem Abstand hinter dem Gate bleibt die Korrektur voll aktiv in m
    T_MIN_IN = 0.3  # kürzeste Einblendzeit in s
    T_OUT_MAX = 1.2  # längste Ausblendzeit danach in s (sonst bis kurz vor das nächste Gate)
    ROT_SCALE = 0.5  # Anteil der Gate-Drehung, den die Bahn mitmacht (weniger Pfosten-Treffer)

    # Regelung
    K_POS = 2.0  # zusätzliche Verstärkung auf den Positionsfehler (nur x, y)
    K_VEL = 1.0  # zusätzliche Verstärkung auf den Geschwindigkeitsfehler (nur x, y)
    ACC_LOOKAHEAD = 0.1  # Sollbeschleunigung so weit vorausschicken in s
    FF_SCALE = 1.1  # Faktor auf die Sollbeschleunigung (gleicht die zu kleine Reglermasse aus)
    MAX_POS_BOOST = 0.3  # Begrenzung des Positions-Korrekturterms in m
    MAX_VEL_BOOST = 1.0  # Begrenzung des Geschwindigkeits-Korrekturterms in m/s
    MAX_ACC = 8.0  # Begrenzung der waagerechten Sollbeschleunigung in m/s²
    KI_Z = 1.5  # Integralverstärkung für die Höhe
    Z_INT_MAX = 0.2  # Begrenzung der Höhenkorrektur in m

    # Diagnose
    VERBOSE = True  # bei einem Absturz die Ursache ausgeben

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Rennlinie aufbauen und Geschwindigkeitsprofil berechnen."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        track = config.env.track
        self._nom_gate_pos = np.array([g["pos"] for g in track.gates], dtype=float)
        self._nom_gate_yaw = np.array([g["rpy"][2] for g in track.gates], dtype=float)
        order = np.asarray(track.gate_order)
        self._pass_gate = np.abs(order) - 1  # Gate-Nummer je Durchflug
        self._pass_dir = np.sign(order)  # Durchflugrichtung je Durchflug

        # 1) Wegpunkte aus der Rennlinie
        points, pass_idx = [np.array(obs["pos"], dtype=float)], {}
        for kind, a, *rest in RACING_LINE:
            if kind == "frei":
                points.append(np.array(a, dtype=float))
                continue
            k, offset = a, np.array(rest[0], dtype=float)
            g = self._pass_gate[k]
            offset[0] *= self._pass_dir[k]
            points.append(self._nom_gate_pos[g] + self._yaw_matrix(self._nom_gate_yaw[g]) @ offset)
            if not offset.any():
                pass_idx[k] = len(points) - 1
        points = np.array(points)

        # 2) Form der Bahn: Spline durch die Wegpunkte (Parameter = Länge der Verbindungen)
        seg = np.maximum(np.linalg.norm(np.diff(points, axis=0), axis=1), 1e-3)
        u = np.concatenate([[0.0], np.cumsum(seg)])
        zero = np.zeros(3)
        geo = CubicSpline(u, points, bc_type=((1, zero), (1, zero)))

        # 3) Geschwindigkeitsprofil und Zeit-Spline
        self._pass_u = {k: u[i] for k, i in sorted(pass_idx.items())}
        self._spline, self._t_total = self._time_parametrize(geo, u[-1])
        self._pass_t = {k: float(np.interp(uu, self._s, self._t)) for k, uu in self._pass_u.items()}
        self._windows = self._make_windows()

        self._gate_corr = {}  # Gate -> (Zeitpunkt der Aufdeckung, Drehmatrix, nominal, echt)
        self._z_int = 0.0
        self._tick = 0
        self._finished = False
        self._last_obs = obs
        if self.VERBOSE:
            print(f"[Level2Controller] Rennlinie geladen, geplante Zeit bis zum letzten Gate: "
                  f"{max(self._pass_t.values()):.2f} s")

    @staticmethod
    def _yaw_matrix(yaw: float) -> NDArray[np.floating]:
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _time_parametrize(self, geo: CubicSpline, u_end: float):
        """Zeitoptimales Geschwindigkeitsprofil entlang der festen Bahnform."""
        s = np.append(np.arange(0.0, u_end, 0.005), u_end)
        p, d1, d2 = geo(s), geo(s, 1), geo(s, 2)
        ds = np.maximum(np.linalg.norm(np.diff(p, axis=0), axis=1), 1e-6)
        speed = np.maximum(np.linalg.norm(d1, axis=1), 1e-6)
        kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / speed**3  # Krümmung
        v = np.minimum(self.V_TOP, np.sqrt(self.A_LAT / np.maximum(kappa, 1e-6)))

        # Anflugzonen: vor einem Gate, das bis dahin noch nicht in Sensorreichweite war, langsamer
        for k, uk in self._pass_u.items():
            c = self._nom_gate_pos[self._pass_gate[k]]
            zone = (s < uk) & (np.linalg.norm(p - c, axis=1) < self.D_SLOW)
            zone_start = s[zone].min() if zone.any() else uk
            seen_before = np.any((s < zone_start) & (np.linalg.norm(p[:, :2] - c[:2], axis=1) < 0.6))
            if not seen_before:
                v = np.where(zone, np.minimum(v, self.V_GATE), v)

        v[0] = v[-1] = 0.0  # Start und Ende im Stillstand
        for i in range(len(v) - 1):  # vorwärts: Beschleunigen begrenzen
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * self.A_LONG * ds[i]))
        for i in range(len(v) - 2, -1, -1):  # rückwärts: rechtzeitig bremsen
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * self.A_LONG * ds[i]))
        dt = 2 * ds / np.maximum(v[:-1] + v[1:], 1e-3)
        t = np.concatenate([[0.0], np.cumsum(dt)])

        self._s, self._p, self._t = s, p, t
        keep = np.r_[0, np.arange(4, len(t) - 4, 4), len(t) - 1]  # jeden 4. Punkt, glättet
        zero = np.zeros(3)
        return CubicSpline(t[keep], p[keep], bc_type=((1, zero), (1, zero))), float(t[-1])

    def _make_windows(self) -> dict:
        """Je Durchflug: wann die Korrektur voll aktiv ist, bis wann sie bleibt, wie lange Ausblenden."""
        s, p, t = self._s, self._p, self._t
        ks = sorted(self._pass_t)
        windows = {}
        for k in ks:
            c = self._nom_gate_pos[self._pass_gate[k]]
            after = (s > self._pass_u[k]) & (np.linalg.norm(p - c, axis=1) >= self.D_HOLD)
            t_down = t[after].min() if after.any() else t[-1]
            windows[k] = [self._pass_t[k], t_down, self.T_OUT_MAX]
        for a, b in zip(ks[:-1], ks[1:]):  # Ausblenden muss vor dem nächsten Gate fertig sein
            windows[a][1] = max(min(windows[a][1], windows[b][0] - 0.35), self._pass_t[a])
            windows[a][2] = float(np.clip(windows[b][0] - 0.25 - windows[a][1], 0.1, self.T_OUT_MAX))
        return windows

    @staticmethod
    def _min_jerk(x):
        """Weiche Rampe von 0 auf 1 und ihre ersten beiden Ableitungen nach x."""
        x = np.clip(x, 0.0, 1.0)
        return (
            10 * x**3 - 15 * x**4 + 6 * x**5,
            30 * x**2 - 60 * x**3 + 30 * x**4,
            60 * x - 180 * x**2 + 120 * x**3,
        )

    def _ramp(self, t, t_start, t_full, t_down, t_out):
        """Gewicht 0 -> 1 (t_start bis t_full), 1 bis t_down, dann 1 -> 0 in t_out. Mit Ableitungen."""
        T_in = t_full - t_start
        w_up, w1_up, w2_up = self._min_jerk((t - t_start) / T_in)
        w_dn, w1_dn, w2_dn = self._min_jerk((t - t_down) / t_out)
        up = t < t_full
        w = np.where(up, w_up, 1.0 - w_dn)
        w1 = np.where(up, w1_up / T_in, -w1_dn / t_out)
        w2 = np.where(up, w2_up / T_in**2, -w2_dn / t_out**2)
        return w, w1, w2

    def _reference(self, t):
        """Sollposition, -geschwindigkeit und -beschleunigung inkl. Gate-Korrekturen (t: Array)."""
        t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, self._t_total)
        p, v, a = self._spline(t), self._spline(t, 1), self._spline(t, 2)
        P, V, A = p.copy(), v.copy(), a.copy()
        for k in self._pass_t:
            g = self._pass_gate[k]
            if g not in self._gate_corr:
                continue
            t_rev, M, g_nom, g_true = self._gate_corr[g]
            t_full, t_down, t_out = self._windows[k]
            earlier = [kk for kk in self._pass_t if kk < k]
            # Einblenden erst, wenn das vorherige Gate hinter uns liegt
            t_start = max(t_rev, self._windows[max(earlier)][1] if earlier else 0.0)
            t_full = max(t_full, t_start + self.T_MIN_IN)
            t_down = max(t_down, t_full)
            w, w1, w2 = (x[:, None] for x in self._ramp(t, t_start, t_full, t_down, t_out))
            c = g_true + (p - g_nom) @ M.T - p  # Bahn starr mit dem Gate verschieben und drehen
            c1 = v @ M.T - v
            c2 = a @ M.T - a
            P += w * c
            V += w * c1 + w1 * c
            A += w * c2 + 2 * w1 * c1 + w2 * c
        return P, V, A

    def _update_gates(self, obs, t):
        """Neu aufgedeckte Gates: Korrektur aus echter und nominaler Lage bestimmen."""
        for g in range(len(self._nom_gate_pos)):
            if g in self._gate_corr or not obs["gates_visited"][g]:
                continue
            true_yaw = R.from_quat(obs["gates_quat"][g]).as_euler("xyz")[2]
            dyaw = (true_yaw - self._nom_gate_yaw[g] + np.pi) % (2 * np.pi) - np.pi
            M = self._yaw_matrix(self.ROT_SCALE * dyaw)
            self._gate_corr[g] = (t, M, self._nom_gate_pos[g], np.array(obs["gates_pos"][g]))

    @staticmethod
    def _clip_norm(x, limit):
        n = np.linalg.norm(x)
        return x if n <= limit else x * (limit / n)

############Planer Ende

#############Regler 



    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Sollzustand [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz] berechnen."""
        t = self._tick / self._freq
        self._last_obs = obs
        self._update_gates(obs, t)

        Ps, Vs, As = self._reference([t, t + self.ACC_LOOKAHEAD])
        P, V, A = Ps[0], Vs[0], As[1]  # Beschleunigung aus der nahen Zukunft
        if t >= self._t_total:
            V, A = np.zeros(3), np.zeros(3)
            self._finished = True
        pos, vel = np.asarray(obs["pos"]), np.asarray(obs["vel"])

        # Höhe: Integralanteil gegen bleibende Abweichung
        self._z_int += self.KI_Z * (pos[2] - P[2]) / self._freq
        self._z_int = float(np.clip(self._z_int, -self.Z_INT_MAX, self.Z_INT_MAX))
        cmd_pos = P.copy()
        cmd_pos[2] -= self._z_int

        # Waagerecht: Regler verstärken, Vorsteuerung skalieren und begrenzen
        xy = np.array([1.0, 1.0, 0.0])
        cmd_pos += self._clip_norm(self.K_POS * (P - pos) * xy, self.MAX_POS_BOOST)
        cmd_vel = V + self._clip_norm(self.K_VEL * (V - vel) * xy, self.MAX_VEL_BOOST)
        cmd_acc = A * self.FF_SCALE
        cmd_acc[:2] = self._clip_norm(cmd_acc[:2], self.MAX_ACC)

        quat = np.array([0.0, 0.0, 0.0, 1.0])  # Gierwinkel 0
        return np.concatenate((cmd_pos, cmd_vel, cmd_acc, quat, np.zeros(3)), dtype=np.float32)



#############Reglerende





    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Zeitschritt mitzählen und bei einem Absturz die Ursache ausgeben."""
        self._tick += 1
        n_passed = int(obs["n_gates_passed"])
        if self.VERBOSE and terminated and n_passed < len(self._pass_gate):
            self._print_crash(n_passed)
        return self._finished

    def _print_crash(self, n_passed: int):
        """Ursache eines Absturzes aus der letzten gültigen Beobachtung bestimmen und ausgeben."""
        o = self._last_obs
        t = (self._tick - 1) / self._freq
        pos, speed = np.asarray(o["pos"]), np.linalg.norm(o["vel"])
        err = self._reference(t)[0][0] - pos
        cause = "unklar"
        d_obst = np.linalg.norm(np.asarray(o["obstacles_pos"])[:, :2] - pos[:2], axis=1)
        j = int(np.argmin(d_obst))
        if d_obst[j] < 0.14:
            cause = f"Pfosten {j} getroffen (Abstand {d_obst[j]:.2f} m)"
        elif pos[2] < 0.1:
            cause = "Boden berührt"
        else:
            for g in range(len(self._nom_gate_pos)):
                rel = R.from_quat(o["gates_quat"][g]).inv().apply(pos - np.asarray(o["gates_pos"][g]))
                if abs(rel[0]) < 0.15 and max(abs(rel[1]), abs(rel[2])) < 0.5:
                    where = ("seitlich" if abs(rel[1]) > abs(rel[2])
                             else ("oben" if rel[2] > 0 else "unten"))
                    cause = (f"Rahmen von Gate {g} getroffen ({where}; vor/hinter {rel[0]:+.2f}, "
                             f"seitlich {rel[1]:+.2f}, Höhe {rel[2]:+.2f} m)")
                    break
        seen = ""
        if n_passed < len(self._pass_gate):
            g_next = self._pass_gate[n_passed]
            if g_next in self._gate_corr:
                ahead = self._pass_t[n_passed] - self._gate_corr[g_next][0]
                seen = f", nächstes Gate {g_next} wurde {ahead:.2f} s vor dem Durchflug sichtbar"
        print(f"[Level2Controller] ABSTURZ bei t = {t:.2f} s nach {n_passed} Gates: {cause}. "
              f"Abweichung Soll-Ist {np.linalg.norm(err):.2f} m, Tempo {speed:.1f} m/s{seen}")

    def episode_callback(self):
        """Zurücksetzen."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Sollbahn (grün) und aktuellen Sollpunkt (rot) zeichnen."""
        t = self._tick / self._freq
        draw_line(sim, self._reference(np.linspace(t, self._t_total, 100))[0], rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._reference(t)[0], rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)