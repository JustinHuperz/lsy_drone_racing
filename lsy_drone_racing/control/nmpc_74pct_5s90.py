"""Level-2-Controller: feste Rennlinie, Anpassung an die echte Strecke und NMPC (acados).

Ergebnis in der Simulation (Level 2, 600 neue zufällige Strecken): 74 % Erfolg, im Mittel 5,90 s.
Sicherer, aber langsamer: A_LONG_START = 3.5, THR_ACC_MAX = 14.0 (etwa 81 % bei 5,98 s).

Ablauf:
1. __init__ (einmal): Rennlinie -> glatte Bahn (Spline) -> schnellstes Geschwindigkeitsprofil, das
   Kurven, Schub und Anflugzonen erlaubt -> Sollbahn P(tau).
2. compute_control (50-mal pro Sekunde):
   a) Gate neu aufgedeckt: Bahn in seiner Nähe mit dem Gate mitschieben. Reicht die Zeit bis zum
      Durchflug dafür nicht, läuft die Sollbahn langsamer ab ("Zeitlupe").
   b) Gate oder Pfosten neu aufgedeckt: das nächste Stück Bahn von Rahmen und Pfosten wegbiegen.
   c) NMPC: Sollbahn der nächsten 0,5 s plus Vorsteuerung -> Befehl [roll, pitch, yaw, Schub].

Start: python scripts/sim.py --config level2.toml --controller nmpc_controller.py
Benötigt in config/level2.toml: control_mode = "attitude".
Beim ersten Start wird der Solver kompiliert (Ordner c_generated_code/, etwa eine Minute).
Für die echte Drohne SIM_FIT = False setzen (dann gilt das am echten Crazyflie gemessene Modell).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import casadi as cs
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim.visualize import draw_line, draw_points
from lsy_drone_racing.control import Controller
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# Rennlinie für die Level-2-Strecke, Punkt für Punkt in Flugreihenfolge (Start = Drohnenposition).
# Mit dem Optimierer aus tools/rennlinie für ein schnelleres Profil und mehr Abstand zu den
# Gate-Rahmen neu berechnet (m_own 0.08, m_x 0.03), dann zusammen mit diesem Controller getestet.
#   ("frei", [x, y, z])          Wegpunkt in Weltkoordinaten
#   ("gate", k, [dx, dy, dz])    Punkt relativ zu dem Gate, das beim k-ten Durchflug dran ist
#                                (dx = Durchflugrichtung, dy = seitlich, dz = nach oben).
#                                [0, 0, 0] ist die Gate-Mitte und markiert den Durchflug.
RACING_LINE = [
    ("frei", [-0.344, 0.348, 0.52]),
    ("gate", 0, [-0.309, -0.126, -0.047]),
    ("gate", 0, [0.0, 0.0, 0.0]),
    ("gate", 0, [0.199, -0.01, 0.031]),
    ("frei", [1.062, -0.141, 0.86]),
    ("gate", 1, [-0.521, 0.055, -0.09]),
    ("gate", 1, [0.0, 0.0, 0.0]),
    ("gate", 1, [0.226, 0.149, 0.075]),
    ("frei", [0.158, 0.375, 1.297]),  # über die obere Ecke von Gate 0 hinweg
    ("gate", 2, [-0.301, 0.033, 0.017]),
    ("gate", 2, [0.0, 0.0, 0.0]),
    ("gate", 2, [0.131, 0.088, 0.092]),
    ("frei", [-1.163, -1.039, 0.987]),
    ("frei", [-0.866, -1.145, 1.1]),
    ("frei", [-0.187, -0.93, 1.192]),  # südlich an Pfosten 3 vorbei
    ("gate", 3, [-0.121, -0.066, 0.004]),
    ("gate", 3, [0.0, 0.0, 0.0]),
    ("gate", 3, [0.443, 0.124, -0.099]),
    ("frei", [1.366, 0.168, 1.139]),
    ("gate", 4, [-0.595, 0.131, -0.05]),
    ("gate", 4, [0.0, 0.0, 0.0]),
    ("gate", 4, [0.496, -0.087, 0.099]),
    ("frei", [0.936, 1.245, 1.448]),  # Auslauf nach dem letzten Gate
]

# Kollisionsboxen eines Gates im Gate-System (aus dem Simulationsmodell): Mitte und halbe Größe
GATE_BOX_C = np.array(
    [[0.0, 0.0, 0.28], [0.0, 0.0, -0.28], [0.0, -0.28, 0.0], [0.0, 0.28, 0.0], [0.0, 0.0, -0.86]]
)
GATE_BOX_H = np.array(
    [
        [0.01, 0.36, 0.08],
        [0.01, 0.36, 0.08],
        [0.01, 0.08, 0.36],
        [0.01, 0.08, 0.36],
        [0.05, 0.05, 0.5],
    ]
)
DRONE_HALF = 0.07  # halbe Kantenlänge der Drohne (Box 14 x 14 x 4 cm)
POLE_CONTACT = 0.11  # ab diesem waagerechten Abstand Drohnenmitte <-> Pfostenachse droht Kontakt

_SOLVERS = {}  # ein Solver pro Prozess, wird über alle Läufe wiederverwendet (Kompilieren dauert)


def create_mpc_solver(
    params: dict, n: int, tf: float, q: list, r: list, max_angle: float, t_min: float
) -> AcadosOcpSolver:
    """NMPC mit dem Modell so_rpy plus Luftwiderstand.

    Zustand [pos, rpy, vel, drpy], Eingang [roll, pitch, yaw, Schub].
    Kosten: quadratische Abweichung von Sollzustand und vorgesteuertem Befehl (LINEAR_LS).
    """
    key = (n, tf, tuple(q), tuple(r), max_angle, t_min, tuple(np.ravel(params["drag_acc"])),
           tuple(np.ravel(params["rpy_coef"])), float(params["cmd_f_coef"]))  # fmt: skip
    if key in _SOLVERS:
        return _SOLVERS[key]
    x_dot, x, u, _ = symbolic_dynamics_euler(
        mass=params["mass"],
        gravity_vec=params["gravity_vec"],
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
    )
    drag = cs.DM(np.asarray(params["drag_acc"], dtype=float))  # Luftwiderstand: a = -drag * v
    x_dot = cs.vertcat(x_dot[0:6], x_dot[6:9] - drag * x[6:9], x_dot[9:12])
    model = AcadosModel()
    model.name = "level2_nmpc_" + str(abs(hash(key)) % 10**8)
    model.f_expl_expr, model.x, model.u = x_dot, x, u
    ocp = AcadosOcp()
    ocp.model = model
    nx, nu = x.rows(), u.rows()
    ocp.solver_options.N_horizon = n
    ocp.cost.cost_type = ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(np.diag(q), np.diag(r))
    ocp.cost.W_e = np.diag(q)
    vx = np.zeros((nx + nu, nx))
    vx[:nx, :nx] = np.eye(nx)
    vu = np.zeros((nx + nu, nu))
    vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx, ocp.cost.Vu, ocp.cost.Vx_e = vx, vu, np.eye(nx)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(nx + nu), np.zeros(nx)
    t_lo = max(4 * params["thrust_min"], t_min)
    ocp.constraints.lbu = np.array([-max_angle, -max_angle, -0.5, t_lo])
    ocp.constraints.ubu = np.array([max_angle, max_angle, 0.5, 4 * params["thrust_max"]])
    ocp.constraints.idxbu = np.arange(4)
    ocp.constraints.x0 = np.zeros(nx)
    so = ocp.solver_options
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.hessian_approx = "GAUSS_NEWTON"
    so.integrator_type = "ERK"
    so.nlp_solver_type = "SQP_RTI"  # Echtzeit-Iteration: eine SQP-Iteration pro Regelschritt
    so.qp_solver_warm_start = 1
    so.tf = tf
    solver = AcadosOcpSolver(ocp, json_file=f"c_generated_code/{model.name}.json", verbose=False)
    _SOLVERS[key] = solver
    return solver


class NMPCController(Controller):
    """Rennlinie mit Online-Anpassung an die echte Strecke, geregelt von einem NMPC (Attitude)."""

    # Geschwindigkeitsprofil (die Rennlinie ist für genau diese Werte optimiert)
    V_TOP = 4.0  # Höchstgeschwindigkeit in m/s
    A_LAT = 7.5  # maximale Querbeschleunigung in Kurven in m/s²
    A_LONG = 5.5  # maximale Beschleunigung und Bremsung in m/s²
    A_LONG_START = 4.0  # Beschleunigung vom Start bis zum ersten Gate in m/s²
    THR_ACC_MAX = 14.5  # mehr Schub-Beschleunigung darf das Profil nie verlangen (m/s²) ...
    THR_ACC_MIN = 5.0  # ... und nie weniger (sonst zu wenig Reserve für Abfangbögen)
    V_GATE = 2.0  # Tempolimit im Anflug auf ein Gate, dessen echte Lage noch unbekannt ist, in m/s
    D_SLOW = 0.7  # Länge dieser Anflugzone vor dem Gate in m
    R_SURE = 0.45  # kommt die Bahn einem Gate waagerecht so nah, ist es danach sicher aufgedeckt

    # Gate aufgedeckt: Bahn mitschieben
    ROT_SCALE = 0.5  # Anteil der Gate-Drehung, den die Bahn mitmacht
    D_HOLD = 0.5  # bis zu diesem Abstand hinter dem Gate bleibt die Verschiebung voll aktiv in m
    T_MIN_IN = 0.3  # kürzeste Einblendzeit in s
    T_OUT_MAX = 1.2  # längste Ausblendzeit danach in s (sonst bis kurz vor das nächste Gate)

    # Zeitlupe: braucht das Mitschieben mehr Querbeschleunigung als SLOW_ACC, läuft die Sollbahn
    # bis kurz nach dem Gate langsamer ab
    SLOW_ACC = 5.0  # erlaubte Querbeschleunigung für das Mitschieben in m/s²
    SLOW_RMIN = 0.5  # langsamste Zeitlupe (Anteil der normalen Geschwindigkeit)
    SLOW_TR = 0.12  # Übergangszeit in s
    SLOW_HOLD = 0.05  # so lange nach dem Durchflug noch langsam in s
    SLOW_UP_W = 0.5  # Gewicht der Höhenverschiebung gegenüber der seitlichen

    # Wegbiegen: nach jeder Aufdeckung wird das nächste Stück Bahn so verformt, dass der
    # Drohnenkörper möglichst DEF_MREQ Abstand zu allen Rahmen und Pfosten hat
    DEF_MREQ = 0.08  # gewünschter Abstand in m
    DEF_HOR = 1.4  # so weit voraus wird verformt (s)
    DEF_LOCK = 0.12  # das unmittelbar nächste Stück bleibt unverändert (s)
    DEF_NK = 4  # Anzahl Buckel, aus denen die Verformung besteht
    DEF_MAX = 0.15  # größte Höhe eines Buckels in m (seitlich; nach oben/unten die Hälfte)
    DEF_LAM = 0.02  # Strafe für große Verformungen

    # NMPC
    N = 25  # Schritte im Horizont (25 x 20 ms = 0,5 s)
    Q_POS = (200.0, 200.0, 400.0)  # Gewichte: Abweichung Position x, y, z
    Q_RPY = 1.0  # Abweichung von der Soll-Neigung
    Q_VEL = 10.0  # Abweichung Geschwindigkeit
    Q_DRPY = 1.0  # Drehraten
    R_RPY = 2.0  # Befehl weicht von der vorgesteuerten Neigung ab
    R_THRUST = 50.0  # Befehl weicht vom vorgesteuerten Schub ab
    MAX_ANGLE = 1.0  # größte befohlene Neigung in rad (57°)
    THRUST_MIN_FRAC = 0.5  # kleinster erlaubter Schub als Anteil des Schwebeschubs
    SIM_FIT = True  # an die Simulation angepasstes Lagemodell verwenden (echte Drohne: False)
    CMD_F = 0.946  # Schub-Faktor im Modell
    DRAG = (0.0216, 0.0216, 0.0248)  # Luftwiderstand in N pro m/s (aus der Simulation)

    VERBOSE = True  # bei einem Absturz die Ursache ausgeben

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Rennlinie aufbauen, Geschwindigkeitsprofil berechnen, Solver anlegen."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq
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

        # 4) NMPC
        self._model = dict(load_params(Dynamics.so_rpy, config.sim.drone))  # Modellparameter
        if self.SIM_FIT:  # Lage-Dynamik an die Simulation angepasst (aus Testflügen geschätzt)
            self._model["rpy_coef"] = np.array([-246.0, -246.0, -144.7])
            self._model["rpy_rates_coef"] = np.array([-36.4, -36.4, -17.4])
            self._model["cmd_rpy_coef"] = np.array([241.8, 241.8, 127.0])
            self._model["cmd_f_coef"] = self.CMD_F
        self._model["drag_acc"] = np.array(self.DRAG) / float(self._model["mass"])
        q = [*self.Q_POS, *[self.Q_RPY] * 3, *[self.Q_VEL] * 3, *[self.Q_DRPY] * 3]
        r = [self.R_RPY] * 3 + [self.R_THRUST]
        hover = float(self._model["mass"]) * 9.81 / float(self._model["cmd_f_coef"])
        self._solver = create_mpc_solver(
            self._model,
            self.N,
            self.N * self._dt,
            q,
            r,
            self.MAX_ANGLE,
            self.THRUST_MIN_FRAC * hover,
        )

        # Zustand während des Flugs
        self._tau = 0.0  # Zeit auf der Sollbahn (läuft bei Zeitlupe langsamer als die Uhr)
        self._gate_corr = {}  # Gate -> (Zeit der Aufdeckung, Drehmatrix, nominal, Ziel)
        self._slow = []  # Zeitlupen-Abschnitte (Start, voll, Ende, Übergang, Faktor)
        self._deform = []  # Verformungen (von, bis, Spline der Verschiebung)
        self._known = None  # zuletzt bekannte Aufdeckungen (Gates, Pfosten)
        self._first = True
        self._tick = 0
        self._finished = False
        self._last_obs = obs
        if self.VERBOSE:
            print(f"[NMPCController] geplante Zeit: {max(self._pass_t.values()):.2f} s bis Ziel")

    # ------------------------------------------------------------------------------------------
    # Planung (einmal beim Start)
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _yaw_matrix(yaw: float) -> NDArray[np.floating]:
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _time_parametrize(self, geo: CubicSpline, u_end: float) -> tuple[CubicSpline, float]:
        """Schnellstes Geschwindigkeitsprofil entlang der festen Bahnform, daraus Soll(t)."""
        s = np.append(np.arange(0.0, u_end, 0.005), u_end)
        p, d1, d2 = geo(s), geo(s, 1), geo(s, 2)
        ds = np.maximum(np.linalg.norm(np.diff(p, axis=0), axis=1), 1e-6)
        speed = np.maximum(np.linalg.norm(d1, axis=1), 1e-6)

        # Kurven: Querbeschleunigung v² * Krümmung <= A_LAT
        kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / speed**3
        v = np.minimum(self.V_TOP, np.sqrt(self.A_LAT / np.maximum(kappa, 1e-6)))

        # Schub: |v² * Krümmungsvektor + g| muss zwischen THR_ACC_MIN und THR_ACC_MAX bleiben
        # (wichtig bei Abfangbögen nach unten: dort muss der Schub Kurve UND Schwerkraft tragen)
        tng = d1 / speed[:, None]
        kv = (d2 - np.sum(d2 * tng, axis=1, keepdims=True) * tng) / speed[:, None] ** 2
        k2 = np.maximum(np.sum(kv**2, axis=1), 1e-12)
        gz = 9.81 * kv[:, 2]
        disc = np.maximum(gz**2 - k2 * (9.81**2 - self.THR_ACC_MAX**2), 0.0)
        v2_hi = (-gz + np.sqrt(disc)) / k2
        down = kv[:, 2] < 0
        v2_lo = np.full_like(v2_hi, np.inf)
        v2_lo[down] = (9.81 - self.THR_ACC_MIN) / -kv[down, 2]
        v = np.minimum(v, np.sqrt(np.maximum(np.minimum(v2_hi, v2_lo), 0.0)))

        # Anflugzonen: langsamer vor einem Gate, dessen echte Lage noch nicht sicher bekannt ist
        u_prev = -1.0
        for k, uk in self._pass_u.items():
            g = self._pass_gate[k]
            c = self._nom_gate_pos[g]
            zone = (s < uk) & (s > u_prev) & (np.linalg.norm(p - c, axis=1) < self.D_SLOW)
            u_prev = uk
            zone_start = s[zone].min() if zone.any() else uk
            passed_before = any(self._pass_gate[j] == g for j in self._pass_u if j < k)
            near = np.linalg.norm(p[:, :2] - c[:2], axis=1) < self.R_SURE
            seen_before = np.any((s < zone_start) & near)
            if not (seen_before or passed_before):
                v = np.where(zone, np.minimum(v, self.V_GATE), v)

        # Beschleunigen und Bremsen begrenzen (vorwärts und rückwärts durchgehen)
        v[0] = v[-1] = 0.0
        u0 = self._pass_u[min(self._pass_u)]
        for i in range(len(v) - 1):
            a_long = self.A_LONG_START if s[i] < u0 else self.A_LONG
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_long * ds[i]))
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * self.A_LONG * ds[i]))
        t = np.concatenate([[0.0], np.cumsum(2 * ds / np.maximum(v[:-1] + v[1:], 1e-3))])

        self._s, self._p, self._t = s, p, t
        keep = np.r_[0, np.arange(4, len(t) - 4, 4), len(t) - 1]  # jeden 4. Punkt, glättet
        zero = np.zeros(3)
        return CubicSpline(t[keep], p[keep], bc_type=((1, zero), (1, zero))), float(t[-1])

    def _make_windows(self) -> dict:
        """Je Durchflug: ab wann die Verschiebung voll wirkt, bis wann, wie lange sie ausblendet."""
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
            windows[a][2] = float(
                np.clip(windows[b][0] - 0.25 - windows[a][1], 0.1, self.T_OUT_MAX)
            )
        return windows

    @staticmethod
    def _min_jerk(x: NDArray[np.floating]) -> tuple:
        """Weiche Rampe von 0 auf 1 und ihre ersten beiden Ableitungen nach x."""
        x = np.clip(x, 0.0, 1.0)
        return (
            10 * x**3 - 15 * x**4 + 6 * x**5,
            30 * x**2 - 60 * x**3 + 30 * x**4,
            60 * x - 180 * x**2 + 120 * x**3,
        )

    def _ramp(
        self, t: NDArray, t_start: float, t_full: float, t_down: float, t_out: float
    ) -> tuple:
        """Gewicht 0 -> 1 (t_start bis t_full), 1 bis t_down, dann 1 -> 0 in t_out (+ Ableitung)."""
        t_in = t_full - t_start
        w_up, w1_up, w2_up = self._min_jerk((t - t_start) / t_in)
        w_dn, w1_dn, w2_dn = self._min_jerk((t - t_down) / t_out)
        up = t < t_full
        w = np.where(up, w_up, 1.0 - w_dn)
        w1 = np.where(up, w1_up / t_in, -w1_dn / t_out)
        w2 = np.where(up, w2_up / t_in**2, -w2_dn / t_out**2)
        return w, w1, w2

    # ------------------------------------------------------------------------------------------
    # Sollbahn
    # ------------------------------------------------------------------------------------------
    def _reference(self, t: NDArray | float) -> tuple[NDArray, NDArray, NDArray]:
        """Sollposition, -geschwindigkeit und -beschleunigung (nach tau) inkl. aller Anpassungen."""
        t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, self._t_total)
        p, v, a = self._spline(t), self._spline(t, 1), self._spline(t, 2)
        P, V, A = p.copy(), v.copy(), a.copy()
        # a) Gates: Bahn starr mit dem Gate verschieben und drehen, weich ein- und ausgeblendet
        for k in self._pass_t:
            g = self._pass_gate[k]
            if g not in self._gate_corr:
                continue
            t_rev, M, g_nom, g_target = self._gate_corr[g]
            t_full, t_down, t_out = self._windows[k]
            earlier = [kk for kk in self._pass_t if kk < k]
            # Einblenden erst, wenn das vorherige Gate hinter uns liegt
            t_start = max(t_rev, self._windows[max(earlier)][1] if earlier else 0.0)
            t_full = max(t_full, t_start + self.T_MIN_IN)
            t_down = max(t_down, t_full)
            w, w1, w2 = (x[:, None] for x in self._ramp(t, t_start, t_full, t_down, t_out))
            c = g_target + (p - g_nom) @ M.T - p
            c1 = v @ M.T - v
            c2 = a @ M.T - a
            P += w * c  # Produktregel für die Ableitungen
            V += w * c1 + w1 * c
            A += w * c2 + 2 * w1 * c1 + w2 * c
        # b) Verformungen (von Rahmen und Pfosten weggebogen)
        for ta, tb, sp in self._deform:
            m = (t > ta) & (t < tb)
            if m.any():
                P[m] += sp(t[m])
                V[m] += sp(t[m], 1)
                A[m] += sp(t[m], 2)
        return P, V, A

    def _rate(self, tau: NDArray | float) -> tuple[NDArray, NDArray]:
        """Zeitlupen-Faktor r(tau) = d tau / dt und seine Ableitung dr/dtau."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        r, dr = np.ones_like(tau), np.zeros_like(tau)
        for t0, ta, tb, tr, rs in self._slow:
            w_up, w1_up, _ = self._min_jerk((tau - t0) / (ta - t0))
            w_dn, w1_dn, _ = self._min_jerk((tau - tb) / tr)
            w = np.where(tau < ta, w_up, 1.0 - w_dn)
            w1 = np.where(tau < ta, w1_up / (ta - t0), -w1_dn / tr)
            ri, dri = 1.0 - (1.0 - rs) * w, -(1.0 - rs) * w1
            take = ri < r
            r, dr = np.where(take, ri, r), np.where(take, dri, dr)
        return r, dr

    # ------------------------------------------------------------------------------------------
    # Reaktion auf neu aufgedeckte Gates und Pfosten
    # ------------------------------------------------------------------------------------------
    def _update_gates(self, obs: dict, tau: float):
        """Neu aufgedeckte Gates: Ziel der Verschiebung festlegen, bei Bedarf Zeitlupe planen."""
        for g in range(len(self._nom_gate_pos)):
            if g in self._gate_corr or not obs["gates_visited"][g]:
                continue
            true_yaw = R.from_quat(obs["gates_quat"][g]).as_euler("xyz")[2]
            dyaw = (true_yaw - self._nom_gate_yaw[g] + np.pi) % (2 * np.pi) - np.pi
            M = self._yaw_matrix(self.ROT_SCALE * dyaw)
            self._gate_corr[g] = (tau, M, self._nom_gate_pos[g], np.array(obs["gates_pos"][g]))
            self._plan_slowdown(g, tau)

    def _plan_slowdown(self, g: int, tau: float):
        """Reicht die Zeit bis zum Durchflug für die Verschiebung nicht: Zeitlupe bis danach."""
        _, M, g_nom, g_target = self._gate_corr[g]
        lateral = self._yaw_matrix(self._nom_gate_yaw[g])[:, 1]  # seitliche Richtung des Gates
        for k, tp in self._pass_t.items():
            if self._pass_gate[k] != g or tp <= tau + 0.04:
                continue
            pts = self._spline(np.array([tp, max(tau, tp - 0.3)]))
            c = g_target + (pts - g_nom) @ M.T - pts
            shift = max(np.abs(c @ lateral).max(), self.SLOW_UP_W * np.abs(c[:, 2]).max())
            t_req = np.sqrt(
                5.8 * shift / self.SLOW_ACC
            )  # Min-Jerk: Spitzenbeschl. = 5,8 * Weg / T²
            if t_req > tp - tau:
                rs = max(self.SLOW_RMIN, (tp - tau) / t_req)
                self._slow.append((tau, tau + self.SLOW_TR, tp + self.SLOW_HOLD, self.SLOW_TR, rs))

    def _make_clearance(self, obs: dict) -> Callable[[NDArray], NDArray]:
        """Abstandsfunktion Drohnenkörper <-> alle Rahmen und Pfosten (Stand des Wissens: obs)."""
        gates = []
        for g in range(len(obs["gates_pos"])):
            rg = R.from_quat(obs["gates_quat"][g]).as_matrix()
            phi = -np.arctan2(rg[1, 0], rg[0, 0])  # Drohne (Gier 0) im Gate-System
            d1 = np.array([np.cos(phi), np.sin(phi)])
            d2 = np.array([-np.sin(phi), np.cos(phi)])
            axes = np.array([[1.0, 0.0], [0.0, 1.0], d1, d2])  # Trennachsen (Satz der Trennachsen)
            reach = (
                GATE_BOX_H[:, :1] * np.abs(axes[:, 0])
                + GATE_BOX_H[:, 1:2] * np.abs(axes[:, 1])
                + DRONE_HALF * (np.abs(axes @ d1) + np.abs(axes @ d2))
            )  # (5 Boxen, 4 Achsen)
            gates.append((np.asarray(obs["gates_pos"][g], dtype=float), rg, axes, reach))
        poles = np.asarray(obs["obstacles_pos"], dtype=float)[:, :2]

        def clearance(P: NDArray) -> NDArray:
            clr = np.linalg.norm(P[..., None, :2] - poles, axis=-1).min(axis=-1) - POLE_CONTACT
            for c, rg, axes, reach in gates:
                rel = (P - c) @ rg  # ins Gate-System
                gap = np.abs((rel[..., None, :2] - GATE_BOX_C[:, :2]) @ axes.T)  # (..., 5, 4)
                depth_xy = (reach - gap).min(axis=-1)
                depth_z = (GATE_BOX_H[:, 2] + 0.03) - np.abs(rel[..., None, 2] - GATE_BOX_C[:, 2])
                clr = np.minimum(clr, -np.minimum(depth_xy, depth_z).max(axis=-1))
            return clr

        return clearance

    def _optimize_deformation(self, obs: dict, tau0: float):
        """Das nächste Stück der Sollbahn so verbiegen, dass der Abstand zu Hindernissen wächst.

        Die Verformung ist eine Summe von DEF_NK glatten Buckeln, jeder seitlich (waagerecht quer
        zur Flugrichtung) und vertikal. Ihre Höhen sucht ein kurzes Gradientenverfahren.
        """
        ta = tau0 + self.DEF_LOCK
        tb = min(tau0 + self.DEF_HOR, self._t_total)
        if tb - ta < 0.3:
            return
        ts = np.linspace(ta, tb, max(int((tb - ta) / 0.04) + 1, 8))
        P0, V0, _ = self._reference(ts)
        clearance = self._make_clearance(obs)
        if clearance(P0).min() >= self.DEF_MREQ:
            return
        side = np.stack([-V0[:, 1], V0[:, 0], np.zeros(len(ts))], 1)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-6)
        up = np.array([0.0, 0.0, 1.0])
        nk = self.DEF_NK
        width = (tb - ta) / (nk + 1)
        x = (ts[None, :] - (ta + width * np.arange(1, nk + 1))[:, None]) / width
        bump = np.where(np.abs(x) < 1, (1 - x**2) ** 3, 0.0)  # (nk, n), glatt, 0 an den Rändern
        basis = np.concatenate(
            [bump[:, :, None] * side[None], bump[:, :, None] * up[None, None]], 0
        )
        lim = np.r_[np.full(nk, self.DEF_MAX), np.full(nk, 0.5 * self.DEF_MAX)]

        def cost(heights: NDArray) -> NDArray:  # für mehrere Kandidaten auf einmal
            clr = clearance(P0[None] + np.einsum("bk,knd->bnd", heights, basis))
            lack = np.maximum(self.DEF_MREQ - clr, 0.0)
            return np.mean(lack**2, axis=1) + self.DEF_LAM * np.mean(heights**2, axis=1)

        xk = np.zeros(2 * nk)
        fk = f0 = cost(xk[None])[0]
        steps = 0.03 * 2.0 ** np.arange(-4, 2)
        for _ in range(15):  # Gradient (Differenzenquotient), dann beste von 6 Schrittweiten
            fv = cost(np.vstack([xk, xk + 0.004 * np.eye(2 * nk)]))
            grad = (fv[1:] - fv[0]) / 0.004
            gn = np.linalg.norm(grad)
            if gn < 1e-9:
                break
            cand = np.clip(xk[None] - steps[:, None] * (grad / gn)[None], -lim, lim)
            fc = cost(cand)
            j = int(np.argmin(fc))
            if fc[j] >= fk:
                break
            xk, fk = cand[j], fc[j]
        if fk >= f0 - 1e-9:
            return
        shift = np.einsum("k,knd->nd", xk, basis)
        dt = ts[1] - ts[0]
        tt = np.r_[ts[0] - 3 * dt : ts[0] - 0.5 * dt : dt, ts, ts[-1] + dt : ts[-1] + 3.5 * dt : dt]
        shift = np.vstack([np.zeros((3, 3)), shift, np.zeros((3, 3))])
        self._deform.append((ta, tb, CubicSpline(tt, shift, bc_type="clamped")))

    # ------------------------------------------------------------------------------------------
    # Regelung (50 Hz)
    # ------------------------------------------------------------------------------------------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """NMPC: Befehl [roll, pitch, yaw, Schub] so, dass die Drohne der Sollbahn folgt."""
        tau = self._tau
        self._last_obs = obs
        self._update_gates(obs, tau)
        known = (
            tuple(np.asarray(obs["gates_visited"], bool)),
            tuple(np.asarray(obs["obstacles_visited"], bool)),
        )
        if known != self._known:  # etwas Neues aufgedeckt -> Bahn ggf. wegbiegen
            self._known = known
            self._optimize_deformation(obs, tau)
        if tau >= self._t_total:
            self._finished = True
        pos, vel = np.asarray(obs["pos"]), np.asarray(obs["vel"])
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((pos, rpy, vel, drpy))

        # Sollbahn für den Horizont: Zeitpunkte auf der Bahn mit Zeitlupe weiterzählen
        ts = np.empty(self.N + 1)
        ts[0] = tau
        for j in range(self.N):
            ts[j + 1] = ts[j] + self._dt * float(self._rate(ts[j])[0][0])
        P, V, A = self._reference(ts)
        r, dr = self._rate(ts)
        A = r[:, None] ** 2 * A + (r * dr)[:, None] * V  # Kettenregel: d/dt = r * d/dtau
        V = r[:, None] * V

        # Vorsteuerung: Neigung und Schub, die genau die Sollbeschleunigung erzeugen
        m = self._model
        f = (
            A + np.array([0.0, 0.0, 9.81]) + m["drag_acc"] * V
        )  # Schub muss auch den Widerstand tragen
        fn = np.linalg.norm(f, axis=1)
        b3 = f / fn[:, None]
        roll = -np.arcsin(np.clip(b3[:, 1], -1, 1))
        pitch = np.arctan2(b3[:, 0], b3[:, 2])
        rpy_ref = np.stack([roll, pitch, np.zeros_like(roll)], axis=1)
        cmd_rpy = rpy_ref * (-np.asarray(m["rpy_coef"]) / np.asarray(m["cmd_rpy_coef"]))
        thrust = float(m["mass"]) * fn / float(m["cmd_f_coef"])

        s = self._solver
        s.set(0, "lbx", x0)
        s.set(0, "ubx", x0)
        if self._first:  # Startwerte für den ersten Schritt
            for j in range(self.N + 1):
                s.set(j, "x", x0)
            for j in range(self.N):
                s.set(j, "u", np.r_[cmd_rpy[j], thrust[j]])
            self._first = False
        else:  # Lösung des letzten Schritts um einen Schritt verschoben als Startwert
            for j in range(self.N - 1):
                s.set(j, "x", s.get(j + 1, "x"))
                s.set(j, "u", s.get(j + 1, "u"))
            s.set(self.N - 1, "x", s.get(self.N, "x"))
        zeros = np.zeros(3)
        for j in range(self.N):
            s.set(
                j, "yref", np.concatenate((P[j], rpy_ref[j], V[j], zeros, cmd_rpy[j], [thrust[j]]))
            )
        s.set(self.N, "yref", np.concatenate((P[self.N], rpy_ref[self.N], V[self.N], zeros)))
        s.solve()
        self._tau += self._dt * float(self._rate(self._tau)[0][0])
        return np.asarray(s.get(0, "u"), dtype=np.float32)

    # ------------------------------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------------------------------
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
        pos, speed = np.asarray(o["pos"]), np.linalg.norm(o["vel"])
        err = np.linalg.norm(self._reference(self._tau)[0][0] - pos)
        cause = "unklar"
        d_obst = np.linalg.norm(np.asarray(o["obstacles_pos"])[:, :2] - pos[:2], axis=1)
        j = int(np.argmin(d_obst))
        if d_obst[j] < 0.14:
            cause = f"Pfosten {j} getroffen (Abstand {d_obst[j]:.2f} m)"
        elif pos[2] < 0.1:
            cause = "Boden berührt"
        else:
            for g in range(len(self._nom_gate_pos)):
                rel = (
                    R.from_quat(o["gates_quat"][g]).inv().apply(pos - np.asarray(o["gates_pos"][g]))
                )
                if abs(rel[0]) < 0.15 and max(abs(rel[1]), abs(rel[2])) < 0.5:
                    where = (
                        "seitlich"
                        if abs(rel[1]) > abs(rel[2])
                        else ("oben" if rel[2] > 0 else "unten")
                    )
                    cause = f"Rahmen von Gate {g} getroffen ({where}, seitlich {rel[1]:+.2f} m, "
                    cause += f"Höhe {rel[2]:+.2f} m)"
                    break
        t = self._tick / self._freq
        print(f"[NMPCController] ABSTURZ bei t = {t:.2f} s nach {n_passed} Gates: {cause}. "
              f"Abweichung Soll-Ist {err:.2f} m, Tempo {speed:.1f} m/s")  # fmt: skip

    def episode_callback(self):
        """Zurücksetzen."""
        self._tick = 0
        self._tau = 0.0

    def render_callback(self, sim: Sim):
        """Sollbahn (grün) und aktuellen Sollpunkt (rot) zeichnen."""
        tau = self._tau
        draw_line(
            sim, self._reference(np.linspace(tau, self._t_total, 100))[0], rgba=(0.0, 1.0, 0.0, 1.0)
        )
        draw_points(sim, self._reference(tau)[0], rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
