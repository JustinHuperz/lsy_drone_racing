"""Level-2-Controller mit NMPC (Prototyp).

Aufbau:
1. Planer (unverändert aus my_controller.py): feste, optimierte Rennlinie, zeitoptimales
   Geschwindigkeitsprofil und weiche Korrektur, sobald ein Gate sichtbar wird -> _reference(t).
2. Regler (neu): ein NMPC mit acados im Attitude-Modus. Er bekommt die Sollbahn der nächsten 0,5 s
   und rechnet daraus Neigung und Schub [roll, pitch, yaw, Schub]. Sein Modell ist so_rpy aus
   crazyflow mit zwei Anpassungen an die Simulation: Luftwiderstand und gemessene Lage-Dynamik.
   Für die echte Drohne SIM_FIT = False setzen (dann gilt das am echten Crazyflie gemessene Modell).

Stand (300 Läufe Level 2): etwa 6,8 s bei etwa 70 % Erfolg. Schneller (A_LONG_START 2.5,
V_GATE_PASS nur für Gate 3): etwa 6,5 s bei etwa 55 %. Häufigste Absturzstelle: Anflug auf Gate 3.

Benötigt in config/level2.toml: control_mode = "attitude".
Beim ersten Start wird der Solver kompiliert (Ordner c_generated_code/, etwa eine Minute).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# Rennlinie für die Level-2-Strecke (unverändert aus my_controller.py), Punkt für Punkt in Flugreihenfolge (Start = Drohnenposition):
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


_SOLVERS = {}  # ein Solver pro Prozess, wird über alle Läufe wiederverwendet (Kompilieren dauert)

##Kopfzeile: Nimmt die Modellparameter (params), die Anzahl der vorausschauenden Schritte im Horizont (N), die Horizontzeit in Sekunden (Tf), die Gewichtungsmatrizen (Q für Zustände, R_ für Stellgrößen), den maximalen Neigungswinkel (max_angle), den Flag für Real-Time Iterations (rti) sowie den Mindestschub (t_min) entgegen.
##1.Physikalisches Systemmodell: kontinuierliche Systemdynamik $\dot{x} = f(x, u)$ als CasADi-Ausdruck (Position, Orientierung, Geschwindigkeit, Drehraten) und erweitern sie um ein lineares Dämpfungsmodell für den aerodynamischen Widerstand.
##2.Optimal Control Problem (OCP): quadratisches Gütekriterium (Linear Least Squares) , das Abweichungen vom Sollzustand mit der Matrix $Q$ und Stellgrößenaufwand mit $R$ über den Prädiktionshorizont $N=25$ ($T_f = 0{,}5\text{ s}$) gewichtet.
##3.Sicherheit & Stellgrößenbegrenzungen (Constraints):Einschränken des Lösungsraums auf die physikalisch machbaren Aktuatorgrenzen der Drohne (maximale Neigungswinkel und Schubgrenzen der 4 Motoren).
##4.Solver-Kompilierung in C:„Um die Rechenzeit unter $2\text{ ms}$ zu halten, nutzen wir acados mit RTI (Real-Time Iteration) und HPIPM als QP-Solver, der das Problem in C-Code übersetzt.“
def create_mpc_solver(params: dict, N: int, Tf: float, Q, R_, max_angle: float, rti: bool, t_min: float = 0.0):
    """NMPC mit dem identifizierten Modell so_rpy: Zustand [pos, rpy, vel, drpy], Eingang [r, p, y, Schub]."""
    key = (N, Tf, tuple(Q), tuple(R_), max_angle, rti, t_min, tuple(np.ravel(params.get('drag_acc', np.zeros(3)))), tuple(np.ravel(params['rpy_coef'])), tuple(np.ravel(params['rpy_rates_coef'])), float(params['cmd_f_coef']))
    if key in _SOLVERS:
        return _SOLVERS[key]
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=params["mass"], gravity_vec=params["gravity_vec"], acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"], rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"], cmd_rpy_coef=params["cmd_rpy_coef"],
    )
    drag = np.asarray(params.get("drag_acc", np.zeros(3)), dtype=float)  # Luftwiderstand: a = -drag * v
    if np.any(drag):
        X_dot = cs.vertcat(X_dot[0:6], X_dot[6:9] - cs.DM(drag) * X[6:9], X_dot[9:12])
    model = AcadosModel()
    model.name = "level2_mpc_" + str(abs(hash(key)) % 10**8)
    model.f_expl_expr, model.x, model.u = X_dot, X, U
    ocp = AcadosOcp()
    ocp.model = model
    nx, nu = X.rows(), U.rows()
    ny = nx + nu
    ocp.solver_options.N_horizon = N
    ocp.cost.cost_type = ocp.cost.cost_type_e = "LINEAR_LS" ##Kostenfunktions-Typ: Wählt für die Zwischenschritte (cost_type) und den Endschritt (cost_type_e) ein lineares Kleinste-Quadrate-Funktional (Linear Least Squares).
    ocp.cost.W = scipy.linalg.block_diag(np.diag(Q), np.diag(R_))
    ocp.cost.W_e = np.diag(Q)
    Vx = np.zeros((ny, nx)); Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu)); Vu[nx:, :] = np.eye(nu) ##Vx und Vu ordnen die Zustände $x$ und Stellgrößen $u$ dem kombinierten Ausgabevektor $y = [x^T, u^T]^T$ zu.
    ocp.cost.Vx, ocp.cost.Vu, ocp.cost.Vx_e = Vx, Vu, np.eye(nx)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(ny), np.zeros(nx)
    ocp.constraints.lbu = np.array([-max_angle, -max_angle, -0.5, max(4 * params["thrust_min"], t_min)])
    ocp.constraints.ubu = np.array([max_angle, max_angle, 0.5, 4 * params["thrust_max"]])
    ocp.constraints.idxbu = np.arange(4)
    ocp.constraints.x0 = np.zeros(nx)
    so = ocp.solver_options
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.hessian_approx = "GAUSS_NEWTON"  ##Gauss-Newton-Approximation für beschleunigte Konvergenz.
    so.integrator_type = "ERK"          ##Explizites Runge-Kutta-Verfahren (4. Ordnung) zur Simulation der Flugphysik innerhalb des Horizonts.
    so.nlp_solver_type = "SQP_RTI" if rti else "SQP"
    so.nlp_solver_max_iter = 20         ##(Real-Time Iteration). Führt nur eine einzige SQP-Iteration pro Zeitschritt aus, um extrem schnelle Antwortzeiten <20ms zu garantieren.
    so.qp_solver_warm_start = 1
    so.tf = Tf                          ##Prädiktionshorizont (0,5 Sekunden).
    solver = AcadosOcpSolver(ocp, json_file=f"c_generated_code/{model.name}.json", verbose=False)
    _SOLVERS[key] = solver
    return solver


class MPCController(Controller):
    """Folgt einer festen Rennlinie und korrigiert sie lokal, sobald Gates sichtbar werden."""

    # Geschwindigkeitsprofil (die Rennlinie ist für genau diese Werte optimiert)
    V_TOP = 3.0  # Höchstgeschwindigkeit in m/s
    A_LAT = 5.5  # maximale Querbeschleunigung in Kurven in m/s² (alter Regler: 3.5)
    A_LONG = 4.0  # maximale Beschleunigung und Bremsung in m/s²
    V_GATE = 2.2  # Tempolimit im Anflug auf ein Gate, dessen echte Lage noch unbekannt ist, in m/s
    D_SLOW = 0.7  # Länge dieser Anflugzone vor dem Gate in m

    # Korrektur, wenn ein Gate sichtbar wird
    D_HOLD = 0.5  # bis zu diesem Abstand hinter dem Gate bleibt die Korrektur voll aktiv in m
    T_MIN_IN = 0.3  # kürzeste Einblendzeit in s
    T_OUT_MAX = 1.2  # längste Ausblendzeit danach in s (sonst bis kurz vor das nächste Gate)

    # Höhe
    KI_Z = 0.0  # Integralverstärkung für die Höhe (beim NMPC aus, sonst Überschwingen beim Start)
    Z_INT_MAX = 0.2  # Begrenzung der Höhenkorrektur in m

    ROT_SCALE = 0.5  # Anteil der Gate-Drehung, den die Bahn mitmacht

    # NMPC
    N = 25  # Schritte im Horizont (25 x 20 ms = 0,5 s)
    Q_POS = (200.0, 200.0, 400.0)  # Gewichte: Abweichung Position x, y, z
    Q_RPY = 1.0  # Abweichung von der Soll-Neigung
    Q_VEL = 10.0  # Abweichung Geschwindigkeit
    Q_DRPY = 1.0  # Drehraten
    R_RPY = 2.0  # Befehl weicht von der vorgesteuerten Neigung ab
    R_THRUST = 50.0  # Befehl weicht vom vorgesteuerten Schub ab
    MAX_ANGLE = 0.7  # größte befohlene Neigung in rad (40°)
    RTI = True  # Echtzeit-Iteration: eine SQP-Iteration pro Schritt
    T_SHIFT = 0.0  # Sollbahn um so viele Schritte vorziehen (gleicht Verzögerung aus)
    SIM_FIT = True  # an die Simulation angepasstes Lagemodell verwenden
    CMD_F = 0.946  # Schub-Faktor im Modell
    DRAG = (0.0216, 0.0216, 0.0248)  # Luftwiderstand in N pro m/s (aus der Simulation)
    THRUST_MIN_FRAC = 0.5  # kleinster erlaubter Schub als Anteil des Schwebeschubs
    V_GATE_PASS = {"0": 1.5, "3": 1.5}  # eigenes Anflugtempo je Durchflug (langsamer vor Gate 0 und 3)
    A_LONG_START = 2.0  # Beschleunigung vom Start bis zum ersten Gate in m/s² (sanfter Start)

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
        self._p = dict(load_params(Dynamics.so_rpy, config.sim.drone))
        if self.SIM_FIT:  # Lage-Dynamik an die Simulation angepasst (aus Testflügen geschätzt)
            self._p["rpy_coef"] = np.array([-246.0, -246.0, -144.7])
            self._p["rpy_rates_coef"] = np.array([-36.4, -36.4, -17.4])
            self._p["cmd_rpy_coef"] = np.array([241.8, 241.8, 127.0])
            self._p["cmd_f_coef"] = self.CMD_F
        # Luftwiderstand der Simulation (first_principles: Kraft = -0.0216 N/(m/s) * v)
        self._p["drag_acc"] = np.array(self.DRAG) / float(self._p["mass"])
        Q = [*self.Q_POS, *[self.Q_RPY] * 3, *[self.Q_VEL] * 3, *[self.Q_DRPY] * 3]
        R_ = [self.R_RPY] * 3 + [self.R_THRUST]
        self._dt = 1.0 / self._freq
        self._solver = create_mpc_solver(self._p, self.N, self.N * self._dt, Q, R_,
                                         self.MAX_ANGLE, self.RTI,
                                         self.THRUST_MIN_FRAC * float(self._p["mass"]) * 9.81 / float(self._p["cmd_f_coef"]))
        self._first = True
        self._z_int = 0.0
        self._tick = 0
        self._finished = False
        self._last_obs = obs
        if self.VERBOSE:
            print(f"[MPCController] Rennlinie geladen, geplante Zeit bis zum letzten Gate: "
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
                vg = self.V_GATE_PASS.get(str(k), self.V_GATE) if self.V_GATE_PASS else self.V_GATE
                v = np.where(zone, np.minimum(v, vg), v)

        v[0] = v[-1] = 0.0  # Start und Ende im Stillstand
        u0 = self._pass_u[min(self._pass_u)]  # bis zum ersten Gate sanfter beschleunigen (Start)
        for i in range(len(v) - 1):  # vorwärts: Beschleunigen begrenzen
            a_long = self.A_LONG_START if s[i] < u0 else self.A_LONG
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_long * ds[i]))
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


    ##Absolutes Minimum an Ruck ($\dddot{x} \to \text{min}$)Mathematisch bewiesen die schonendste Bewegungsform.Minimiert die 3. Ableitung der Position (den Ruck), wodurch Schläge und abrupte Krafteinwirkungen auf Motoren, Regler und den Drohnenrahmen verhindert werden.Perfekte Übergänge an allen Rändern (6 Randbedingungen)Erfüllt an beiden Enden ($x=0$ und $x=1$) exakt 6 physische Nullstellen:Position: Verbindet Start- und Zielwert punktgenau.Geschwindigkeit: Startet und endet mit $0$ (kein Geschwindigkeits-Sprung).Beschleunigung: Startet und endet mit $0$ (keine plötzliche Kraftspitze).Das niedrigste mögliche Polynom (Grad 5)Es ist das einzige Polynom niedrigsten Grades, das alle 6 Randbedingungen gleichzeitig erfüllt.Höhere Polynome (z. B. Grad 7) sind rechenintensiver, während niedrigere Polynome (z. B. Grad 3 oder 4) Knicke und Kraftsprünge an den Rändern hinterlassen.
    ##mathematische basis 
    ##der mathematische Ruck (engl. Jerk), also die 3. Ableitung der Position nach der Zeit ($\dddot{x}$).
    ##$$w(0) = 0, \quad w(1) = 1$$$$\dot{w}(0) = 0, \quad \dot{w}(1) = 0 \quad \text{(Anfang/Ende in Ruhe)}$$$$\ddot{w}(0) = 0, \quad \ddot{w}(1) = 0 \quad \text{(keine plötzliche Kraft)}$$
    @staticmethod
    def _min_jerk(x):
        """Weiche Rampe von 0 auf 1 und ihre ersten beiden Ableitungen nach x."""
        x = np.clip(x, 0.0, 1.0)
        return (
            10 * x**3 - 15 * x**4 + 6 * x**5,
            30 * x**2 - 60 * x**3 + 30 * x**4,
            60 * x - 180 * x**2 + 120 * x**3,
        )

    ##nutzt mathematische basis der min_jerk funktion und wendet es auf das Physikalisches Zeitfenster an -> steuert den gesamten Lebenszyklus einer Korrektur (Einblenden $\rightarrow$ Halten $\rightarrow$ Ausblenden)
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

    ##_reference nimmt dann w,w1,w2 aus _ramp und das kam ja aus _min_jerk
    ###über reference (online anpassung lokal) passt er die offline geplante Bahn (racing line) an die echten Gate-Positionen an, sobald diese sichtbar werden. (0.7m)
    def _reference(self, t):
        ##clip: Stellt sicher, dass Zeit ($t$) nicht vor Start ($0.0$) oder nach Rennende 
        ##self._spline liefert für Zeitpunkt ($t$)  ungeänderte Offline-Position ($p$). Die 1. Ableitung ist die Soll-Geschwindigkeit ($v$), die 2. Ableitung die Soll-Beschleunigung ($a$)
        """Sollposition, -geschwindigkeit und -beschleunigung inkl. Gate-Korrekturen (t: Array)."""
        t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, self._t_total)
        p, v, a = self._spline(t), self._spline(t, 1), self._spline(t, 2)
        P, V, A = p.copy(), v.copy(), a.copy()
        for k in self._pass_t:
            g = self._pass_gate[k]
            if g not in self._gate_corr:
                continue
            t_rev, M, g_nom, g_true = self._gate_corr[g]  ##t_rev: Zeitpunkt, an dem das Gate aufgedeckt wurde.
            t_full, t_down, t_out = self._windows[k]
            earlier = [kk for kk in self._pass_t if kk < k]
            # Einblenden erst, wenn das vorherige Gate hinter uns liegt -> t_start: Die Korrektur beginnt erst, wenn das vorherige Gate sicher passiert/abgeschlossen ist (oder sofort bei Sichtung).
            t_start = max(t_rev, self._windows[max(earlier)][1] if earlier else 0.0)
            t_full = max(t_full, t_start + self.T_MIN_IN) ##t_full: Zeitpunkt, ab dem die Korrektur zu 100 % wirkt. Um abrupte Sprünge zu vermeiden, wird mindestens T_MIN_IN (0.3 s) Einblendzeit eingeplant.
            t_down = max(t_down, t_full)             ##t_down / t_out: Zeitpunkt, ab dem die Korrektur nach dem Durchflug wieder sanft ausgeblendet wird.
            ##$w$ (Gewicht): Ein Wert zwischen $0.0$ (keine Korrektur) und $1.0$ (volle Korrektur).
            ##$w_1, w_2$: Die 1. und 2. Ableitung dieser Übergangsfunktion nach der Zeit ($\dot{w}$ und $\ddot{w}$). Sie basieren auf einem Minimum-Jerk-Polynom ($10x^3 - 15x^4 + 6x^5$), damit der Übergang ruckfrei verläuft.
            w, w1, w2 = (x[:, None] for x in self._ramp(t, t_start, t_full, t_down, t_out))
            ##c (Positionsversatz): Nimmt den Punkt relativ zum Soll-Gate $(p - g_{nom})$, dreht ihn um die Drehmatrix $M$ und platziert ihn am realen Gate $g_{true}$. Abzüglich der ursprünglichen Position $p$ ergibt sich der reine Korrekturvektor $\Delta p$.
            c = g_true + (p - g_nom) @ M.T - p       # Bahn starr mit dem Gate verschieben und drehen
            c1 = v @ M.T - v                         ##c1 & c2: Drehen auch die Geschwindigkeits- und Beschleunigungsvektoren entsprechend mit.
            c2 = a @ M.T - a
            P += w * c                               ##Da $P(t) = p(t) + w(t) \cdot c(t)$ gilt, greift beim Ableiten die Produktregel:Position: ## $P = p + w \cdot c$
            V += w * c1 + w1 * c                     ## Geschwindigkeit: $\dot{P} = \dot{p} + w \cdot \dot{c} + \dot{w} \cdot c = v + w \cdot c_1 + w_1 \cdot c$
            A += w * c2 + 2 * w1 * c1 + w2 * c       ## Beschleunigung: $\ddot{P} = a + w \cdot c_2 + 2 w_1 c_1 + w_2 c$
        return P, V, A

    def _update_gates(self, obs, t):
        """Neu aufgedeckte Gates: Korrektur aus echter und nominaler Lage bestimmen."""
        for g in range(len(self._nom_gate_pos)):
            if g in self._gate_corr or not obs["gates_visited"][g]: ##g in self._gate_corr: Wenn für Gate bereits Korrektur berechnet,nicht noch einmal berechnen, not obs["gates_visited"][g]: Gate wurde vom Sensor noch nicht gesichtet/aufgedeckt (noch mehr als 0.7m entfernt),
                continue 
            true_yaw = R.from_quat(obs["gates_quat"][g]).as_euler("xyz")[2] 
            dyaw = (true_yaw - self._nom_gate_yaw[g] + np.pi) % (2 * np.pi) - np.pi  ##Uns interessiert nur die Drehung um die Z-Achse ([2] = Yaw / Gierwinkel). ##direkt inklusive modulo-Operation, um den Winkel in den Bereich [-π, π] zu bringen (kürzester Drehwinkel).
            M = self._yaw_matrix(self.ROT_SCALE * dyaw) ##Gedämpfte Drehung: Der Code übernimmt nicht zwingend die volle Drehung des Gates auf die Flugbahn, sondern skaliert sie mit self.ROT_SCALE (im Code meist 0.5). Das bedeutet: Die Flugbahn dreht sich nur zu 50 % mit dem Gate mit, damit die Kurve vor/nach dem Gate nicht zu extrem verbogen wird.self._yaw_matrix(...): Erzeugt eine standardmäßige $3 \times 3$ Rotationsmatrix für eine Drehung um die Z-Achse im Raum.
            self._gate_corr[g] = (t, M, self._nom_gate_pos[g], np.array(obs["gates_pos"][g]))


    ##wirkliche umsetzung des MPC
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """NMPC: Befehl [roll, pitch, yaw, Schub] so, dass die Drohne der Sollbahn folgt."""
        t = self._tick / self._freq
        self._last_obs = obs
        self._update_gates(obs, t)
        if t >= self._t_total:
            self._finished = True
        pos, vel = np.asarray(obs["pos"]), np.asarray(obs["vel"])
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((pos, rpy, vel, drpy))

        ts = t + self._dt * (np.arange(self.N + 1) + self.T_SHIFT)
        P, V, A = self._reference(ts)
        # Höhe: Integralanteil gegen bleibende Abweichung (z. B. andere Masse)
        self._z_int += self.KI_Z * (pos[2] - P[0, 2]) / self._freq
        self._z_int = float(np.clip(self._z_int, -self.Z_INT_MAX, self.Z_INT_MAX))
        P[:, 2] -= self._z_int

        # Vorsteuerung: Neigung und Schub, die genau die Sollbeschleunigung erzeugen
        p = self._p
        f = A + np.array([0.0, 0.0, 9.81]) + self._p["drag_acc"] * V  # Schub muss auch den Widerstand ausgleichen
        fn = np.linalg.norm(f, axis=1)
        b3 = f / fn[:, None]
        roll = -np.arcsin(np.clip(b3[:, 1], -1, 1))
        pitch = np.arctan2(b3[:, 0], b3[:, 2])
        rpy_ref = np.stack([roll, pitch, np.zeros_like(roll)], axis=1)
        cmd_rpy = rpy_ref * (-np.asarray(p["rpy_coef"]) / np.asarray(p["cmd_rpy_coef"]))
        thrust = float(p["mass"]) * fn / float(p["cmd_f_coef"])

        s = self._solver
        s.set(0, "lbx", x0)
        s.set(0, "ubx", x0)
        if self._first:  # Startwerte für den ersten Schritt
            for j in range(self.N + 1):
                s.set(j, "x", x0)
            for j in range(self.N):
                s.set(j, "u", np.r_[cmd_rpy[j], thrust[j]])
            self._first = False
        zeros = np.zeros(3)
        for j in range(self.N):
            s.set(j, "yref", np.concatenate((P[j], rpy_ref[j], V[j], zeros, cmd_rpy[j], [thrust[j]])))
        s.set(self.N, "yref", np.concatenate((P[self.N], rpy_ref[self.N], V[self.N], zeros)))
        s.solve()
        u0 = s.get(0, "u")
        return np.asarray(u0, dtype=np.float32)

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
        print(f"[MPCController] ABSTURZ bei t = {t:.2f} s nach {n_passed} Gates: {cause}. "
              f"Abweichung Soll-Ist {np.linalg.norm(err):.2f} m, Tempo {speed:.1f} m/s{seen}")

    def episode_callback(self):
        """Zurücksetzen."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Sollbahn (grün) und aktuellen Sollpunkt (rot) zeichnen."""
        t = self._tick / self._freq
        draw_line(sim, self._reference(np.linspace(t, self._t_total, 100))[0], rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._reference(t)[0], rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
