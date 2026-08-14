"""
Kinematik-Analyse v10 – Carrera Hybrid

Ueberarbeitete Auswertelogik. Die folgenden Fehler aus v9 sind behoben:

A) velocity_sg – Skalierungsfehler. v9 hat auf ein Gitter mit len(t)
   Punkten resampled, aber savgol delta=median(diff(t)) uebergeben. Weil
   einzelne Frames fehlen, war der echte Gitterabstand 2-4 % groesser als
   delta => alle SG-Geschwindigkeiten systematisch 2-4 % zu hoch.
   Jetzt wird das Gitter mit np.arange im echten Framerate-Takt gebaut.

B) abs(v) entfernt. Die Fahrtrichtung wird einmal global bestimmt; eine
   echte Rueckwaertsbewegung bleibt damit sichtbar, statt nach oben
   geklappt zu werden.

C) Ein- und Ausfahrframes. Solange das Auto am Bildrand angeschnitten
   ist, waechst der Blob, statt sich zu bewegen – sein Median wandert
   dann etwa halb so schnell wie das Auto und taeuscht eine viel zu
   kleine Geschwindigkeit vor. v9 wollte das ueber die Pixelzahl
   abfangen, was genau die Uebergangsframes durchlaesst. Jetzt wird
   geometrisch geprueft, ob Ober- UND Unterkante im Bild liegen.

D) Bremsauswertung. v9 nahm v_peak als argmax eines verrauschten
   Signals (argmax ist nach oben verzerrt) und fittete a_brake bei
   FESTEM v_peak – das Rauschen ging damit direkt in a_brake. Jetzt:
   segmentierter Fit auf die Position (die Rohmessgroesse, rund 30x
   weniger Rauschen als ihre Ableitung) mit robuster Verlustfunktion.
   Ergebnis: beide Versuche liefern jetzt konsistente Werte.

E) Durchfahrten. v9 fittete eine Gerade ueber das letzte Viertel; das
   liefert den Mittelwert dieses Fensters, nicht die Austritts-
   geschwindigkeit. Jetzt quadratischer Positionsfit ueber die ganze
   Spur, v_ein und v_aus sauber getrennt.

F) Beschleunigung aus dem Stand. Das Auto steht im Startmoment
   teilweise oberhalb der Y_MIN-Linie. Der Zeitnullpunkt wird deshalb
   ueber die Blob-Unterkante bestimmt (die bewegt sich mit der echten
   Fahrzeuggeschwindigkeit) und v=0 dort verankert. RMSE dadurch von
   2.4 cm auf 0.5 cm.

G) Plot. In v9 lagen Streupunkte (Nullpunkt = Datenbeginn) und Fitkurve
   (Nullpunkt = t_peak) mit verschiedenen Zeitnullpunkten uebereinander.
   Ausserdem war die Sim-Vergleichskurve auf der aneinandergereihten
   20-s-Achse ein 0.2 s schmaler, unsichtbarer Strich. Die Bremsungen
   werden jetzt am Bremsbeginn ausgerichtet uebereinandergelegt.

H) v_max. Die Extrapolation auf die Endgeschwindigkeit war eine
   Handformel ohne Datenbasis. Jetzt wird das physikalische Modell
   dv/dt = (v_inf - v)/T gefittet UND geprueft, ob v_inf durch die
   Daten ueberhaupt bestimmt ist. Ist es nicht – siehe Ausgabe.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

MEDIA  = Path(r'e:\Hochschule\Master\Carrera Hybrid\Code\VS Code\Carrera_rl\carrera_rl\Other tools\Auto Parameter messen\media\Geschnitten')
OUTDIR = Path(r'e:\Hochschule\Master\Carrera Hybrid\Code\VS Code\Carrera_rl\carrera_rl\Other tools\Auto Parameter messen\Code')

PX_PER_M    = 400.0     # 14 px = 35 mm
X_MIN       = 700
X_MAX       = 1100
Y_MIN       = 30        # oberste Zeilen weg (Beine im Bild)
DIFF_THRESH = 25
N_MIN_PX    = 5
N_MAX_PX    = 2000
MAX_JUMP_PX = 60
SG_WIN      = 21
EDGE_MARGIN = 4.0       # px Abstand zum ROI-Rand fuer "ganz sichtbar"

# bisheriges Simulationsmodell (carrera_2d_env.py)
SIM_MASS        = 0.080
SIM_BRAKE_FORCE = 0.4
SIM_DRAG        = 0.5

V_MAX_LIT = 1.8


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def track_video(path):
    """Hintergrundmodell = Median aus 30 gleichverteilten Frames.

    Pro Frame wird zusaetzlich die Ober- und Unterkante des Blobs
    festgehalten (robuste Perzentile statt min/max), damit spaeter
    geprueft werden kann, ob das Auto vollstaendig im Bild ist.
    """
    cap   = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    samples = []
    for fi in np.linspace(0, total - 1, min(30, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, f = cap.read()
        if ret:
            samples.append(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    bg = np.median(samples, axis=0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    raw = []
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        diff = np.clip(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) - bg,
            0, 255).astype(np.uint8)
        mask = diff[Y_MIN:, X_MIN:X_MAX] > DIFF_THRESH
        n    = int(np.count_nonzero(mask))
        if N_MIN_PX <= n <= N_MAX_PX:
            ys, _ = np.where(mask)
            raw.append((fi / fps,
                        float(np.median(ys))          + Y_MIN,
                        n,
                        float(np.percentile(ys, 2.0)) + Y_MIN,
                        float(np.percentile(ys, 98.0)) + Y_MIN))
    cap.release()

    arr = np.array(raw) if len(raw) >= 5 else None
    if arr is not None:
        print(f"  Tracking: {len(arr)}/{total} Frames, "
              f"y=[{arr[:,1].min():.0f}..{arr[:,1].max():.0f}] px, "
              f"n=[{arr[:,2].min():.0f}..{arr[:,2].max():.0f}] px")
    else:
        print("  Tracking: KEINE Detektion")
    return arr, float(bg.shape[0])


# ---------------------------------------------------------------------------
# Vorverarbeitung
# ---------------------------------------------------------------------------

def remove_jumps(t, y):
    """Einzelne Ausreisserframes entfernen (beide Nachbarn springen)."""
    keep = np.ones(len(t), bool)
    for i in range(1, len(t) - 1):
        if (abs(y[i] - y[i-1]) > MAX_JUMP_PX and
                abs(y[i+1] - y[i]) > MAX_JUMP_PX):
            keep[i] = False
    return keep


def clip_forward(t, y):
    """Ab dem Maximum abschneiden – verhindert Rueckwaertssprung-Artefakte."""
    turn = int(np.argmax(y))
    if turn < len(y) - 1:
        return t[:turn + 1], y[:turn + 1]
    return t, y


def longest_run(mask):
    """Laengster zusammenhaengender True-Block, (Start, Ende) inklusiv."""
    best, cur, best_len = None, None, 0
    for i, v in enumerate(mask):
        if v:
            cur = i if cur is None else cur
            if i - cur + 1 > best_len:
                best_len, best = i - cur + 1, (cur, i)
        else:
            cur = None
    return best


def prepare(tracked):
    """Nur Frames mit vollstaendig sichtbarem Auto, Zeit ab 0, Richtung
    normiert, Weg in Metern.

    Der Sichtbarkeitstest ist der wichtigste Fix gegenueber v9: solange
    das Auto ein- oder ausfaehrt, ist der Blob am Rand abgeschnitten und
    sein Median bewegt sich langsamer als das Fahrzeug.
    """
    raw, h_frame = tracked
    if raw is None:
        return None
    t, y, n      = raw[:, 0], raw[:, 1], raw[:, 2]
    y_top, y_bot = raw[:, 3], raw[:, 4]

    keep = remove_jumps(t, y)
    t, y, n, y_top, y_bot = (t[keep], y[keep], n[keep],
                             y_top[keep], y_bot[keep])
    t, y = clip_forward(t, y)
    y_top, y_bot = y_top[:len(t)], y_bot[:len(t)]

    vis = ((y_top > Y_MIN + EDGE_MARGIN) & (y_bot < h_frame - EDGE_MARGIN))
    run = longest_run(vis)
    if run is None or run[1] - run[0] + 1 < 12:
        return None
    i0, i1 = run

    t_abs = t[i0:i1 + 1]
    y     = y[i0:i1 + 1]
    sgn   = 1.0 if (y[-1] - y[0]) >= 0 else -1.0
    print(f"  Auto ganz im Bild: Frames {i0}..{i1} "
          f"({i1 - i0 + 1} von {len(vis)} detektierten)")
    return dict(t=t_abs - t_abs[0], t_abs=t_abs,
                p=sgn * (y - y[0]) / PX_PER_M, n_pts=i1 - i0 + 1)


def motion_start(raw):
    """Zeitpunkt des Losfahrens, auch bei am Rand angeschnittenem Auto.

    Die Blob-Unterkante ist hier die Vorderkante in Fahrtrichtung und
    bewegt sich mit der echten Fahrzeuggeschwindigkeit, waehrend der
    Median eines angeschnittenen Blobs nachhinkt.
    """
    t, y_bot = raw[:, 0], raw[:, 4]
    base = float(np.median(y_bot[:min(10, len(y_bot))]))
    idx  = np.where(y_bot > base + 3.0)[0]
    return float(t[idx[0]]) if len(idx) else None


# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------

def velocity_sg(t, p, win=SG_WIN):
    """SG-Ableitung auf einem Gitter im echten Framerate-Takt.

    Nur fuer die Darstellung der Messpunkte – alle Kennwerte kommen aus
    Positionsfits, weil Differenzieren das Rauschen stark verstaerkt.
    """
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return t, np.gradient(p, t)
    t_uni = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    p_uni = np.interp(t_uni, t, p)
    w = min(win, len(p_uni))
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return t_uni, np.gradient(p_uni, dt)
    return t_uni, savgol_filter(p_uni, w, 3, deriv=1, delta=dt)


def drag_pos(t, p0, v_inf, tau):
    """Weg bei geschwindigkeitsabhaengigem Widerstand, Start aus dem Stand:
       dv/dt = (v_inf - v)/tau  =>  v(t) = v_inf*(1 - e^(-t/tau))
    """
    return p0 + v_inf * (t - tau * (1.0 - np.exp(-t / tau)))


def brake_pos(t, v0, a1, tb, a2):
    """Weg: bis tb Anfahrt (v0, leichte Beschleunigung a1), danach Bremsen
    mit konstanter Verzoegerung a2; nach Stillstand konstant."""
    t    = np.asarray(t, float)
    v_tb = v0 + a1 * tb
    p_tb = v0 * tb + 0.5 * a1 * tb**2
    dt2  = np.clip(t - tb, 0.0, None)
    dt2  = np.minimum(dt2, max(v_tb, 0.0) / max(a2, 1e-6))
    return np.where(t <= tb,
                    v0 * t + 0.5 * a1 * t**2,
                    p_tb + v_tb * dt2 - 0.5 * a2 * dt2**2)


def brake_vel(t, v0, a1, tb, a2):
    t   = np.asarray(t, float)
    dt2 = np.clip(t - tb, 0.0, None)
    return np.where(t <= tb,
                    v0 + a1 * t,
                    np.clip(v0 + a1 * tb - a2 * dt2, 0.0, None))


def sim_brake_curve(v0, t_max=3.0, dt=1/30.0):
    """Bremsverlauf des bisherigen Simulationsmodells:
       dv/dt = (-brake_force - drag*v) / masse
    """
    ts, vs = [0.0], [float(v0)]
    v, t = float(v0), 0.0
    while v > 0.001 and t < t_max:
        v  = max(v + (-SIM_BRAKE_FORCE - SIM_DRAG * v) / SIM_MASS * dt, 0.0)
        t += dt
        ts.append(t)
        vs.append(v)
    return np.array(ts), np.array(vs)


# ---------------------------------------------------------------------------
# Auswertungen
# ---------------------------------------------------------------------------

def analyse_accel(tracked):
    """Beschleunigung aus dem Stand.

    Das Auto steht im Startmoment teilweise oberhalb der Y_MIN-Linie und
    ist erst nach rund 0.5 s ganz sichtbar. Der Zeitnullpunkt kommt
    deshalb aus motion_start(), und v=0 wird dort verankert; der
    Weg-Offset p0 bleibt frei, weil die vorher zurueckgelegte Strecke
    unbekannt ist.
    """
    raw, _ = tracked
    d = prepare(tracked)
    if d is None:
        return None
    t_move = motion_start(raw)
    if t_move is None:
        return None

    tau = d['t_abs'] - t_move
    p   = d['p']
    if len(tau) < 12 or tau[0] < 0:
        return None

    # v(0)=0 exakt erzwungen: keine Potenz t^1 im Ansatz
    A = np.column_stack([np.ones_like(tau), tau**2, tau**3, tau**4])
    c, *_ = np.linalg.lstsq(A, p, rcond=None)
    p0, c2, c3, c4 = c

    def pf(x): return p0 + c2*x**2 + c3*x**3 + c4*x**4
    def vf(x): return 2*c2*x + 3*c3*x**2 + 4*c4*x**3

    t_sm = np.linspace(0, tau[-1], 400)
    v_sm = np.clip(vf(t_sm), 0, None)
    rmse = float(np.sqrt(np.mean((p - pf(tau))**2)))

    # Widerstandsmodell: liefert v_inf als Fitparameter statt als Handformel
    span = float(tau[-1] - tau[0])
    drag = None
    try:
        popt, pcov = curve_fit(drag_pos, tau, p, p0=[float(p[0]), 1.8, 2.0],
                               bounds=([-1.0, 0.5, 0.1], [1.0, 8.0, 60.0]),
                               method='trf', max_nfev=20000)
        v_inf, T = float(popt[1]), float(popt[2])
        r_d   = float(np.sqrt(np.mean((p - drag_pos(tau, *popt))**2)))
        # Ist v_inf durch die Daten ueberhaupt bestimmt? Nur wenn die
        # Saettigung im Messfenster sichtbar wird. Sonst tauschen v_inf
        # und tau beliebig gegeneinander (flaches Fitminimum).
        determined = (T < 1.5 * span) and (v_inf < 7.5)
        drag = dict(v_inf=v_inf, tau=T, rmse=r_d, determined=determined,
                    v_inf_err=float(np.sqrt(np.diag(pcov))[1]))
        if determined:
            print(f"  Widerstandsmodell: v_max = {v_inf:.2f} m/s, "
                  f"Zeitkonstante {T:.2f} s, RMSE {r_d*100:.2f} cm")
        else:
            print(f"  Widerstandsmodell: v_max NICHT bestimmbar "
                  f"(Zeitkonstante {T:.1f} s >> Messfenster {span:.1f} s, "
                  f"Fitminimum flach)")
    except Exception as e:
        print(f"  Widerstandsmodell nicht gefittet: {e}")

    t_sg, v_sg = velocity_sg(tau, p)
    print(f"  Losfahren bei t={t_move:.2f} s, ganz sichtbar ab "
          f"tau={tau[0]:.2f} s (davor rekonstruiert)")
    print(f"  v(0)=0 verankert | v am Bildende = {v_sm[-1]:.3f} m/s | "
          f"RMSE {rmse*100:.2f} cm | {len(tau)} Punkte")
    return dict(t=tau, p=p, pf=pf, vf=vf, rmse=rmse, drag=drag,
                t_sm=t_sm, v_sm=v_sm, v_end=float(v_sm[-1]),
                tau_data0=float(tau[0]), t_sg=t_sg, v_sg=v_sg)


def analyse_speed(tracked):
    """Durchfahrt mit Anlauf: quadratischer Positionsfit
    p = v_ein*t + 0.5*a*t^2 => v_ein und v_aus sauber getrennt."""
    d = prepare(tracked)
    if d is None:
        return None
    t, p = d['t'], d['p']

    A = np.column_stack([t, t**2])
    c, *_ = np.linalg.lstsq(A, p, rcond=None)
    v_ein, a = float(c[0]), float(2.0 * c[1])

    resid = p - (v_ein * t + 0.5 * a * t**2)
    rmse  = float(np.sqrt(np.mean(resid**2)))
    dof   = max(len(t) - 2, 1)
    cov   = (float(resid @ resid) / dof) * np.linalg.inv(A.T @ A)
    v_err = float(np.sqrt(cov[0, 0]))

    t_end = float(t[-1])
    v_aus = v_ein + a * t_end
    t_sg, v_sg = velocity_sg(t, p)
    tl = np.linspace(0, t_end, 200)

    print(f"  v_ein = {v_ein:.3f} +/- {v_err:.3f} m/s  ->  "
          f"v_aus = {v_aus:.3f} m/s | a = {a:.3f} m/s^2 | "
          f"RMSE {rmse*100:.2f} cm | {len(t)} Punkte")
    return dict(t_span=t_end, v_ein=v_ein, v_aus=v_aus, a=a,
                v_err=v_err, rmse=rmse, t_sg=t_sg, v_sg=v_sg,
                t_line=tl, v_line=v_ein + a * tl)


def analyse_brake(tracked):
    """Bremsung: segmentierter Positionsfit statt Peak-Suche im
    verrauschten Geschwindigkeitssignal."""
    d = prepare(tracked)
    if d is None:
        return None
    t, p = d['t'], d['p']
    T = float(t[-1])

    first = t <= T / 3.0
    v0_g  = float(np.polyfit(t[first], p[first], 1)[0]) if first.sum() > 3 else 1.0
    p_init = [max(v0_g, 0.1), 0.0, 0.5 * T, 1.0]
    lo     = [0.05, -2.0, 0.05 * T, 0.02]
    hi     = [6.00,  6.0, 0.95 * T, 40.0]

    try:
        popt, pcov = curve_fit(brake_pos, t, p, p0=p_init, bounds=(lo, hi),
                               method='trf', loss='soft_l1', f_scale=0.01,
                               max_nfev=20000)
    except Exception as e:
        print(f"  Fit fehlgeschlagen: {e}")
        return None

    v0, a1, tb, a_brake = [float(x) for x in popt]
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(4)

    v_bs       = v0 + a1 * tb
    t_stop_abs = tb + v_bs / max(a_brake, 1e-6)
    v_end_data = float(brake_vel(T, *popt))
    stopped    = t_stop_abs <= T
    rmse       = float(np.sqrt(np.mean((p - brake_pos(t, *popt))**2)))

    t_sg, v_sg = velocity_sg(t, p)
    t_fit = np.linspace(0, T, 500)

    if not stopped and v_end_data > 0.05:
        t_ext = np.linspace(T, t_stop_abs, 80)
        v_ext = np.clip(v_bs - a_brake * (t_ext - tb), 0, None)
    else:
        t_ext = v_ext = None

    print(f"  Bremsbeginn bei t={tb:.2f} s aus v={v_bs:.3f} m/s | "
          f"a_brake = {a_brake:.3f} +/- {perr[3]:.3f} m/s^2")
    print(f"  v am Datenende = {v_end_data:.3f} m/s | "
          + (f"steht bei t={t_stop_abs:.2f} s" if stopped
             else f"verlaesst Bild fahrend, Stillstand extrapoliert "
                  f"bei t={t_stop_abs:.2f} s")
          + f" | RMSE {rmse*100:.2f} cm")

    return dict(v0=v0, a1=a1, tb=tb, a_brake=a_brake, v_brake_start=v_bs,
                v_end_data=v_end_data, t_stop_abs=t_stop_abs,
                stopped=stopped, rmse=rmse, a_err=float(perr[3]), T=T,
                t_sg=t_sg, v_sg=v_sg, t_fit=t_fit,
                v_fit=brake_vel(t_fit, *popt),
                t_sim=sim_brake_curve(v_bs)[0],
                v_sim=sim_brake_curve(v_bs)[1],
                t_ext=t_ext, v_ext=v_ext,
                t=t, p=p, p_fit=brake_pos(t, *popt))


# ---------------------------------------------------------------------------
# Darstellung
# ---------------------------------------------------------------------------

C_ACC = "#1b5e20"
C_SP  = ["#0d47a1", "#42a5f5"]
C_BR  = ["#b71c1c", "#ef6c00"]
C_SIM = "#455a64"
SG_EDGE = SG_WIN // 2


def sg_clean(t_sg, v_sg):
    """Filterartefakte an den Enden nicht als Messpunkte darstellen."""
    if len(t_sg) > 2 * SG_EDGE + 4:
        return t_sg[SG_EDGE:-SG_EDGE], v_sg[SG_EDGE:-SG_EDGE]
    return t_sg, v_sg


def make_plot(r_a0, speeds, brakes, v_max_obs, a_mean):
    fig = plt.figure(figsize=(16, 10.5))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                           hspace=0.30, wspace=0.20)
    fig.suptitle("Carrera Hybrid – Kinematik aus Videomessung",
                 fontsize=14, fontweight="bold", y=0.975)
    fig.text(0.5, 0.943,
             f"Kalibrierung 14 px = 35 mm  ·  30 fps  ·  "
             f"hoechste gemessene Geschwindigkeit {v_max_obs:.2f} m/s  ·  "
             f"Bremsverzoegerung {a_mean:.2f} m/s²",
             ha="center", fontsize=10, color="#37474f")

    # ---- A: Weg-Zeit mit Fit ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A · Weg-Zeit-Messung und Modellfit", fontsize=11,
                 fontweight="bold", loc="left")
    if r_a0:
        ax.scatter(r_a0['t'], r_a0['p'], s=7, color=C_ACC, alpha=0.30, zorder=2)
        tf = np.linspace(r_a0['t'][0], r_a0['t'][-1], 300)
        ax.plot(tf, r_a0['pf'](tf), color=C_ACC, lw=2.2, zorder=3,
                label=f"aus dem Stand  ({r_a0['rmse']*100:.1f} cm)")
    for i, (r, c) in enumerate(zip(brakes, C_BR), 1):
        ax.scatter(r['t'], r['p'], s=7, color=c, alpha=0.30, zorder=2)
        ax.plot(r['t'], r['p_fit'], color=c, lw=2.2, zorder=3,
                label=f"Bremsen V{i}  ({r['rmse']*100:.1f} cm)")
        ax.axvline(r['tb'], color=c, lw=1.1, ls=":", alpha=0.8)
    ax.set_xlabel("Zeit seit Messbeginn (s)")
    ax.set_ylabel("zurueckgelegter Weg (m)")
    ax.legend(fontsize=8.5, loc="upper left",
              title="Messung (RMSE des Fits)", title_fontsize=8)
    ax.grid(True, alpha=0.28)

    # ---- B: Beschleunigung ----
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B · Beschleunigung", fontsize=11, fontweight="bold",
                 loc="left")
    if r_a0:
        ts, vs = r_a0['t_sm'], r_a0['v_sm']
        early  = ts <= r_a0['tau_data0']
        tt, vv = sg_clean(r_a0['t_sg'], r_a0['v_sg'])
        ax.plot(tt, vv, color=C_ACC, lw=0, marker='.', ms=3, alpha=0.28,
                zorder=2)
        ax.fill_between(ts, 0, vs, alpha=0.10, color=C_ACC)
        ax.plot(ts[early], vs[early], color=C_ACC, lw=1.8, ls="--",
                alpha=0.8, zorder=3, label="aus dem Stand (Start rekonstruiert)")
        ax.plot(ts[~early], vs[~early], color=C_ACC, lw=2.6, zorder=3,
                label=f"aus dem Stand  ->  {r_a0['v_end']:.2f} m/s")
    for i, (r, c) in enumerate(zip(speeds, C_SP), 1):
        tt, vv = sg_clean(r['t_sg'], r['v_sg'])
        ax.plot(tt, vv, color=c, lw=0, marker='.', ms=3, alpha=0.28, zorder=2)
        ax.plot(r['t_line'], r['v_line'], color=c, lw=2.4, zorder=3,
                label=f"Durchfahrt V{i}: {r['v_ein']:.2f} -> {r['v_aus']:.2f} m/s")

    # Ehrliche Darstellung des unbestimmten Bereichs oberhalb der Messung
    ax.axhspan(v_max_obs, V_MAX_LIT * 1.20, color="#9e9e9e", alpha=0.13,
               zorder=0)
    ax.axhline(v_max_obs, color="#01579b", lw=1.3, ls="-.", alpha=0.85)
    ax.text(0.985, v_max_obs - 0.06,
            f"hoechste gemessene Geschwindigkeit {v_max_obs:.2f} m/s",
            color="#01579b", fontsize=8, ha="right", va="top",
            transform=ax.get_yaxis_transform())
    ax.text(0.985, (v_max_obs + V_MAX_LIT * 1.18) / 2,
            "v_max liegt hier oben –\naus diesen Videos nicht bestimmbar,\n"
            "das Auto saettigt im Bildausschnitt nicht",
            color="#546e7a", fontsize=8, ha="right", va="center",
            transform=ax.get_yaxis_transform())
    ax.axhline(V_MAX_LIT, color="#e65100", lw=1.1, ls=":", alpha=0.8)
    ax.text(0.015, V_MAX_LIT + 0.02, f"Literaturwert {V_MAX_LIT} m/s",
            color="#e65100", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("Zeit seit Beginn der jeweiligen Messung (s)")
    ax.set_ylabel("Geschwindigkeit (m/s)")
    ax.set_ylim(0, V_MAX_LIT * 1.20)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.94)
    ax.grid(True, alpha=0.28)

    # ---- C: Bremsungen, am Bremsbeginn ausgerichtet ----
    ax = fig.add_subplot(gs[1, :])
    ax.set_title("C · Bremsung – beide Versuche am Bremsbeginn ausgerichtet, "
                 "im Vergleich zum bisherigen Simulationsmodell",
                 fontsize=11, fontweight="bold", loc="left")

    t_lo, t_hi = -1.0, 0.5
    for i, (r, c) in enumerate(zip(brakes, C_BR), 1):
        sh = r['tb']
        tt, vv = sg_clean(r['t_sg'], r['v_sg'])
        ax.plot(tt - sh, vv, color=c, lw=0, marker='.', ms=4, alpha=0.30,
                zorder=2)
        tf = r['t_fit'] - sh
        ax.plot(tf, r['v_fit'], color=c, lw=2.8, zorder=4,
                label=f"Bremsen V{i} gemessen:  a = {r['a_brake']:.2f} "
                      f"± {r['a_err']:.2f} m/s²   (ab {r['v_brake_start']:.2f} m/s)")
        ax.fill_between(tf, 0, r['v_fit'], alpha=0.10, color=c)
        if r['t_ext'] is not None:
            ax.plot(r['t_ext'] - sh, r['v_ext'], color=c, lw=1.6, ls="--",
                    alpha=0.7, zorder=4)
            ax.plot([r['t_stop_abs'] - sh], [0], marker='o', ms=6,
                    mfc="white", mec=c, mew=1.8, zorder=5)
            ax.annotate(f"Stillstand nach {r['t_stop_abs'] - sh:.1f} s",
                        xy=(r['t_stop_abs'] - sh, 0),
                        xytext=(r['t_stop_abs'] - sh, 0.20 + 0.16 * i),
                        fontsize=8, color=c, ha="center",
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.9))
        t_hi = max(t_hi, (r['t_stop_abs'] - sh) + 0.6)
        ax.plot(r['t_sim'], r['v_sim'], color=C_SIM, lw=2.2, ls=":",
                alpha=0.9, zorder=5,
                label="bisheriges Simulationsmodell" if i == 1 else "_nolegend_")

    ax.axvline(0, color="#37474f", lw=1.2, alpha=0.65)
    ax.axvspan(t_lo, 0, color="#000000", alpha=0.03, zorder=0)
    ax.text(t_lo / 2, V_MAX_LIT * 1.03, "Anfahrt", ha="center", fontsize=9,
            color="#546e7a")
    ax.text(0.06, V_MAX_LIT * 1.03, "Bremsbeginn", fontsize=9, color="#37474f")

    # Beide Zeiten auf dieselbe Referenzgeschwindigkeit beziehen, sonst
    # weicht der Faktor in der Grafik von dem in der Konsole ab.
    t_sim_stop = float(sim_brake_curve(v_max_obs)[0][-1])
    t_meas     = v_max_obs / a_mean
    ax.annotate(
        f"Das bisherige Modell bremst in {t_sim_stop:.2f} s auf null.\n"
        f"Gemessen dauert es rund {t_meas:.1f} s –\n"
        f"das Modell bremst etwa {t_meas / max(t_sim_stop, 1e-6):.0f}-mal zu stark.",
        xy=(t_sim_stop, 0.06), xytext=(0.62, 1.30),
        fontsize=9.5, color=C_SIM, ha="left",
        bbox=dict(boxstyle="round,pad=0.45", fc="#eceff1", ec=C_SIM,
                  alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=C_SIM, lw=1.4,
                        connectionstyle="arc3,rad=-0.25"))

    box = (f"Empfohlene Simulationsparameter\n"
           f"   a_brake = {a_mean:.2f} m/s²   (konstant, gut belegt)\n"
           f"   v_max  >= {v_max_obs:.2f} m/s   (untere Schranke,\n"
           f"              Literatur {V_MAX_LIT:.1f} m/s)")
    ax.text(0.012, 0.055, box, transform=ax.transAxes, fontsize=8.8,
            va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                      ec="#9e9e9e", alpha=0.95))

    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(0, V_MAX_LIT * 1.12)
    ax.set_xlabel("Zeit relativ zum Bremsbeginn (s)")
    ax.set_ylabel("Geschwindigkeit (m/s)")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.94)
    ax.grid(True, alpha=0.28)

    out = OUTDIR / f"kinematik_v10_{datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("ANALYSE v10 – Positionsfits statt Ableitungsfits")
    print("=" * 72)

    print("\nBeschleunigung aus dem Stand:")
    r_a0 = analyse_accel(track_video(MEDIA / "Beschleunigung von 0.mp4"))

    print("\nDurchfahrt mit Anlauf V1:")
    r_s1 = analyse_speed(track_video(MEDIA / "Beschleunigung mit speed.mp4"))

    print("\nDurchfahrt mit Anlauf V2:")
    r_s2 = analyse_speed(track_video(MEDIA / "Beschleunigung mit Speed V2.mp4"))

    print("\nBremsen V1:")
    r_b1 = analyse_brake(track_video(MEDIA / "Bremsen mit geschwindigkeit V1.mp4"))

    print("\nBremsen V2:")
    r_b2 = analyse_brake(track_video(MEDIA / "Bremsen mit geschwindigkeit V2.mp4"))

    speeds = [r for r in (r_s1, r_s2) if r]
    brakes = [r for r in (r_b1, r_b2) if r]

    v_obs = [r['v_aus'] for r in speeds] + [r['v_brake_start'] for r in brakes]
    if r_a0:
        v_obs.append(r_a0['v_end'])
    v_max_obs = max(v_obs) if v_obs else V_MAX_LIT
    a_list = [r['a_brake'] for r in brakes]
    a_mean = float(np.mean(a_list)) if a_list else 0.6

    print("\n" + "=" * 72)
    print("ERGEBNISSE")
    print("=" * 72)
    if r_a0:
        print(f"  aus dem Stand:  erreicht {r_a0['v_end']:.2f} m/s am Bildende")
    for i, r in enumerate(speeds, 1):
        print(f"  Durchfahrt V{i}: {r['v_ein']:.2f} -> {r['v_aus']:.2f} m/s "
              f"(a = {r['a']:+.2f} m/s^2)")
    for i, r in enumerate(brakes, 1):
        print(f"  Bremsen V{i}:    a_brake = {r['a_brake']:.3f} "
              f"+/- {r['a_err']:.3f} m/s^2  aus {r['v_brake_start']:.2f} m/s")

    print(f"\n  a_brake (Mittel beider Versuche): {a_mean:.3f} m/s^2")
    if a_list:
        print(f"  Streuung zwischen den Versuchen:  "
              f"{abs(a_list[0] - a_list[-1]) / a_mean * 100:.0f} %")
    print(f"  hoechste beobachtete Geschwindigkeit: {v_max_obs:.2f} m/s "
          f"(untere Schranke fuer v_max)")

    t_ref, _ = sim_brake_curve(v_max_obs)
    print(f"\n  Bisheriges Modell bremst von {v_max_obs:.2f} m/s in "
          f"{t_ref[-1]:.2f} s auf null,")
    print(f"  gemessen dauert es {v_max_obs / a_mean:.2f} s  =>  Faktor "
          f"{(v_max_obs / a_mean) / t_ref[-1]:.0f} zu stark.")

    print(f"\nPlot: {make_plot(r_a0, speeds, brakes, v_max_obs, a_mean)}")
    return r_a0, speeds, brakes


if __name__ == "__main__":
    main()
