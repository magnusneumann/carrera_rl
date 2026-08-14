"""
Kinematik-Analyse v8

Basis: v6 (bewaehrt), mit gezielten Verbesserungen:
1. N_MAX_PX-Filter (n>2000 = falsches Blob): Entfernt grosse Fehldetektionen
   am Ende von Bremsen-Videos (Frames 110-113: n=3065-4229).
2. abs(v) in Geschwindigkeitsberechnung -> richtungsunabhaengig.
3. "Mit Geschwindigkeit": SG-Filter statt rohe Differenzen + zeigt v(t)-Kurven.
4. Bremsen: i0-Start erst bei erster vollstaendiger Sichtbarkeit.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

MEDIA    = Path(r'e:\Hochschule\Master\Carrera Hybrid\Code\VS Code\Carrera_rl\carrera_rl\Other tools\Auto Parameter messen\media\Geschnitten')
OUTDIR   = Path(r'e:\Hochschule\Master\Carrera Hybrid\Code\VS Code\Carrera_rl\carrera_rl\Other tools\Auto Parameter messen\Code')

PX_PER_M    = 400.0
X_MIN       = 700
X_MAX       = 1100
DIFF_THRESH = 25
N_MIN_PX    = 5       # Minimale Pixelzahl (wie v6)
N_MAX_PX    = 2000    # NEU: Maximale Pixelzahl (entfernt grosse Fehldetektionen)
MAX_JUMP_PX = 60
SG_WIN      = 21      # SG-Fenster: 21 Frames @ 30fps = 0.7s


# ---------------------------------------------------------------------------
# Tracking: Hintergrundsubtraktion + roher Schwerpunkt (wie v6, bewaehrt)
# + N_MAX_PX-Filter fuer grosse Fehldetektionen
# ---------------------------------------------------------------------------

def track_video(path):
    """Tracking wie v6 (Median-Hintergrund + roher Schwerpunkt), aber mit
    N_MAX_PX-Filter um grosse Fehldetektionen auszuschliessen."""
    cap   = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    # Hintergrund: Median aus 30 gleichmaessig verteilten Frames (wie v6)
    samples = []
    for fi in np.linspace(0, total - 1, min(30, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, f = cap.read()
        if ret:
            samples.append(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            )
    bg = np.median(samples, axis=0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    raw = []

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        diff = np.clip(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) - bg,
            0, 255
        ).astype(np.uint8)
        roi = diff[:, X_MIN:X_MAX]
        mask = roi > DIFF_THRESH
        n = int(np.count_nonzero(mask))

        if N_MIN_PX <= n <= N_MAX_PX:
            ys, _ = np.where(mask)
            raw.append((fi / fps, float(np.median(ys)), n))

    cap.release()

    arr = np.array(raw) if len(raw) >= 5 else None
    if arr is not None:
        print(f"  Tracking: {len(arr)}/{total} Frames, "
              f"y=[{arr[:,1].min():.0f}..{arr[:,1].max():.0f}]px, "
              f"n=[{arr[:,2].min():.0f}..{arr[:,2].max():.0f}]")
    else:
        print(f"  Tracking: KEINE Detektion")
    return arr


# ---------------------------------------------------------------------------
# Hilfs-Filter (unveraendert aus v6)
# ---------------------------------------------------------------------------

def remove_jumps(t, y):
    """Entfernt isolierte Frames wo y in beide Richtungen > MAX_JUMP_PX springt."""
    keep = np.ones(len(t), bool)
    for i in range(1, len(t) - 1):
        if (abs(y[i] - y[i-1]) > MAX_JUMP_PX and
                abs(y[i+1] - y[i]) > MAX_JUMP_PX):
            keep[i] = False
    return keep


def clip_forward(t, y):
    """Schneidet nach dem Maximum von y ab (verhindert Rueckwaerts-Artefakte).
    Nur kein Clip wenn max das allerletzte Element ist (Auto noch in Bewegung)."""
    turn = int(np.argmax(y))
    if turn < len(y) - 1:
        return t[:turn + 1], y[:turn + 1]
    return t, y


# ---------------------------------------------------------------------------
# Geschwindigkeit via SG-Filter
# ---------------------------------------------------------------------------

def velocity_sg(t, y_px, win=SG_WIN):
    """
    SG-Ableitung auf y_px-Positionen -> Geschwindigkeit in m/s.
    abs(v) macht sie richtungsunabhaengig (NEU gegenueber v6).
    """
    p  = y_px / PX_PER_M
    dt = float(np.median(np.diff(t)))

    t_uni = np.linspace(t[0], t[-1], len(t))
    p_uni = np.interp(t_uni, t, p)

    w = min(win, len(p_uni))
    if w % 2 == 0:
        w -= 1

    if w >= 5:
        v = savgol_filter(p_uni, w, 3, deriv=1, delta=dt)
    else:
        v = np.gradient(p_uni, dt)

    return t_uni, np.abs(v)


# ---------------------------------------------------------------------------
# Beschleunigung von 0 (Eingeschraenktes Polynom, v(0)=0 exakt - aus v6)
# ---------------------------------------------------------------------------

def analyse_accel(raw):
    t, y, n = raw[:, 0], raw[:, 1], raw[:, 2]

    # Sprung-Filter und Vorwaerts-Clip (wie v6)
    keep = remove_jumps(t, y)
    t, y = t[keep], y[keep]
    t, y = clip_forward(t, y)

    p  = y / PX_PER_M
    dp = np.abs(np.diff(p)) / np.diff(t)
    i0 = max(1, int(np.argmax(dp > 0.05)))

    t_use = t[i0:] - t[i0]
    p_use = p[i0:] - p[i0]

    if len(t_use) < 10:
        print("  Zu wenige Frames nach i0")
        return None

    # Eingeschraenktes Polynom: p(t) = c2*t^2 + c3*t^3 + c4*t^4 -> v(0)=0 exakt
    A = np.column_stack([t_use**2, t_use**3, t_use**4])
    c, _, _, _ = np.linalg.lstsq(A, t_use * 0 + 0, rcond=None)  # dummy init
    c, _, _, _ = np.linalg.lstsq(A, p_use, rcond=None)
    c2, c3, c4 = c

    def pf(ta): return c2*ta**2 + c3*ta**3 + c4*ta**4
    def vf(ta): return 2*c2*ta + 3*c3*ta**2 + 4*c4*ta**3

    t_sm  = np.linspace(0, t_use[-1], 500)
    v_sm  = np.clip([vf(ti) for ti in t_sm], 0, None)
    v_peak = float(v_sm.max())
    rmse   = float(np.sqrt(np.mean((p_use - pf(t_use))**2)))

    print(f"  v(0)={vf(0):.4f} m/s  v_peak={v_peak:.3f} m/s  "
          f"RMSE={rmse*100:.2f}cm  n={len(t_use)} frames")
    return dict(t=t_use, p=p_use, pf=pf, vf=vf, rmse=rmse,
                t_sm=t_sm, v_sm=v_sm, v_peak=v_peak)


# ---------------------------------------------------------------------------
# "Mit Geschwindigkeit" -> v_max via linearem Fit auf Mittelteil (robust)
# ---------------------------------------------------------------------------

def analyse_speed(raw, n_min=300):
    t, y, n = raw[:, 0], raw[:, 1], raw[:, 2]

    # Sprung-Filter + Ausfahrt-Clip (wie bei Bremsen)
    keep = remove_jumps(t, y)
    t, y, n = t[keep], y[keep], n[keep]
    t, y = clip_forward(t, y)

    # Nur Frames mit genug sichtbarem Auto
    mask = n[:len(t)] >= n_min
    if mask.sum() < 8:
        mask = np.ones(len(t), bool)
    t, y = t[mask], y[mask]

    if len(t) < 8:
        return None

    t = t - t[0]

    # i0: stationaere Fehldetektion am Anfang ueberspringen
    # (z.B. y=14 konstant, n=950 - gleiche Fehlquelle wie bei Bremsen-Videos)
    dy_first = float(np.median(np.abs(np.diff(y[:6])))) if len(y) >= 6 else 5.0
    if dy_first < 2.0:
        diffs = np.abs(np.diff(y))
        i0 = int(next((i for i, d in enumerate(diffs) if d > 5), 0))
    else:
        i0 = 0

    t = t[i0:] - t[i0]
    y = y[i0:]

    if len(t) < 6:
        return None

    # v_max: letztes Viertel der Durchfahrt (schnellste Phase bei Volldurchfahrt)
    lo   = 0.75 * t[-1]
    last = t >= lo
    if last.sum() < 3:
        last = t >= 0.50 * t[-1]   # Fallback: letztes Halb

    c     = np.polyfit(t[last], y[last], 1)
    v_max = float(abs(c[0]) / PX_PER_M)
    resid = y[last] - np.polyval(c, t[last])
    dt_l  = float(np.median(np.diff(t[last]))) if last.sum() > 1 else 0.033
    v_std = float(np.std(resid) / (PX_PER_M * max(dt_l, 1e-6)))

    print(f"  v_max={v_max:.3f} m/s  (letztes Viertel, n_last={last.sum()}, "
          f"i0={i0}, n_total={mask.sum()} frames)")
    return dict(t_span=float(t[-1]), v_max=v_max, v_std=min(v_std, v_max * 0.15))


# ---------------------------------------------------------------------------
# Bremsung: SG + linearer Fit ab Peak (wie v6, aber mit n_min-Einfahrt-Filter)
# ---------------------------------------------------------------------------

def analyse_brake(raw, n_min=300):
    t, y, n = raw[:, 0], raw[:, 1], raw[:, 2]

    # Sprung-Filter und Vorwaerts-Clip
    keep = remove_jumps(t, y)
    t, y, n = t[keep], y[keep], n[keep]
    t, y = clip_forward(t, y)

    # NEU: Einfahrt-Filter - warte bis Auto vollstaendig sichtbar ist
    # Falls kaum Frames mit genug Pixeln: keine Filterung
    full_mask = n[:len(t)] >= n_min
    if full_mask.sum() >= 5:
        i0 = int(np.where(full_mask)[0][0])
    else:
        # Fallback: Skip Frames wo y < 70px (wie in v6)
        i0 = sum(1 for yi in y[:15] if yi < 70)

    t = t[i0:] - t[i0]
    y = y[i0:]

    if len(t) < 10:
        print(f"  Zu wenige Frames nach Einfahrt-Filter (i0={i0})")
        return None

    t_uni, v = velocity_sg(t, y)

    peak_i = int(np.argmax(v))
    v_peak = float(v[peak_i])
    t_peak = float(t_uni[peak_i])

    print(f"  i0={i0}  v_peak={v_peak:.3f} m/s @ t={t_peak:.2f}s  "
          f"n_frames={len(t_uni)}")

    if v_peak < 0.15:
        print(f"  v_peak zu niedrig -> uebersprungen")
        return None

    # Bremsphase ab Peak
    brake = t_uni >= t_peak
    t_b   = t_uni[brake] - t_peak
    v_b   = v[brake]

    # Abschneiden beim Stillstand
    i_stop = int(np.argmax(v_b < 0.05))
    if i_stop > 3:
        t_b, v_b = t_b[:i_stop], v_b[:i_stop]

    if len(t_b) < 5:
        t_b = t_uni[brake] - t_peak
        v_b = v[brake]

    def v_lin(tt, a):
        return np.maximum(v_peak - a * tt, 0)

    try:
        popt, _ = curve_fit(v_lin, t_b, v_b, p0=[0.8],
                            bounds=([0.05], [20.]))
        a_brake = float(popt[0])
    except Exception as e:
        print(f"  curve_fit Fehler ({e}), Fallback")
        a_brake = (v_peak - float(v_b[-1])) / (float(t_b[-1]) + 1e-9)

    t_stop = v_peak / a_brake
    t_plot = np.linspace(0, t_stop + 0.05, 300)
    v_plot = np.maximum(v_peak - a_brake * t_plot, 0)

    print(f"  a_brake={a_brake:.3f} m/s^2  t_stop={t_stop:.2f}s")

    # Nur Bremsteil fuer Plot: raw data ab Peak
    peak_idx = int(np.searchsorted(t_uni, t_peak))
    t_raw_brk = t_uni[peak_idx:] - t_peak
    v_raw_brk = v[peak_idx:]

    return dict(v_peak=v_peak, a_brake=a_brake,
                t_raw_brk=t_raw_brk, v_raw_brk=v_raw_brk,
                t_plot=t_plot, v_plot=v_plot,
                t_peak=t_peak)


# ---------------------------------------------------------------------------
# Ausfuehren
# ---------------------------------------------------------------------------

print("=" * 65)
print("ANALYSE v8")
print("=" * 65)

print("\nBeschleunigung von 0:")
r_a0 = analyse_accel(track_video(MEDIA / "Beschleunigung von 0.mp4"))

print("\nMit Geschwindigkeit V1:")
r_s1 = analyse_speed(track_video(MEDIA / "Beschleunigung mit speed.mp4"))

print("\nMit Geschwindigkeit V2:")
r_s2 = analyse_speed(track_video(MEDIA / "Beschleunigung mit Speed V2.mp4"))

print("\nBremsen V1:")
r_b1 = analyse_brake(track_video(MEDIA / "Bremsen mit geschwindigkeit V1.mp4"))

print("\nBremsen V2:")
r_b2 = analyse_brake(track_video(MEDIA / "Bremsen mit geschwindigkeit V2.mp4"))

speed_results = [r for r in [r_s1, r_s2] if r]
brake_results = [r for r in [r_b1, r_b2] if r]

v_max_list   = [r['v_max'] for r in speed_results]
v_max_meas   = float(np.mean(v_max_list)) if v_max_list else 1.62
a_brake_list = [r['a_brake'] for r in brake_results]
a_brake_mean = float(np.mean(a_brake_list)) if a_brake_list else None
V_MAX_LIT    = 1.8

print("\n" + "=" * 65)
print("ERGEBNISSE")
print("=" * 65)
if r_a0:
    print(f"  v_peak (Beschl.): {r_a0['v_peak']:.2f} m/s  "
          f"RMSE={r_a0['rmse']*100:.2f}cm")
if v_max_list:
    parts = ", ".join(f"V{i+1}={v:.2f}" for i, v in enumerate(v_max_list))
    print(f"  v_max gemessen:   {v_max_meas:.2f} m/s  ({parts})")
if a_brake_list:
    parts = ", ".join(f"V{i+1}={a:.3f}" for i, a in enumerate(a_brake_list))
    print(f"  a_brake:          {a_brake_mean:.3f} m/s^2  ({parts})")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

C_ACCEL  = "#1b5e20"
C_EXTRAP = "#558b2f"
C_SPEED1 = "#0d47a1"
C_SPEED2 = "#1565c0"
C_BRK1   = "#b71c1c"
C_BRK2   = "#e53935"
C_VMAX   = "#01579b"

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Carrera Hybrid – Kinematik (v8)\n"
    f"14 px = 35 mm  |  "
    f"v_max gemessen: {v_max_meas:.2f} m/s  |  "
    f"v_max Literatur: {V_MAX_LIT} m/s",
    fontsize=11, fontweight="bold"
)

# ---- Panel 1: Positionsdaten ----
ax = axes[0]
ax.set_title("Position – Beschleunigung von 0", fontsize=10)
if r_a0:
    ax.scatter(r_a0['t'], r_a0['p'], s=4, color=C_ACCEL, alpha=0.4, zorder=2,
               label="Rohdaten")
    t_fit = np.linspace(0, r_a0['t'][-1], 400)
    ax.plot(t_fit, r_a0['pf'](t_fit), color=C_ACCEL, lw=2.5, zorder=3,
            label=f"Polynom-Fit  (RMSE={r_a0['rmse']*100:.2f} cm)")
else:
    ax.text(0.5, 0.5, "Keine Daten", transform=ax.transAxes,
            ha='center', va='center', fontsize=12, color='red')
ax.set_xlabel("Zeit (s)")
ax.set_ylabel("Position (m)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ---- Panel 2: v(t) ----
ax = axes[1]
ax.set_title("Geschwindigkeit  v(t)  –  alle Messungen", fontsize=10)

GAP = 0.5   # optische Luecke zwischen Phasen
t_cursor = 0.0

# --- Phase 1: Beschleunigung von 0 ---
if r_a0:
    ts = r_a0['t_sm']
    vs = r_a0['v_sm']
    ax.fill_between(ts, 0, vs, alpha=0.12, color=C_ACCEL)
    ax.plot(ts, vs, color=C_ACCEL, lw=2.5,
            label=f"Beschl. von 0  (v_peak={r_a0['v_peak']:.2f} m/s)")

    # Extrapolation via Drag-Modell bis v_max
    v_end = float(r_a0['vf'](ts[-1]))
    dv    = float(r_a0['vf'](ts[-1]) - r_a0['vf'](max(ts[-1] - 0.1, 0)))
    a_end = dv / 0.1
    v_inf = v_max_meas
    if v_end < v_inf and a_end > 0.01:
        ratio = max(1 - v_end / v_inf, 0.02)
        k_x   = max(a_end / (v_inf * ratio), 0.05)
        t_eq  = -np.log(max(ratio, 0.001)) / k_x
        t_99  = min(-np.log(0.01) / k_x - t_eq, 5.0)
        t_ext = np.linspace(0, t_99, 200)
        v_ext = v_inf * (1 - np.exp(-k_x * (t_ext + t_eq)))
        m_ext = v_ext > v_end
        if m_ext.any():
            ax.plot(ts[-1] + t_ext[m_ext], v_ext[m_ext],
                    color=C_EXTRAP, lw=2, ls="--",
                    label=f"Extrapolation -> {v_inf:.2f} m/s")
            t_cursor = ts[-1] + t_ext[m_ext][-1]
        else:
            t_cursor = ts[-1]
    else:
        t_cursor = ts[-1]

# --- Phase 2: Mit Geschwindigkeit -> horizontales Band bei v_max ---
for r, c, lbl in [(r_s1, C_SPEED1, "Mit Geschw. V1"),
                  (r_s2, C_SPEED2, "Mit Geschw. V2")]:
    if r is None:
        continue
    t_off = t_cursor + GAP
    t_end = t_off + r['t_span']
    # Shaded band: ±10 % als Messungenauigkeit
    band = max(r['v_std'], r['v_max'] * 0.07)
    ax.fill_between([t_off, t_end],
                    r['v_max'] - band, r['v_max'] + band,
                    alpha=0.18, color=c)
    ax.hlines(r['v_max'], t_off, t_end, color=c, lw=2.5,
              label=f"{lbl}  (v_max={r['v_max']:.2f} m/s)")
    t_cursor = t_end

# --- Phase 3: Bremsung ---
for r, c, lbl in [(r_b1, C_BRK1, "Bremsen V1"),
                  (r_b2, C_BRK2, "Bremsen V2")]:
    if r is None:
        continue
    t_off = t_cursor + GAP
    # Rohe SG-Daten NUR ab Peak (transparent, als Punktewolke)
    ax.plot(r['t_raw_brk'] + t_off, r['v_raw_brk'],
            color=c, lw=0, marker='.', ms=2.5, alpha=0.35, zorder=2)
    # Gefittete lineare Bremskurve
    ax.fill_between(r['t_plot'] + t_off, 0, r['v_plot'],
                    alpha=0.13, color=c)
    ax.plot(r['t_plot'] + t_off, r['v_plot'],
            color=c, lw=2.8, zorder=3,
            label=f"{lbl}  (a={r['a_brake']:.2f} m/s²)")
    t_cursor = t_off + r['t_plot'][-1]

# Referenzlinien
ax.axhline(v_max_meas, color=C_VMAX, lw=1.3, ls="-.", alpha=0.8)
ax.text(0.01, v_max_meas + 0.04,
        f"v_max (gemessen) = {v_max_meas:.2f} m/s",
        color=C_VMAX, fontsize=8.5, transform=ax.get_yaxis_transform())
ax.axhline(V_MAX_LIT, color="#e65100", lw=1.0, ls=":", alpha=0.6)
ax.text(0.01, V_MAX_LIT + 0.04,
        f"v_max (Literatur) = {V_MAX_LIT} m/s",
        color="#e65100", fontsize=8.5, transform=ax.get_yaxis_transform())

ax.set_xlabel("Zeit (relativ, s)")
ax.set_ylabel("Geschwindigkeit (m/s)")
ax.set_ylim(-0.05, V_MAX_LIT * 1.20)
ax.set_xlim(-0.3, t_cursor + 0.5)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(True, alpha=0.3)

# Parameterbox
if a_brake_mean:
    box = (
        f"Simulationsparameter:\n"
        f"  v_max (gemessen)  = {v_max_meas:.2f} m/s\n"
        f"  v_max (Literatur) = {V_MAX_LIT:.1f} m/s\n"
        f"  a_brake (Mittel)  = {a_brake_mean:.3f} m/s^2"
    )
    ax.text(0.98, 0.25, box, transform=ax.transAxes, fontsize=8.5,
            va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.5",
                      fc="lightyellow", ec="gray", alpha=0.92))

plt.tight_layout()
ts  = datetime.now().strftime("%H%M%S")
out = OUTDIR / f"kinematik_v8_{ts}.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot: {out}")
