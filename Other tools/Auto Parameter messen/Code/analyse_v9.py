"""
Kinematik-Analyse v9

Aenderungen ggue. v8:
1. Y_MIN=30: Oberste 30 Pixel abschneiden (Beine / falsches Objekt y=14).
2. analyse_brake: kein i_stop, trailing-Trim (n<300), SG-Rand-Trim
   (margin = SG_WIN//2 = 10 Frames) auf beiden Seiten bevor v_peak
   gesucht wird => keine SG-Randeffekte mehr in v_peak / a_brake.
3. Mit Geschwindigkeit: echte SG-v(t)-Kurve (Mitte-Trim) + v_max gestrichelt.
4. Simulationsphysik-Linie als Vergleich bei Bremsung.
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

PX_PER_M    = 400.0
X_MIN       = 700
X_MAX       = 1100
Y_MIN       = 30
DIFF_THRESH = 25
N_MIN_PX    = 5
N_MAX_PX    = 2000
N_VALID     = 300
MAX_JUMP_PX = 60
SG_WIN      = 21
SG_MARGIN   = SG_WIN // 2   # = 10 Frames von jedem Ende abschneiden

SIM_MASS        = 0.080
SIM_BRAKE_FORCE = 0.4
SIM_DRAG        = 0.5


def track_video(path):
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
        roi  = diff[Y_MIN:, X_MIN:X_MAX]
        mask = roi > DIFF_THRESH
        n    = int(np.count_nonzero(mask))
        if N_MIN_PX <= n <= N_MAX_PX:
            ys, _ = np.where(mask)
            raw.append((fi / fps, float(np.median(ys)) + Y_MIN, n))
    cap.release()

    arr = np.array(raw) if len(raw) >= 5 else None
    if arr is not None:
        print(f"  Tracking: {len(arr)}/{total} Frames, "
              f"y=[{arr[:,1].min():.0f}..{arr[:,1].max():.0f}]px, "
              f"n=[{arr[:,2].min():.0f}..{arr[:,2].max():.0f}]")
    else:
        print("  Tracking: KEINE Detektion")
    return arr


def remove_jumps(t, y):
    keep = np.ones(len(t), bool)
    for i in range(1, len(t) - 1):
        if (abs(y[i] - y[i-1]) > MAX_JUMP_PX and
                abs(y[i+1] - y[i]) > MAX_JUMP_PX):
            keep[i] = False
    return keep


def clip_forward(t, y):
    turn = int(np.argmax(y))
    if turn < len(y) - 1:
        return t[:turn + 1], y[:turn + 1]
    return t, y


def velocity_sg(t, y_px, win=SG_WIN):
    p  = y_px / PX_PER_M
    dt = float(np.median(np.diff(t)))
    t_uni = np.linspace(t[0], t[-1], len(t))
    p_uni = np.interp(t_uni, t, p)
    w = min(win, len(p_uni))
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return t_uni, np.abs(np.gradient(p_uni, dt))
    v = savgol_filter(p_uni, w, 3, deriv=1, delta=dt)
    return t_uni, np.abs(v)


def sim_brake_curve(v0, t_max=3.0, dt=1/30.0):
    t_list, v_list = [0.0], [float(v0)]
    v, t = float(v0), 0.0
    while v > 0.001 and t < t_max:
        dv = (-SIM_BRAKE_FORCE - SIM_DRAG * v) / SIM_MASS * dt
        v  = max(v + dv, 0.0)
        t += dt
        t_list.append(t)
        v_list.append(v)
    return np.array(t_list), np.array(v_list)


def analyse_accel(raw):
    t, y, n = raw[:, 0], raw[:, 1], raw[:, 2]
    keep = remove_jumps(t, y)
    t, y = t[keep], y[keep]
    t, y = clip_forward(t, y)
    p    = y / PX_PER_M
    dp   = np.abs(np.diff(p)) / np.diff(t)
    i0   = max(1, int(np.argmax(dp > 0.05)))
    t_use = t[i0:] - t[i0]
    p_use = p[i0:] - p[i0]
    if len(t_use) < 10:
        return None
    A = np.column_stack([t_use**2, t_use**3, t_use**4])
    c, _, _, _ = np.linalg.lstsq(A, p_use, rcond=None)
    c2, c3, c4 = c
    def pf(ta): return c2*ta**2 + c3*ta**3 + c4*ta**4
    def vf(ta): return 2*c2*ta + 3*c3*ta**2 + 4*c4*ta**3
    t_sm   = np.linspace(0, t_use[-1], 500)
    v_sm   = np.clip([vf(ti) for ti in t_sm], 0, None)
    v_peak = float(v_sm.max())
    rmse   = float(np.sqrt(np.mean((p_use - pf(t_use))**2)))
    print(f"  v(0)=0  v_peak={v_peak:.3f} m/s  RMSE={rmse*100:.2f}cm")
    return dict(t=t_use, p=p_use, pf=pf, vf=vf, rmse=rmse,
                t_sm=t_sm, v_sm=v_sm, v_peak=v_peak)


def analyse_speed(raw, n_min=N_VALID):
    t, y, n = raw[:, 0], raw[:, 1], raw[:, 2]
    keep = remove_jumps(t, y)
    t, y, n = t[keep], y[keep], n[keep]
    t, y = clip_forward(t, y)
    mask = n[:len(t)] >= n_min
    if mask.sum() < 8:
        mask = np.ones(len(t), bool)
    t, y = t[mask], y[mask]
    if len(t) < 8:
        return None
    t = t - t[0]
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
    lo   = 0.75 * t[-1]
    last = t >= lo
    if last.sum() < 3:
        last = t >= 0.50 * t[-1]
    c     = np.polyfit(t[last], y[last], 1)
    v_max = float(abs(c[0]) / PX_PER_M)
    resid = y[last] - np.polyval(c, t[last])
    dt_l  = float(np.median(np.diff(t[last]))) if last.sum() > 1 else 0.033
    v_std = float(np.std(resid) / (PX_PER_M * max(dt_l, 1e-6)))

    sg_win_use = min(SG_WIN, max(5, len(t) - 1))
    if sg_win_use % 2 == 0:
        sg_win_use -= 1
    t_sg, v_sg = velocity_sg(t, y, win=sg_win_use)
    margin = max(2, sg_win_use // 2)
    if len(t_sg) > 2 * margin + 3:
        t_sg = t_sg[margin:-margin]
        v_sg = v_sg[margin:-margin]

    print(f"  v_max={v_max:.3f} m/s  (i0={i0}, {mask.sum()} Frames)")
    return dict(t_span=float(t[-1]), v_max=v_max,
                v_std=min(v_std, v_max * 0.12),
                t_sg=t_sg, v_sg=v_sg)


def analyse_brake(raw, n_min=N_VALID):
    t, y, n = raw[:, 0], raw[:, 1], raw[:, 2]
    keep = remove_jumps(t, y)
    t, y, n = t[keep], y[keep], n[keep]
    t, y = clip_forward(t, y)

    n_arr = n[:len(t)]
    good  = n_arr >= n_min

    if good.sum() >= 5:
        i0        = int(np.where(good)[0][0])
        last_good = int(np.where(good)[0][-1])
    else:
        i0        = sum(1 for yi in y[:15] if yi < Y_MIN + 15)
        last_good = len(t) - 1

    # Trailing-Trim: Exit-Frames (n < n_min) am Ende entfernen
    t_use = t[i0:last_good + 1] - t[i0]
    y_use = y[i0:last_good + 1]

    if len(t_use) < 2 * SG_MARGIN + 10:
        return None

    # SG-Geschwindigkeit auf vollen Daten (fuer Scatter-Anzeige)
    t_sg_full, v_sg_full = velocity_sg(t_use, y_use)

    # Rand-Trim: erste + letzte SG_MARGIN Frames verwerfen
    # => kein SG-Randeffekt mehr in v_peak und a_brake
    t_mid = t_sg_full[SG_MARGIN:-SG_MARGIN]
    v_mid = v_sg_full[SG_MARGIN:-SG_MARGIN]

    peak_i = int(np.argmax(v_mid))
    v_peak = float(v_mid[peak_i])
    t_peak = float(t_mid[peak_i])
    if v_peak < 0.15:
        return None

    # Bremsphase: ab v_peak bis Ende des getrimmten Bereichs
    brake = t_mid >= t_peak
    t_b   = t_mid[brake] - t_peak
    v_b   = v_mid[brake]

    if len(t_b) < 5:
        return None

    def v_lin(tt, a):
        return np.maximum(v_peak - a * tt, 0)

    try:
        popt, _ = curve_fit(v_lin, t_b, v_b, p0=[0.6],
                            bounds=([0.01], [20.]))
        a_brake = float(popt[0])
    except Exception:
        a_brake = max((v_peak - float(v_b[-1])) / (float(t_b[-1]) + 1e-9), 0.05)

    t_data_end = float(t_b[-1])
    v_end      = max(float(v_b[-1]), 0.0)
    t_stop     = v_peak / max(a_brake, 0.001)

    t_solid = np.linspace(0, min(t_data_end, t_stop), 300)
    v_solid = np.maximum(v_peak - a_brake * t_solid, 0)

    if v_end > 0.08:
        # Extrapolation maximal 2 s über Messdaten-Ende hinaus
        t_ext_end = min(t_stop + 0.05, t_data_end + 2.0)
        t_ext = np.linspace(t_data_end, t_ext_end, 100)
        v_ext = np.maximum(v_peak - a_brake * t_ext, 0)
    else:
        t_ext = np.array([t_data_end])
        v_ext = np.array([0.0])

    t_sim, v_sim = sim_brake_curve(v_peak)

    print(f"  i0={i0} last={last_good}  v_peak={v_peak:.3f} m/s  "
          f"a_brake={a_brake:.3f} m/s²  t_data={t_data_end:.2f}s  "
          f"v_end={v_end:.3f} m/s")

    return dict(v_peak=v_peak, a_brake=a_brake,
                t_raw_brk=t_sg_full, v_raw_brk=v_sg_full,
                t_solid=t_solid, v_solid=v_solid,
                t_ext=t_ext, v_ext=v_ext,
                t_sim=t_sim, v_sim=v_sim,
                t_stop=t_stop, t_data_end=t_data_end, v_end=v_end)


# ---------------------------------------------------------------------------
print("=" * 65)
print("ANALYSE v9  (Y_MIN={}, SG-Rand-Trim, kein i_stop)".format(Y_MIN))
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
a_brake_mean = float(np.mean(a_brake_list)) if a_brake_list else 0.6
V_MAX_LIT    = 1.8

print("\n" + "=" * 65)
print("ERGEBNISSE")
print("=" * 65)
if r_a0:
    print(f"  v_peak (Beschl.):  {r_a0['v_peak']:.2f} m/s  "
          f"RMSE={r_a0['rmse']*100:.2f}cm")
if v_max_list:
    print(f"  v_max gemessen:    {v_max_meas:.2f} m/s  "
          + ", ".join(f"V{i+1}={v:.2f}" for i, v in enumerate(v_max_list)))
if a_brake_list:
    print(f"  a_brake:           {a_brake_mean:.3f} m/s²  "
          + ", ".join(f"V{i+1}={a:.3f}" for i, a in enumerate(a_brake_list)))

t_sim_ref, _ = sim_brake_curve(v_max_meas)
print(f"\n  Sim-Physik: v={v_max_meas:.2f} m/s -> Stopp in "
      f"{t_sim_ref[-1]:.3f}s  (gem. ~{v_max_meas/a_brake_mean:.2f}s)")


# ---------------------------------------------------------------------------
# Plot
C_ACCEL  = "#1b5e20"
C_EXTRAP = "#558b2f"
C_SPEED1 = "#0d47a1"
C_SPEED2 = "#1565c0"
C_BRK1   = "#b71c1c"
C_BRK2   = "#e53935"
C_VMAX   = "#01579b"
C_SIM    = "#546e7a"

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
fig.suptitle(
    "Carrera Hybrid – Kinematik (v9)\n"
    f"14 px = 35 mm  |  v_max gemessen: {v_max_meas:.2f} m/s  |  "
    f"v_max Literatur: {V_MAX_LIT} m/s",
    fontsize=11, fontweight="bold"
)

# Panel 1: Position
ax = axes[0]
ax.set_title("Position – Beschleunigung von 0", fontsize=10)
if r_a0:
    ax.scatter(r_a0['t'], r_a0['p'], s=4, color=C_ACCEL, alpha=0.4,
               zorder=2, label="Rohdaten")
    t_fit = np.linspace(0, r_a0['t'][-1], 400)
    ax.plot(t_fit, r_a0['pf'](t_fit), color=C_ACCEL, lw=2.5, zorder=3,
            label=f"Polynom-Fit  (RMSE={r_a0['rmse']*100:.2f} cm)")
ax.set_xlabel("Zeit (s)")
ax.set_ylabel("Position (m)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: v(t)
ax = axes[1]
ax.set_title("Geschwindigkeit  v(t)  –  alle Messungen", fontsize=10)
GAP = 0.5
t_cursor = 0.0
sim_label_done = False

if r_a0:
    ts, vs = r_a0['t_sm'], r_a0['v_sm']
    ax.fill_between(ts, 0, vs, alpha=0.12, color=C_ACCEL)
    ax.plot(ts, vs, color=C_ACCEL, lw=2.5,
            label=f"Beschl. von 0  (v_peak={r_a0['v_peak']:.2f} m/s)")
    v_end = float(r_a0['vf'](ts[-1]))
    dv    = float(r_a0['vf'](ts[-1]) - r_a0['vf'](max(ts[-1] - 0.1, 0)))
    a_end = dv / 0.1
    v_inf = v_max_meas
    if v_end < v_inf and a_end > 0.01:
        ratio = max(1 - v_end / v_inf, 0.02)
        k_x   = max(a_end / (v_inf * ratio), 0.05)
        t_ext = np.linspace(0, min(-np.log(0.01) / k_x, 5.0), 200)
        v_ext = v_inf * (1 - np.exp(-k_x * t_ext))
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

for r, c, lbl in [(r_s1, C_SPEED1, "Mit Geschw. V1"),
                  (r_s2, C_SPEED2, "Mit Geschw. V2")]:
    if r is None:
        continue
    t_off = t_cursor + GAP
    t_end = t_off + r['t_span']
    ax.plot(r['t_sg'] + t_off, r['v_sg'],
            color=c, lw=1.8, alpha=0.80,
            label=f"{lbl}  v_max={r['v_max']:.2f} m/s")
    ax.hlines(r['v_max'], t_off, t_end, color=c, lw=1.3, ls='--', alpha=0.65)
    t_cursor = t_end

for r, c, lbl in [(r_b1, C_BRK1, "Bremsen V1"),
                  (r_b2, C_BRK2, "Bremsen V2")]:
    if r is None:
        continue
    t_off = t_cursor + GAP

    # Scatter: volle SG-Kurve als Rohdaten
    ax.plot(r['t_raw_brk'] + t_off, r['v_raw_brk'],
            color=c, lw=0, marker='.', ms=2.5, alpha=0.35, zorder=2)

    # Gefittete Bremskurve
    ax.fill_between(r['t_solid'] + t_off, 0, r['v_solid'], alpha=0.13, color=c)
    ax.plot(r['t_solid'] + t_off, r['v_solid'],
            color=c, lw=2.8, zorder=3,
            label=f"{lbl}  v0={r['v_peak']:.2f} m/s  a={r['a_brake']:.2f} m/s²")

    # Extrapolation
    if r['v_ext'][0] > 0.08:
        ax.plot(r['t_ext'] + t_off, r['v_ext'],
                color=c, lw=1.5, ls='--', alpha=0.55)

    # Simulationsphysik
    sim_lbl = "Sim-Physik (alt)" if not sim_label_done else "_nolegend_"
    sim_label_done = True
    ax.plot(r['t_sim'] + t_off, r['v_sim'],
            color=C_SIM, lw=2.2, ls=':', alpha=0.85, zorder=4, label=sim_lbl)

    t_cursor = t_off + max(
        r['t_ext'][-1] if r['v_ext'][0] > 0.08 else r['t_data_end'],
        r['t_sim'][-1] + 0.1)

ax.axhline(v_max_meas, color=C_VMAX, lw=1.3, ls="-.", alpha=0.8)
ax.text(0.01, v_max_meas + 0.04, f"v_max (gem.) = {v_max_meas:.2f} m/s",
        color=C_VMAX, fontsize=8.5, transform=ax.get_yaxis_transform())
ax.axhline(V_MAX_LIT, color="#e65100", lw=1.0, ls=":", alpha=0.6)
ax.text(0.01, V_MAX_LIT + 0.04, f"v_max (Lit.) = {V_MAX_LIT} m/s",
        color="#e65100", fontsize=8.5, transform=ax.get_yaxis_transform())

ax.set_xlabel("Zeit (relativ, s)")
ax.set_ylabel("Geschwindigkeit (m/s)")
ax.set_ylim(-0.05, V_MAX_LIT * 1.22)
ax.set_xlim(-0.3, t_cursor + 0.3)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(True, alpha=0.3)

if a_brake_mean:
    box = (f"Simulationsparameter:\n"
           f"  v_max (gem.)     = {v_max_meas:.2f} m/s\n"
           f"  v_max (Lit.)     = {V_MAX_LIT:.1f} m/s\n"
           f"  a_brake (Mittel) = {a_brake_mean:.3f} m/s²\n\n"
           f"Sim-Physik (alt):\n"
           f"  dv/dt=(-0.4-0.5v)/0.080\n"
           f"  Stopp in {t_sim_ref[-1]:.2f}s\n"
           f"  (gem. ~{v_max_meas/a_brake_mean:.1f}s)")
    ax.text(0.98, 0.28, box, transform=ax.transAxes, fontsize=8.0,
            va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                      ec="gray", alpha=0.92))

plt.tight_layout()
ts  = datetime.now().strftime("%H%M%S")
out = OUTDIR / f"kinematik_v9_{ts}.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot: {out}")
