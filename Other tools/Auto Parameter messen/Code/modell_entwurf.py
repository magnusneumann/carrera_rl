"""
Modellentwurf aus den Messwerten – NUR Zahlen und Zeichnung.
Die Simulation (carrera_2d_env.py) wird hier bewusst NICHT angefasst.

Messwerte aus analyse_v10 / kinematik_v10_104532.png:
    aus dem Stand : 0 -> 1.258 m/s in 4.07 s   (mittleres a = 0.309)
    Durchfahrt V1 : 0.982 -> 1.844 m/s,  a = +0.488 m/s^2  bei v_m = 1.41
    Durchfahrt V2 : 0.698 -> 1.418 m/s,  a = +0.313 m/s^2  bei v_m = 1.06
    Bremsen V1    : a = 0.723 +/- 0.011 m/s^2  ab 1.507 m/s
    Bremsen V2    : a = 0.627 +/- 0.018 m/s^2  ab 1.221 m/s

Wichtig fuer die Modellwahl: die beiden Durchfahrten unterscheiden sich um
44 % voneinander (0.313 vs 0.488). Kein Modell kann beide besser als auf
etwa +/-22 % treffen – das ist die Streuung der Messung selbst, nicht ein
Mangel des Modells.
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

OUTDIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Messwerte
# ---------------------------------------------------------------------------
M_STAND_V, M_STAND_T = 1.258, 4.07
M_ACC = [("Durchfahrt V1", 1.413, 0.488),      # (Name, v_mitte, a)
         ("Durchfahrt V2", 1.058, 0.313)]
M_BRK = [("Bremsen V1", 1.507, 0.723, 0.011),  # (Name, v_start, a, sigma)
         ("Bremsen V2", 1.221, 0.627, 0.018)]
V_MAX = 1.90        # knapp ueber der hoechsten Messung (1.844 m/s)

# ---------------------------------------------------------------------------
# MODELL 1 – Bremsen.  Zwei Messpunkte, zwei Parameter -> exakt bestimmt.
#   a_brems(v) = B0 + B1*v
# Der Unterschied zwischen V1 und V2 erklaert sich damit vollstaendig als
# Geschwindigkeitsabhaengigkeit statt als Messstreuung.
# ---------------------------------------------------------------------------
B1 = (M_BRK[0][2] - M_BRK[1][2]) / (M_BRK[0][1] - M_BRK[1][1])
B0 = M_BRK[0][2] - B1 * M_BRK[0][1]
B_KONST = 0.5 * (M_BRK[0][2] + M_BRK[1][2])     # einfache Variante

# ---------------------------------------------------------------------------
# MODELL 2 – Gas.  Zwei Varianten.
#
# 2a  konstant:  a = A_KONST, hart gedeckelt bei V_MAX
#     Einfachst moeglich, liegt bei allen drei Messungen innerhalb +/-28 %.
#
# 2b  zweiteilig: unten haftungsbegrenzt (die Raeder drehen beim Anfahren
#     durch, die Beschleunigung STEIGT waehrend das Auto Grip aufbaut),
#     oben motorbegrenzt (Gegen-EMK, a faellt auf 0 bei V_MAX).
#         a(v) = min( G0 + G1*v ,  GM*(1 - v/V_MAX) )
#     Das bildet den gemessenen Anstieg ab, den 2a nicht kann.
# ---------------------------------------------------------------------------
A_KONST = 0.40
G0, G1, GM = 0.25, 0.17, 2.00

acc_konst = lambda v: np.where(v < V_MAX, A_KONST, 0.0)
acc_zwei  = lambda v: np.maximum(
    np.minimum(G0 + G1 * v, GM * (1.0 - v / V_MAX)), 0.0)
brems_lin = lambda v: -(B0 + B1 * v)
brems_kon = lambda v: -B_KONST

# bisheriges Modell: force = gas*1.0 - brake*0.4 ; dv/dt = (force-0.5v)/0.080
OLD_M, OLD_FG, OLD_FB, OLD_C, OLD_CLIP = 0.080, 1.0, 0.4, 0.5, 1.6
old_acc   = lambda v: (OLD_FG - OLD_C * v) / OLD_M
old_brems = lambda v: -(OLD_FB + OLD_C * v) / OLD_M


def integrate(fn, v0, t_end, dt=0.001, ceil=None):
    t, v = [0.0], [float(v0)]
    while t[-1] < t_end:
        vn = max(v[-1] + float(fn(v[-1])) * dt, 0.0)
        if ceil is not None:
            vn = min(vn, ceil)
        v.append(vn); t.append(t[-1] + dt)
    return np.array(t), np.array(v)


def zeit_bis(fn, v_ziel, ceil=None):
    t, v = integrate(fn, 0.0, 30.0, ceil=ceil)
    i = np.argmax(v >= v_ziel)
    return t[i] if v[-1] >= v_ziel else np.inf


# ---------------------------------------------------------------------------
print("=" * 74)
print("MODELL BREMSEN")
print("=" * 74)
print(f"  linear : a(v) = {B0:.3f} + {B1:.3f}*v        [m/s^2]")
print(f"  konstant: a    = {B_KONST:.2f}                    [m/s^2]")
print("\n  Abgleich:")
for name, v0, a_m, a_e in M_BRK:
    print(f"    {name}: v={v0:.2f}  ->  linear {B0+B1*v0:.3f} | "
          f"konstant {B_KONST:.3f} | gemessen {a_m:.3f} +/- {a_e:.3f}")
print("\n  Bremsweg und -zeit (lineares Modell):")
for v0 in (1.0, 1.5, V_MAX):
    tb, vb = integrate(brems_lin, v0, 10.0)
    i = int(np.argmax(vb <= 1e-9))
    print(f"    aus {v0:.2f} m/s: {tb[i]:.2f} s, "
          f"{np.trapezoid(vb[:i+1], tb[:i+1]):.2f} m")

print()
print("=" * 74)
print("MODELL GAS")
print("=" * 74)
print(f"  2a konstant : a = {A_KONST:.2f} m/s^2, hart gedeckelt bei "
      f"v = {V_MAX:.2f} m/s")
print(f"  2b zweiteilig: a(v) = min({G0:.2f} + {G1:.2f}*v ,  "
      f"{GM:.2f}*(1 - v/{V_MAX:.2f}))")
v_cross = (GM - G0) / (G1 + GM / V_MAX)
print(f"     Umschlagpunkt bei v = {v_cross:.2f} m/s "
      f"(darunter Haftung, darueber Motor)")
print("\n  Abgleich:")
t_k = zeit_bis(acc_konst, M_STAND_V, ceil=V_MAX)
t_z = zeit_bis(acc_zwei,  M_STAND_V)
print(f"    aus dem Stand auf {M_STAND_V:.2f} m/s: "
      f"2a {t_k:.2f} s ({(t_k/M_STAND_T-1)*100:+.0f} %) | "
      f"2b {t_z:.2f} s ({(t_z/M_STAND_T-1)*100:+.0f} %) | "
      f"gemessen {M_STAND_T:.2f} s")
for name, v_m, a_m in M_ACC:
    print(f"    {name} bei v={v_m:.2f}: "
          f"2a {A_KONST:.3f} ({(A_KONST/a_m-1)*100:+.0f} %) | "
          f"2b {float(acc_zwei(v_m)):.3f} "
          f"({(float(acc_zwei(v_m))/a_m-1)*100:+.0f} %) | "
          f"gemessen {a_m:.3f}")
print(f"\n    0 -> {V_MAX:.2f} m/s: 2a {V_MAX/A_KONST:.2f} s | "
      f"2b {zeit_bis(acc_zwei, 0.99*V_MAX):.2f} s")

print()
print("=" * 74)
print("VERGLEICH MIT DEM BISHERIGEN MODELL")
print("=" * 74)
print(f"  Gas     a(0)   : bisher {old_acc(0):6.2f} | neu {A_KONST:5.2f}"
      f"   -> Faktor {old_acc(0)/A_KONST:.0f} zu gross")
print(f"  Bremse  a(1.5) : bisher {-old_brems(1.5):6.2f} | "
      f"neu {B0+B1*1.5:5.2f}   -> Faktor {-old_brems(1.5)/(B0+B1*1.5):.0f} zu gross")
t_old = zeit_bis(old_acc, 0.99 * OLD_CLIP, ceil=OLD_CLIP)
print(f"  0 -> v_max     : bisher {t_old:5.2f} s | neu {V_MAX/A_KONST:5.2f} s"
      f"   -> Faktor {(V_MAX/A_KONST)/t_old:.0f} zu schnell")
tb_o, vb_o = integrate(old_brems, V_MAX, 4.0)
tb_n, vb_n = integrate(brems_lin, V_MAX, 6.0)
i_o = int(np.argmax(vb_o <= 1e-9)); i_n = int(np.argmax(vb_n <= 1e-9))
print(f"  Bremsen v_max->0: bisher {tb_o[i_o]:5.2f} s | neu {tb_n[i_n]:5.2f} s"
      f"   -> Faktor {tb_n[i_n]/tb_o[i_o]:.0f} zu schnell")
print(f"  v_max          : bisher {OLD_FG/OLD_C:.2f} (Clip {OLD_CLIP:.1f}) | "
      f"neu {V_MAX:.2f}   -> war schon richtig")
print("\n  Der Fehler liegt nicht bei v_max, sondern bei allen Zeitkonstanten.")
print("  Ursache: 0.080 kg Masse gegen Kraefte von 1.0 N ergibt 12.5 m/s^2.")

# ---------------------------------------------------------------------------
# Zeichnung
# ---------------------------------------------------------------------------
C_2A, C_2B, C_OLD, C_MES = "#1b5e20", "#7cb342", "#b71c1c", "#0d47a1"
fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8))
fig.suptitle("Carrera Hybrid – Modellentwurf aus den Messwerten "
             "(Vorschlag, noch nicht umgesetzt)",
             fontsize=13, fontweight="bold")

# --- links: Beschleunigung bis v_max ---
ax = axes[0]
ax.set_title("A · Beschleunigung aus dem Stand bis v_max",
             fontsize=10.5, loc="left", fontweight="bold")
for fn, c, lbl, ce in [(acc_konst, C_2A, f"2a konstant {A_KONST} m/s²", V_MAX),
                       (acc_zwei,  C_2B, "2b zweiteilig", None)]:
    t, v = integrate(fn, 0.0, 9.0, ceil=ce)
    ax.plot(t, v, color=c, lw=2.6, label=lbl)
t, v = integrate(old_acc, 0.0, 9.0, ceil=OLD_CLIP)
ax.plot(t, v, color=C_OLD, lw=2.0, ls="--", label="bisheriges Modell")
ax.plot([M_STAND_T], [M_STAND_V], marker="o", ms=11, mfc="white",
        mec=C_MES, mew=2.5, zorder=6)
ax.annotate(f"gemessen: {M_STAND_V:.2f} m/s\nnach {M_STAND_T:.1f} s",
            xy=(M_STAND_T, M_STAND_V), xytext=(M_STAND_T + 0.9, 0.72),
            fontsize=9, color=C_MES,
            arrowprops=dict(arrowstyle="->", color=C_MES, lw=1.5))
ax.axhline(V_MAX, color="#37474f", lw=1.0, ls=":", alpha=0.8)
ax.text(8.9, V_MAX + 0.03, f"v_max = {V_MAX} m/s", ha="right", fontsize=8.5,
        color="#37474f")
ax.set_xlabel("Zeit (s)"); ax.set_ylabel("Geschwindigkeit (m/s)")
ax.set_xlim(0, 9); ax.set_ylim(0, 2.05)
ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.3)

# --- mitte: Bremsen ---
ax = axes[1]
ax.set_title("B · Bremsen aus v_max", fontsize=10.5, loc="left",
             fontweight="bold")
t, v = integrate(brems_lin, V_MAX, 5.0)
ax.plot(t, v, color=C_2A, lw=2.6,
        label=f"neu linear  a = {B0:.2f} + {B1:.2f}·v")
t, v = integrate(brems_kon, V_MAX, 5.0)
ax.plot(t, v, color=C_2B, lw=2.0, ls="-.",
        label=f"neu konstant  a = {B_KONST:.2f}")
t, v = integrate(old_brems, V_MAX, 5.0)
ax.plot(t, v, color=C_OLD, lw=2.0, ls="--", label="bisheriges Modell")
for (name, v0, a_m, _), c in zip(M_BRK, ["#0d47a1", "#42a5f5"]):
    tm = np.linspace(0, v0 / a_m, 40)
    ax.plot(tm, v0 - a_m * tm, color=c, lw=2.0, alpha=0.9,
            label=f"{name} gemessen")
ax.set_xlabel("Zeit seit Bremsbeginn (s)")
ax.set_ylabel("Geschwindigkeit (m/s)")
ax.set_xlim(0, 4.4); ax.set_ylim(0, 2.05)
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

# --- rechts: a(v) ---
ax = axes[2]
ax.set_title("C · Kennlinien a(v) – hier steckt der ganze Unterschied",
             fontsize=10.5, loc="left", fontweight="bold")
vs = np.linspace(0, V_MAX, 300)
ax.plot(vs, acc_konst(vs), color=C_2A, lw=2.6, label="2a Gas konstant")
ax.plot(vs, acc_zwei(vs), color=C_2B, lw=2.6, label="2b Gas zweiteilig")
ax.plot(vs, brems_lin(vs), color=C_2A, lw=2.6, ls="-.", label="Bremse neu")
ax.plot(vs, old_acc(vs), color=C_OLD, lw=1.8, ls="--", label="Gas bisher")
ax.plot(vs, old_brems(vs), color=C_OLD, lw=1.8, ls=":", label="Bremse bisher")
for name, v_m, a_m in M_ACC:
    ax.plot([v_m], [a_m], marker="o", ms=9, mfc="white", mec=C_MES, mew=2.2,
            zorder=6)
for name, v0, a_m, a_e in M_BRK:
    ax.errorbar([v0], [-a_m], yerr=[a_e], marker="s", ms=8, mfc="white",
                mec=C_MES, mew=2.2, ecolor=C_MES, capsize=3, zorder=6)
ax.axvline(v_cross, color=C_2B, lw=1.0, ls=":", alpha=0.7)
ax.text(v_cross - 0.04, 11, "Haftung |  Motor", rotation=90, fontsize=7.5,
        color=C_2B, ha="right", va="center")
ax.axhline(0, color="black", lw=0.9)
ax.set_yscale("symlog", linthresh=1.0)
ax.set_yticks([-15, -5, -1, 0, 1, 5, 15])
ax.set_yticklabels(["-15", "-5", "-1", "0", "1", "5", "15"])
ax.set_xlabel("Geschwindigkeit (m/s)")
ax.set_ylabel("Beschleunigung (m/s²)   – symlog-Achse")
ax.text(0.03, 0.06, "Kreise = Gas-Messungen\nQuadrate = Brems-Messungen",
        transform=ax.transAxes, fontsize=8.5, color=C_MES, va="bottom")
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = OUTDIR / f"modell_entwurf_{datetime.now().strftime('%H%M%S')}.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot: {out}")
