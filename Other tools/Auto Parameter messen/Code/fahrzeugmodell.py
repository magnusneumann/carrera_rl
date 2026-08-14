"""
Laengsdynamikmodell Carrera Hybrid – ENTWURF, noch nicht in die Simulation
uebernommen. carrera_2d_env.py bleibt unveraeandert.

Grundgleichung
--------------
    dv/dt = gas * A_drive(v)  -  brake * A_BRAKE  -  R(v)

    R(v)       = R0 + R1*v            Roll- und Luftwiderstand, immer aktiv
    A_BRAKE    = const                reine Bremskraft
    A_drive(v) = min(zwei Aesten)     unten haftungs-, oben motorbegrenzt

Die drei Terme sind so kalibriert, dass sich bei Vollgas bzw. Vollbremsung
genau die gemessenen Kennlinien ergeben:

    Vollgas       a(v) = min(0.25 + 0.17*v,  2.00*(1 - v/1.90))
    Vollbremsung  a(v) = -(0.217 + 0.336*v)

Datenlage
---------
gemessen (analyse_v10, kinematik_v10_104532.png):
    aus dem Stand : 0 -> 1.258 m/s in 4.07 s
    Durchfahrt V1 : a = 0.488 m/s^2 bei v = 1.41
    Durchfahrt V2 : a = 0.313 m/s^2 bei v = 1.06
    Bremsen V1    : a = 0.723 +/- 0.011 m/s^2 bei v = 1.51
    Bremsen V2    : a = 0.627 +/- 0.018 m/s^2 bei v = 1.22

angenommen, NICHT gemessen:
    R0            Rollwiderstand im Stillstand
    oberer Ast    Motorbegrenzung und damit v_max
Beides ist unten gekennzeichnet.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------

V_MAX = 1.90        # m/s    ANNAHME: knapp ueber der hoechsten Messung (1.844)

# --- Widerstand (immer aktiv) ---------------------------------------------
R1 = 0.336          # 1/s    GEMESSEN: v-Anteil aus den beiden Bremsversuchen
R0 = 0.05           # m/s^2  ANNAHME: nie im Ausrollen gemessen. Beeinflusst
                    #        ausschliesslich das Rollverhalten ohne Eingabe –
                    #        Gas- und Bremsverhalten sind davon unabhaengig
                    #        (siehe Herleitung unten).

# --- Bremse ----------------------------------------------------------------
# Vollbremsung soll 0.217 + 0.336*v ergeben:
#   A_BRAKE + R0 + R1*v = 0.217 + 0.336*v   ->   A_BRAKE = 0.217 - R0
A_BRAKE = 0.217 - R0        # m/s^2   GEMESSEN (bis auf die Aufteilung mit R0)

# --- Antrieb ---------------------------------------------------------------
# Vollgas soll min(0.25 + 0.17*v, 2.00*(1 - v/V_MAX)) ergeben:
#   A_drive(v) - R0 - R1*v = gemessene Kennlinie
#   A_drive(v) = min(0.25 + 0.17*v, 2.00 - 1.0526*v) + R0 + R1*v
G0, G1 = 0.25, 0.17         # GEMESSEN: haftungsbegrenzter Ast
GM     = 2.00               # ANNAHME: motorbegrenzter Ast

DRIVE_LO_0 = G0 + R0                    # 0.300
DRIVE_LO_1 = G1 + R1                    # 0.506
DRIVE_HI_0 = GM + R0                    # 2.050
DRIVE_HI_1 = GM / V_MAX - R1            # 0.717   (als Abzug, s.u.)


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------

def a_drive(v):
    """Antriebsbeschleunigung bei Vollgas, ohne Widerstand."""
    v = np.asarray(v, float)
    haftung = DRIVE_LO_0 + DRIVE_LO_1 * v
    motor   = DRIVE_HI_0 - DRIVE_HI_1 * v
    return np.maximum(np.minimum(haftung, motor), 0.0)


def a_resist(v):
    """Roll- und Luftwiderstand. Wirkt immer entgegen der Fahrtrichtung."""
    v = np.asarray(v, float)
    return R0 + R1 * np.abs(v)


def accel(v, gas=0.0, brake=0.0):
    """Beschleunigung in m/s^2.  gas, brake jeweils 0..1.

    Widerstand und Bremse wirken der Bewegung entgegen; bei v=0 duerfen sie
    das Fahrzeug nicht rueckwaerts ziehen. Deshalb wird der bremsende Anteil
    auf das begrenzt, was v gerade noch auf 0 bringt (siehe step()).
    """
    v = np.asarray(v, float)
    vor  = gas * a_drive(v)
    zur  = brake * A_BRAKE + a_resist(v)
    return vor - np.sign(np.where(v == 0.0, 1.0, v)) * zur


def step(v, gas=0.0, brake=0.0, dt=1.0/30.0):
    """Ein Zeitschritt. Verhindert, dass Bremse oder Widerstand das
    Fahrzeug durch die Null hindurch rueckwaerts beschleunigen."""
    a  = float(accel(v, gas, brake))
    vn = v + a * dt
    if v > 0.0 and vn < 0.0 and gas == 0.0:
        vn = 0.0                      # steht, statt rueckwaerts zu rollen
    return float(np.clip(vn, 0.0, V_MAX))


def integrate(v0, gas, brake, t_end, dt=1.0/2000.0):
    t, v = [0.0], [float(v0)]
    while t[-1] < t_end:
        v.append(step(v[-1], gas, brake, dt))
        t.append(t[-1] + dt)
        if v[-1] <= 0.0 and gas == 0.0:
            break
    return np.array(t), np.array(v)


# geschlossene Formen (fuer Abschaetzungen ohne Integration) ----------------

def brems_zeit(v0):
    """Zeit bis Stillstand bei Vollbremsung.
       dv/dt = -(0.217 + 0.336 v)  ->  t = ln(1 + R1*v0/0.217)/R1"""
    return np.log(1.0 + R1 * v0 / (A_BRAKE + R0)) / R1


def brems_weg(v0):
    """Bremsweg bei Vollbremsung (analytisch)."""
    c = A_BRAKE + R0
    return (v0 - c * brems_zeit(v0) / 1.0) / R1 if R1 else v0**2 / (2 * c)


# ---------------------------------------------------------------------------
# Selbsttest gegen die Messungen
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime
    from pathlib import Path

    print("=" * 74)
    print("MODELLENTWURF – Abgleich mit den Messungen")
    print("=" * 74)
    print(f"  dv/dt = gas*A_drive(v) - brake*{A_BRAKE:.3f} - "
          f"({R0:.2f} + {R1:.3f}*v)")
    print(f"  A_drive(v) = min({DRIVE_LO_0:.3f} + {DRIVE_LO_1:.3f}*v,  "
          f"{DRIVE_HI_0:.3f} - {DRIVE_HI_1:.3f}*v)")

    print("\n  Vollgas (gas=1, brake=0):")
    for v_m, a_m, name in [(1.413, 0.488, "Durchfahrt V1"),
                           (1.058, 0.313, "Durchfahrt V2")]:
        a_mod = float(accel(v_m, gas=1.0))
        print(f"    {name}: v={v_m:.2f}  Modell {a_mod:6.3f}  "
              f"gemessen {a_m:.3f}   ({(a_mod/a_m-1)*100:+.0f} %)")

    t, v = integrate(0.0, 1.0, 0.0, 12.0)
    i = int(np.argmax(v >= 1.258))
    print(f"    aus dem Stand auf 1.258 m/s: Modell {t[i]:.2f} s, "
          f"gemessen 4.07 s   ({(t[i]/4.07-1)*100:+.0f} %)")
    i90 = int(np.argmax(v >= 0.99 * V_MAX))
    print(f"    0 -> {V_MAX:.2f} m/s: {t[i90]:.2f} s")
    print(f"    v_max erreicht: {v[-1]:.3f} m/s")

    print("\n  Vollbremsung (gas=0, brake=1):")
    for v0, a_m, s_m, name in [(1.507, 0.723, 0.011, "Bremsen V1"),
                               (1.221, 0.627, 0.018, "Bremsen V2")]:
        a_mod = -float(accel(v0, brake=1.0))
        print(f"    {name}: v={v0:.2f}  Modell {a_mod:6.3f}  "
              f"gemessen {a_m:.3f} +/- {s_m:.3f}")
    print()
    for v0 in (1.0, 1.5, V_MAX):
        tb, vb = integrate(v0, 0.0, 1.0, 10.0)
        weg = float(np.trapezoid(vb, tb))
        print(f"    aus {v0:.2f} m/s: {tb[-1]:.2f} s, {weg:.2f} m")

    print("\n  Ausrollen (gas=0, brake=0)  – R0 ist hier die einzige Annahme:")
    for v0 in (1.0, V_MAX):
        tr, vr = integrate(v0, 0.0, 0.0, 30.0)
        print(f"    aus {v0:.2f} m/s: {tr[-1]:.2f} s, "
              f"{float(np.trapezoid(vr, tr)):.2f} m")

    print("\n  Zeitschritt dt=1/30 s:")
    print(f"    groesste Geschwindigkeitsaenderung pro Schritt: "
          f"{max(abs(float(accel(0.0, gas=1.0))), abs(float(accel(V_MAX, brake=1.0))))/30:.4f} m/s")
    lam = max(DRIVE_LO_1 + R1, DRIVE_HI_1 + R1)
    print(f"    Stabilitaet expliziter Euler: dt*|d(dv/dt)/dv| = "
          f"{lam/30:.3f}  (unkritisch, < 2)")

    # -----------------------------------------------------------------------
    C_GAS, C_BRK, C_ROLL, C_MES = "#1b5e20", "#b71c1c", "#f9a825", "#0d47a1"
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8))
    fig.suptitle("Carrera Hybrid – Laengsdynamikmodell (Entwurf, "
                 "noch nicht umgesetzt)", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.set_title("A · Vollgas aus dem Stand", fontsize=10.5, loc="left",
                 fontweight="bold")
    t, v = integrate(0.0, 1.0, 0.0, 10.0)
    ax.plot(t, v, color=C_GAS, lw=2.8, label="Modell, gas = 1")
    ax.plot([4.07], [1.258], marker="o", ms=11, mfc="white", mec=C_MES,
            mew=2.5, zorder=6)
    ax.annotate("gemessen\n1.26 m/s nach 4.07 s", xy=(4.07, 1.258),
                xytext=(4.6, 0.72), fontsize=9, color=C_MES,
                arrowprops=dict(arrowstyle="->", color=C_MES, lw=1.5))
    ax.axhline(V_MAX, color="#37474f", lw=1.0, ls=":", alpha=0.8)
    ax.text(9.9, V_MAX + 0.03, f"v_max = {V_MAX} m/s", ha="right",
            fontsize=8.5, color="#37474f")
    ax.set_xlabel("Zeit (s)"); ax.set_ylabel("Geschwindigkeit (m/s)")
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.05)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("B · Verzoegern aus v_max", fontsize=10.5, loc="left",
                 fontweight="bold")
    t, v = integrate(V_MAX, 0.0, 1.0, 10.0)
    ax.plot(t, v, color=C_BRK, lw=2.8,
            label=f"Vollbremsung  ({t[-1]:.1f} s)")
    t, v = integrate(V_MAX, 0.0, 0.0, 30.0)
    ax.plot(t, v, color=C_ROLL, lw=2.4, ls="--",
            label=f"Ausrollen  ({t[-1]:.1f} s, R0 geschaetzt)")
    for (v0, a_m), c in zip([(1.507, 0.723), (1.221, 0.627)],
                            ["#0d47a1", "#42a5f5"]):
        tm = np.linspace(0, v0 / a_m, 40)
        ax.plot(tm, v0 - a_m * tm, color=c, lw=2.0, alpha=0.9,
                label=f"gemessen ab {v0:.2f} m/s")
    ax.set_xlabel("Zeit (s)"); ax.set_ylabel("Geschwindigkeit (m/s)")
    ax.set_xlim(0, 9); ax.set_ylim(0, 2.05)
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.set_title("C · Kennlinien a(v)", fontsize=10.5, loc="left",
                 fontweight="bold")
    vs = np.linspace(0, V_MAX, 300)
    ax.plot(vs, accel(vs, gas=1.0), color=C_GAS, lw=2.8, label="gas = 1")
    ax.plot(vs, accel(vs, gas=0.5), color=C_GAS, lw=1.6, ls="--",
            label="gas = 0.5")
    ax.plot(vs, accel(vs), color=C_ROLL, lw=2.2, ls="--", label="ausrollen")
    ax.plot(vs, accel(vs, brake=1.0), color=C_BRK, lw=2.8, label="brake = 1")
    for v_m, a_m in [(1.413, 0.488), (1.058, 0.313)]:
        ax.plot([v_m], [a_m], marker="o", ms=9, mfc="white", mec=C_MES,
                mew=2.2, zorder=6)
    for v0, a_m, s in [(1.507, 0.723, 0.011), (1.221, 0.627, 0.018)]:
        ax.errorbar([v0], [-a_m], yerr=[s], marker="s", ms=8, mfc="white",
                    mec=C_MES, mew=2.2, ecolor=C_MES, capsize=3, zorder=6)
    v_cross = (GM - G0) / (G1 + GM / V_MAX)
    ax.axvline(v_cross, color=C_GAS, lw=1.0, ls=":", alpha=0.7)
    ax.text(v_cross - 0.03, 0.62, "Haftung | Motor", rotation=90,
            fontsize=8, color=C_GAS, ha="right", va="center")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Geschwindigkeit (m/s)")
    ax.set_ylabel("Beschleunigung (m/s²)")
    ax.set_ylim(-0.95, 0.95)
    ax.text(0.03, 0.05, "Kreise = Gas-Messungen\nQuadrate = Brems-Messungen",
            transform=ax.transAxes, fontsize=8.5, color=C_MES, va="bottom")
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = (Path(__file__).parent /
           f"fahrzeugmodell_{datetime.now().strftime('%H%M%S')}.png")
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot: {out}")
