"""
Carrera Hybrid RC-Auto - Geschwindigkeit & Beschleunigung aus Video
====================================================================
Voraussetzungen:
  pip install opencv-python numpy scipy matplotlib

Kalibrierung:
  REAL_MM_PER_ARROW: Physische Länge der Pfeile (aus Schieblehre)
  TRACK_X_MIN/MAX:   Horizontale Pixel-Grenzen der Fahrspur im Video
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# KALIBRIERUNG - hier anpassen
# ============================================================
VIDEO_PATH = r"e:\Hochschule\Master\Carrera Hybrid\Code\VS Code\Carrera_rl\carrera_rl\Other tools\Auto Parameter messen\media\Anfahren von 0 und bremsen.mp4"

# Physische Länge der Pfeile (Schieblehre-Messung)
REAL_MM_BLAU  = 35.0   # mm - bitte mit tatsächlichem Wert ersetzen
REAL_MM_WEISS = 35.9   # mm - bitte mit tatsächlichem Wert ersetzen

# Pixelgrenzen der schwarzen Fahrspur (horizontal)
# → im ersten Frame abgelesen: ca. x=30 bis x=160
TRACK_X_MIN = 25
TRACK_X_MAX = 165

# Helligkeitsschwelle: Pixel heller als dieser Wert = Auto-Kandidat
# (schwarze Spur ~20, Auto ~120-200)
BRIGHTNESS_THRESHOLD = 80

# Kalibrierung: Pixellänge eines Pfeils MANUELL im Video messen
# Dazu Script einmal mit CALIBRATION_MODE=True starten → Pfeil anklicken
CALIBRATION_MODE = False
PIXELS_PER_MM    = 3.2   # vorläufiger Wert - wird durch Kalibrierung ersetzt

# Savitzky-Golay Glättung
SG_WINDOW = 11   # muss ungerade sein (Frame-Anzahl für Glättung)
SG_ORDER  = 3

# ============================================================
# SCHRITT 1: Kalibrierung (optional, interaktiv)
# ============================================================
def calibrate(video_path):
    """Klicke zwei Punkte an (Pfeilspitzen) → gibt Pixel/mm aus."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Kalibrierung", frame)
            if len(points) == 2:
                pixel_dist = np.hypot(points[1][0]-points[0][0], points[1][1]-points[0][1])
                mm = float(input(f"Pixeldistanz={pixel_dist:.1f}px. Echte Länge in mm? "))
                print(f"  → {pixel_dist/mm:.3f} Pixel/mm")
                cv2.destroyAllWindows()

    cv2.imshow("Kalibrierung", frame)
    cv2.setMouseCallback("Kalibrierung", click)
    cv2.waitKey(0)
    if len(points) == 2:
        pixel_dist = np.hypot(points[1][0]-points[0][0], points[1][1]-points[0][1])
        return pixel_dist / REAL_MM_BLAU
    return None

# ============================================================
# SCHRITT 2: Auto in jedem Frame tracken
# ============================================================
def track_car(video_path, px_per_mm, x_min, x_max, threshold):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times  = []
    y_px   = []
    frames_with_car = []

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # Nur die Fahrspur ausschneiden
        track_region = frame[:, x_min:x_max]

        # Graustufen + Helligkeitsfilter
        gray   = cv2.cvtColor(track_region, cv2.COLOR_BGR2GRAY)
        bright = gray > threshold

        n_bright = np.count_nonzero(bright)
        if n_bright < 5:   # zu wenig helle Pixel → kein Auto erkannt
            continue

        # Schwerpunkt der hellen Pixel = Auto-Position
        ys, xs = np.where(bright)
        cy = float(np.median(ys))   # Median robuster als Mean bei Rauschen

        times.append(frame_idx / fps)
        y_px.append(cy)
        frames_with_car.append(frame_idx)

    cap.release()

    # Pixel → Meter (Y=0 oben, Y_max unten → umkehren damit pos. Richtung = vorwärts)
    y_px  = np.array(y_px)
    times = np.array(times)

    y_max  = y_px.max()
    y_m    = (y_max - y_px) / (px_per_mm * 1000)   # in Meter, 0 = Startposition

    return times, y_m, frames_with_car

# ============================================================
# SCHRITT 3: Geschwindigkeit & Beschleunigung berechnen
# ============================================================
def compute_kinematics(times, positions, sg_window, sg_order):
    if len(times) < sg_window + 2:
        print(f"Zu wenige Datenpunkte ({len(times)}) für Glättungsfenster {sg_window}")
        sg_window_use = max(5, len(times) // 3 | 1)
    else:
        sg_window_use = sg_window

    # Gleichmäßige Zeitachse interpolieren (nötig für saubere Ableitung)
    t_uniform = np.linspace(times[0], times[-1], len(times))
    pos_interp = np.interp(t_uniform, times, positions)

    # Savitzky-Golay: glättet UND leitet direkt ab
    dt = t_uniform[1] - t_uniform[0]
    pos_smooth = savgol_filter(pos_interp, sg_window_use, sg_order, deriv=0, delta=dt)
    vel_smooth = savgol_filter(pos_interp, sg_window_use, sg_order, deriv=1, delta=dt)
    acc_smooth = savgol_filter(pos_interp, sg_window_use, sg_order, deriv=2, delta=dt)

    return t_uniform, pos_smooth, vel_smooth, acc_smooth

# ============================================================
# SCHRITT 4: Ergebnisse ausgeben und plotten
# ============================================================
def report(times, pos, vel, acc):
    v_max  = vel.max()
    v_min  = vel.min()
    a_max  = acc.max()
    a_min  = acc.min()

    print("\n" + "="*50)
    print("ERGEBNISSE")
    print("="*50)
    print(f"Max. Geschwindigkeit (vorwärts): {v_max:.3f} m/s  ({v_max*3.6:.2f} km/h)")
    print(f"Max. Geschwindigkeit (rückw./nach Bremsen): {v_min:.3f} m/s")
    print(f"Max. Beschleunigung:  {a_max:.2f} m/s²")
    print(f"Max. Bremsverzögerung: {a_min:.2f} m/s²  (negativ = Bremsen)")
    print(f"Strecke gesamt:       {pos[-1]:.3f} m")
    print(f"Dauer:                {times[-1]-times[0]:.2f} s")
    print("="*50)

def plot(times, pos, vel, acc):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Carrera Hybrid RC – Kinematik aus Video", fontsize=14)

    axes[0].plot(times, pos, color="steelblue")
    axes[0].set_ylabel("Position (m)")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(times, vel, color="darkorange")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("Geschwindigkeit (m/s)")
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(times, acc, color="crimson")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_ylabel("Beschleunigung (m/s²)")
    axes[2].set_xlabel("Zeit (s)")
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("kinematik.png", dpi=150)
    plt.show()
    print("Plot gespeichert: kinematik.png")

# ============================================================
# HAUPTPROGRAMM
# ============================================================
if __name__ == "__main__":
    px_per_mm = PIXELS_PER_MM

    if CALIBRATION_MODE:
        result = calibrate(VIDEO_PATH)
        if result:
            px_per_mm = result
            print(f"Kalibrierung: {px_per_mm:.3f} Pixel/mm → bitte PIXELS_PER_MM im Script anpassen")
    else:
        print(f"Nutze gespeicherten Wert: {px_per_mm:.3f} Pixel/mm")

    print("Tracke Auto...")
    times, positions, _ = track_car(
        VIDEO_PATH, px_per_mm, TRACK_X_MIN, TRACK_X_MAX, BRIGHTNESS_THRESHOLD
    )
    print(f"  {len(times)} Frames mit Auto erkannt")

    if len(times) < 10:
        print("WARNUNG: Zu wenige Frames erkannt. BRIGHTNESS_THRESHOLD oder TRACK_X_MIN/MAX anpassen.")
    else:
        times, pos, vel, acc = compute_kinematics(times, positions, SG_WINDOW, SG_ORDER)
        report(times, pos, vel, acc)
        plot(times, pos, vel, acc)
