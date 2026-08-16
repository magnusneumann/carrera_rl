"""Fahrlinien von Experte und Kamera-Agent uebereinanderlegen.

Zeigt auf einen Blick, wo ein Agent von der Ideallinie abweicht und wo er
crasht. Aussagekraeftiger als jede Kennzahl, wenn die Frage lautet WARUM ein
Lauf scheitert - siehe rl_erkenntnisse.md, Abschnitt 13d.

Aufruf:
    python -m src.eval.fahrlinien
"""
import os
import numpy as np, cv2
from stable_baselines3 import SAC, PPO
from src.envs.carrera_2d_env import Carrera2DEnv


def spur_vision():
    m = SAC.load('models/09_Vision_SAC_Finetuned_20260815_2230/best_model.zip',
                 custom_objects={'buffer_size': 100}, device='cpu')
    e = Carrera2DEnv('data/strecke.png', 'data/carrera_car.png', obs_type='vision',
                     render_mode='hidden', camera_view='global', global_size=(250, 150))
    o, _ = e.reset(); st = np.zeros((3,) + o.shape[1:], np.uint8); st[-1] = o[0]
    pts, v = [], []
    while True:
        a, _ = m.predict(st[None], deterministic=True)
        o, r, t, tr, inf = e.step(a[0].astype(np.float32))
        st[:-1] = st[1:]; st[-1] = o[0]
        pts.append(e.state[:2] * e.pixels_per_meter); v.append(float(e.state[2]))
        if t or tr:
            break
    hg = e.screen.copy(); e.close()
    return np.array(pts), np.array(v), hg


def spur_experte():
    m = PPO.load('models/01_Lidar_Baseline_20260728_1800.zip', device='cpu')
    e = Carrera2DEnv('data/strecke.png', 'data/carrera_car.png', obs_type='lidar',
                     render_mode='hidden')
    o, _ = e.reset(); pts, v = [], []
    for _ in range(2000):
        a, _ = m.predict(o, deterministic=True)
        o, r, t, tr, inf = e.step(a)
        pts.append(e.state[:2] * e.pixels_per_meter); v.append(float(e.state[2]))
        if t or tr:
            break
    e.close()
    return np.array(pts), np.array(v)


pv, vv, hg = spur_vision()
pe, ve = spur_experte()

b = hg.copy()
b = (b * 0.45).astype(np.uint8)          # Hintergrund abdunkeln


def male(bild, pts, farbe, dicke=2):
    p = pts.astype(np.int32)
    for i in range(1, len(p)):
        cv2.line(bild, tuple(p[i-1]), tuple(p[i]), farbe, dicke, cv2.LINE_AA)


# Experte nur die erste Runde, sonst ueberdeckt er alles
ende_runde = min(len(pe), 240)
male(b, pe[:ende_runde], (90, 220, 90), 2)
male(b, pv, (60, 160, 255), 2)

# Crashpunkt
c = pv[-1].astype(int)
cv2.circle(b, tuple(c), 14, (60, 60, 255), 2)
cv2.line(b, (c[0]-20, c[1]-20), (c[0]+20, c[1]+20), (60, 60, 255), 2)
cv2.line(b, (c[0]-20, c[1]+20), (c[0]+20, c[1]-20), (60, 60, 255), 2)
cv2.putText(b, f"CRASH  {vv[-1]:.2f} m/s", (c[0]+26, c[1]-6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2, cv2.LINE_AA)

cv2.putText(b, "Experte (Lidar)", (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (90, 220, 90), 2, cv2.LINE_AA)
cv2.putText(b, "Kamera-Agent 250x150", (14, 54), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (60, 160, 255), 2, cv2.LINE_AA)

p = os.path.join('models', 'fahrlinien.png')
cv2.imwrite(p, b)
print('geschrieben:', p, b.shape)
print(f'Kamera: {len(pv)} Frames, Crash bei {c}')
print(f'Experte: {len(pe)} Frames gesamt, gezeichnet {ende_runde}')
