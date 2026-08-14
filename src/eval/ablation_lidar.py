"""Wie sehr haengt der Lidar-Agent an der Geschwindigkeit?

Die Lidar-Beobachtung ist [v/v_max, dist_links, dist_mitte, dist_rechts].
Der Agent bekommt also zwei Dinge geschenkt, die der Kamera-Agent sich aus
Pixeln erarbeiten muss: die Geschwindigkeit UND die Streckengeometrie.

Dieses Skript blendet gezielt Teile aus und misst, was passiert. Damit laesst
sich trennen, welche der beiden Abkuerzungen traegt.
"""
import glob
import numpy as np
from stable_baselines3 import SAC

from src.envs.carrera_2d_env import Carrera2DEnv

RUNDE_M = 7.66


def fahre(modell, aendern, episoden=3, rng=None):
    env = Carrera2DEnv('data/strecke.png', 'data/carrera_car.png',
                       obs_type="lidar", render_mode="hidden")
    lohn, laengen, runden, wege = [], [], [], []
    for _ in range(episoden):
        obs, _ = env.reset()
        r_sum, n, sf_prev, laps, weg = 0.0, 0, False, 0, 0.0
        while True:
            a, _ = modell.predict(aendern(obs.copy(), rng), deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            r_sum += float(r); n += 1
            weg += abs(float(env.state[2])) * env.dt
            if info.get('sf_crossed') and not sf_prev:
                laps += 1
            sf_prev = bool(info.get('sf_crossed'))
            if term or trunc:
                break
        lohn.append(r_sum); laengen.append(n); runden.append(max(0, laps - 1)); wege.append(weg)
    env.close()
    return (np.mean(lohn), np.mean(laengen), np.mean(runden),
            np.mean(np.array(wege) / (np.array(laengen) * (1 / 30))))


d = sorted(glob.glob('models/12_Lidar_SAC_Kontrolle_*/'))[-1]
m = SAC.load(d + 'best_model.zip', device='cpu')
rng = np.random.default_rng(0)

varianten = [
    ("unveraendert",                       lambda o, r: o),
    ("v auf 0 gesetzt",                    lambda o, r: np.array([0.0, o[1], o[2], o[3]], np.float32)),
    ("v eingefroren auf 0.5",              lambda o, r: np.array([0.5, o[1], o[2], o[3]], np.float32)),
    ("v durch Rauschen ersetzt",           lambda o, r: np.array([r.random(), o[1], o[2], o[3]], np.float32)),
    ("v grob gerastert (4 Stufen)",        lambda o, r: np.array([np.floor(o[0] * 4) / 4, o[1], o[2], o[3]], np.float32)),
    ("Abstaende auf 0, v exakt",           lambda o, r: np.array([o[0], 0.0, 0.0, 0.0], np.float32)),
    ("Abstaende verrauscht, v exakt",      lambda o, r: np.array([o[0], *(o[1:] + r.normal(0, .1, 3))], np.float32)),
]

print(f"Modell: {d}")
print("Der Agent sieht die veraenderte Beobachtung; die Physik laeuft normal weiter.\n")
print(f"{'Variante':<34}{'Reward':>10}{'ep_len':>9}{'Runden':>8}{'Tempo':>8}")
print("-" * 69)
basis = None
for name, f in varianten:
    r, l, ru, t = fahre(m, f, rng=rng)
    if basis is None:
        basis = r
    anteil = "" if name == "unveraendert" else f"   {100*(r-basis)/abs(basis):+6.0f} %"
    print(f"{name:<34}{r:>10.1f}{l:>9.0f}{ru:>8.1f}{t:>8.2f}{anteil}")
