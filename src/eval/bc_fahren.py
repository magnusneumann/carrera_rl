"""Faehrt das vortrainierte Netz (Behaviour Cloning) selbst?

Warum das wichtig ist
---------------------
Aus dem Vortraining wird nur der Faltungsteil in die SAC-Policy uebertragen.
Die Linear-Schichten - also die eigentliche Zuordnung "Bild -> was tue ich" -
werden verworfen. SAC muss diese Zuordnung durch Erkundung neu finden.

Das Netz erreicht auf dem Datensatz MSE 0.0037, kann die Expertenaktionen also
sehr genau vorhersagen. Ob es damit auch faehrt, ist eine andere Frage: der
Datensatz enthaelt nur Zustaende, in die der Experte selbst geraet. Weicht der
Agent davon ab, sieht er Bilder, zu denen er nie eine Vorlage bekommen hat, und
die Fehler schaukeln sich auf (Verteilungsdrift, engl. covariate shift).

Der Ausgang trennt zwei Erklaerungen:

    faehrt gut    ->  Aus den Pixeln laesst sich fahren. Das Problem ist, dass
                      SAC diese Faehigkeit nicht nutzt, sondern verwirft.
    faehrt nicht  ->  Die Vorlage ist auf der eigenen Trajektorie wertlos,
                      Wahrnehmung und Verteilungsdrift bleiben das Problem.

Aufruf:
    python -m src.eval.bc_fahren
    python -m src.eval.bc_fahren --episoden 5
"""
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn

from src.envs.carrera_2d_env import Carrera2DEnv

RUNDE_M = 7.66


class VortrainiertesNetz(nn.Module):
    """Exakt die Architektur aus der Vortrainings-Zelle."""

    def __init__(self, action_dim, bild_hw):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, 3, *bild_hw)).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, action_dim))

    def forward(self, x):
        return self.linear(self.cnn(x))


def fahre(netz, env, episoden):
    zeilen = []
    for _ in range(episoden):
        obs, _ = env.reset()
        stapel = np.zeros((3,) + obs.shape[1:], dtype=np.uint8)
        stapel[-1] = obs[0]
        r_sum, n, sf, runden, weg = 0.0, 0, False, 0, 0.0
        tempo = []
        while True:
            with torch.no_grad():
                x = torch.tensor(stapel[None], dtype=torch.float32) / 255.0
                a = netz(x).numpy()[0]
            a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            stapel[:-1] = stapel[1:]
            stapel[-1] = obs[0]
            r_sum += float(r)
            n += 1
            v = float(env.state[2])
            tempo.append(v)
            weg += abs(v) * env.dt
            if info.get('sf_crossed') and not sf:
                runden += 1
            sf = bool(info.get('sf_crossed'))
            if term or trunc:
                grund = 'CRASH' if info.get('is_crashing') else 'Zeitlimit'
                break
        zeilen.append((r_sum, n, max(0, runden - 1), np.mean(tempo), weg, grund))
    return zeilen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episoden', type=int, default=3)
    p.add_argument('--backbone', default=None)
    args = p.parse_args()

    pfad = args.backbone or sorted(glob.glob('models/pretrained_vision_*.pth'))[-1]
    gew = torch.load(pfad, map_location='cpu')
    # Aus der Groesse der Linear-Schicht die Bildgroesse zurueckrechnen, damit
    # das Netz zu der Umgebung passt, auf der es trainiert wurde.
    n_flat = gew['linear.0.weight'].shape[1]
    for gs in ((166, 100), (250, 150), (300, 180), (200, 120)):
        probe = VortrainiertesNetz(gew['linear.2.weight'].shape[0], (gs[1], gs[0]))
        if probe.linear[0].in_features == n_flat:
            break
    else:
        raise ValueError(f"Keine bekannte Aufloesung passt zu {n_flat} Merkmalen")

    netz = probe
    netz.load_state_dict(gew)
    netz.eval()

    env = Carrera2DEnv('data/strecke.png', 'data/carrera_car.png', obs_type="vision",
                       render_mode="hidden", camera_view="global", global_size=gs)

    print(f"Backbone : {pfad}")
    print(f"Aufloesung: {gs[0]}x{gs[1]}  ({n_flat} Merkmale)")
    print(f"Physik   : {env.get_env_config()['longitudinal']['model']}")
    print(f"Runde    : {RUNDE_M} m, Zeitlimit 2000 Frames\n")

    zeilen = fahre(netz, env, args.episoden)
    env.close()

    print(f"{'Ep':>3}{'Reward':>10}{'Frames':>8}{'Runden':>8}{'Tempo':>8}"
          f"{'Strecke':>9}  Ende")
    for i, (r, n, ru, v, w, g) in enumerate(zeilen, 1):
        print(f"{i:>3}{r:>10.1f}{n:>8}{ru:>8}{v:>8.2f}{w:>9.2f}  {g}")

    a = np.array([[z[0], z[1], z[2], z[3], z[4]] for z in zeilen], dtype=float)
    print(f"\nMittel: Reward {a[:,0].mean():.1f}  Frames {a[:,1].mean():.0f}  "
          f"Runden {a[:,2].mean():.1f}  Tempo {a[:,3].mean():.2f} m/s  "
          f"Strecke {a[:,4].mean():.2f} m")
    print("\nZum Vergleich (gleiche Physik, gleicher Reward):")
    print(f"   {'Lidar-Experte (PPO)':<32} 2000 Frames, 11 Runden, 1.22 m/s")
    print(f"   {'Lidar SAC (Kontrolle)':<32} 2000 Frames, 11 Runden, 1.19 m/s")
    print(f"   {'Vision SAC (laufend, 850k)':<32}  193 Frames,  1 Runde,  0.79 m/s")


if __name__ == '__main__':
    main()
