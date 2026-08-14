"""Alle gespeicherten Modelle unter der HEUTIGEN Umgebung und dem heutigen
Reward auswerten.

Warum das noetig ist
--------------------
Die Reward-Zahlen aus `evaluations.npz` stammen jeweils aus dem Reward, der zur
Trainingszeit galt. `2210` lief mit `w_integral = 0.08`, der Kontrollversuch mit
0.0 — die Zahlen sind also nicht vergleichbar, obwohl sie gleich aussehen.

Dieses Skript laesst jedes Modell in derselben aktuellen Umgebung fahren und
misst neben dem Reward auch Groessen, die vom Reward unabhaengig sind: Runden,
Durchschnittstempo, Ueberlebensdauer. Die tragen ueber Reward-Aenderungen hinweg.

Nur Modelle unter `measured_v2` sind sinnvoll auswertbar. Alles aus der Zeit
davor (`force_drag_v1`) fuhr ein zwanzigfach schnelleres Auto; seine Policy in
die heutige Physik zu setzen misst nichts.

Aufruf:
    python -m src.eval.vergleich
    python -m src.eval.vergleich --episoden 20
"""
import argparse
import glob
import io
import json
import os

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.envs.carrera_2d_env import Carrera2DEnv

TRACK, CAR = 'data/strecke.png', 'data/carrera_car.png'
RUNDE_M = 7.66          # Mittellinie, siehe rl_erkenntnisse.md Referenzwerte


def baue_env(obs_type, n_stack):
    def f():
        return Carrera2DEnv(TRACK, CAR, obs_type=obs_type, render_mode="hidden",
                            camera_view="global" if obs_type == "vision" else "crop")
    env = DummyVecEnv([f])
    if n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack, channels_order="first")
    return env


def fahre(modell, env, episoden, dt, ppm):
    """Deterministisch fahren und rewardunabhaengige Groessen mitzaehlen."""
    lohn, laenge, runden, strecke = [], [], [], []
    for _ in range(episoden):
        obs = env.reset()
        r_sum, schritte, sf_zuvor, n_runden = 0.0, 0, False, 0
        weg = 0.0
        while True:
            a, _ = modell.predict(obs, deterministic=True)
            obs, r, done, infos = env.step(a)
            r_sum += float(r[0])
            schritte += 1
            i = infos[0]
            if i.get('sf_crossed') and not sf_zuvor:
                n_runden += 1
            sf_zuvor = bool(i.get('sf_crossed'))
            # Zurueckgelegte Strecke aus dem Zustandsvektor der Umgebung, nicht
            # aus dem Bild - so ist der Wert unabhaengig von der Beobachtungsart.
            # state = [x, y, v, theta, omega], siehe carrera_2d_env.py:336
            roh = env.venv.envs[0].unwrapped if hasattr(env, 'venv') else env.envs[0].unwrapped
            weg += abs(float(roh.state[2])) * dt
            if done[0]:
                break
        lohn.append(r_sum)
        laenge.append(schritte)
        runden.append(max(0, n_runden - 1))   # erste Ueberquerung ist der Start
        strecke.append(weg)
    laenge = np.array(laenge, float)
    strecke = np.array(strecke, float)
    return {
        'reward': float(np.mean(lohn)),
        'reward_std': float(np.std(lohn)),
        'ep_len': float(np.mean(laenge)),
        'runden': float(np.mean(runden)),
        'tempo': float(np.mean(strecke / (laenge * dt))),
        'strecke_m': float(np.mean(strecke)),
    }


def kandidaten():
    """(Anzeigename, Pfad, Algo, obs_type, n_stack) fuer alles Auswertbare."""
    aus = []
    for d in sorted(glob.glob('models/*/')):
        pfad = os.path.join(d, 'best_model.zip')
        cfg = os.path.join(d, 'train_config.json')
        if not (os.path.exists(pfad) and os.path.exists(cfg)):
            continue
        c = json.load(io.open(cfg, encoding='utf-8'))
        phys = c.get('env', {}).get('longitudinal', {}).get('model')
        if phys != 'measured_v2':
            continue                      # andere Physik, Policy nicht uebertragbar
        name = os.path.basename(os.path.normpath(d))
        aus.append((name, pfad, c.get('algorithm', 'SAC'),
                    c.get('obs_type', 'vision'), int(c.get('n_stack', 1) or 1)))
    # Der Lidar-Experte liegt als einzelne Datei daneben
    for z in sorted(glob.glob('models/01_Lidar_Baseline_*.zip')):
        cfg = z.replace('.zip', '_config.json')
        if os.path.exists(cfg):
            aus.append((os.path.basename(z)[:-4], z, 'PPO', 'lidar', 1))
    return aus


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episoden', type=int, default=10)
    args = p.parse_args()

    ref = Carrera2DEnv(TRACK, CAR, obs_type="lidar", render_mode="hidden")
    dt, ppm = ref.dt, ref.pixels_per_meter
    reward_jetzt = ref.reward_calculator.to_dict()
    ref.close()

    print(f"Auswertung unter der heutigen Umgebung, {args.episoden} Episoden je Modell")
    print(f"Reward-Version {reward_jetzt.get('version')}, "
          f"w_integral={reward_jetzt.get('w_integral')}, "
          f"v_slow={reward_jetzt.get('v_slow')}")
    print(f"Rundenlaenge {RUNDE_M} m, Zeitlimit 2000 Frames\n")
    print(f"{'Modell':<40}{'Beob.':<8}{'Reward':>10}{'±':>8}"
          f"{'ep_len':>8}{'Runden':>8}{'Tempo':>8}")
    print("-" * 90)

    zeilen = []
    for name, pfad, algo, obs_type, n_stack in kandidaten():
        try:
            env = baue_env(obs_type, n_stack)
            kls = PPO if algo == 'PPO' else SAC
            m = kls.load(pfad, device='cpu')
            e = fahre(m, env, args.episoden, dt, ppm)
            env.close()
        except Exception as fehler:
            print(f"{name:<40}{obs_type:<8}  uebersprungen: {type(fehler).__name__}: {fehler}")
            continue
        e['name'], e['obs'] = name, obs_type
        zeilen.append(e)
        print(f"{name:<40}{obs_type:<8}{e['reward']:>10.1f}{e['reward_std']:>8.1f}"
              f"{e['ep_len']:>8.0f}{e['runden']:>8.1f}{e['tempo']:>8.2f}")

    if zeilen:
        print("-" * 90)
        best = max(zeilen, key=lambda z: z['reward'])
        print(f"Bester: {best['name']}  ({best['reward']:.1f}, "
              f"{best['runden']:.1f} Runden, {best['tempo']:.2f} m/s)")
        with io.open('models/vergleich.json', 'w', encoding='utf-8') as f:
            json.dump({'episoden': args.episoden, 'reward': reward_jetzt,
                       'ergebnisse': zeilen}, f, indent=2, ensure_ascii=False)
        print("Gespeichert: models/vergleich.json")


if __name__ == '__main__':
    main()
