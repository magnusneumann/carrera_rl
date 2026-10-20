"""Alle Bild-Agenten unter identischen Bedingungen: Rundenzeit, Runden, Episodenlaenge.

Gemessen werden Bestwerte ueber mehrere Episoden, nicht Mittelwerte - gefragt war,
was ein Agent im besten Fall schafft.

    beste Rundenzeit   kuerzeste VOLLE Runde (die erste Ueberquerung zaehlt nicht,
                       der Wagen startet vor der Linie)
    max. Runden        meiste vollstaendige Runden in EINER Episode, ohne Crash
    ep_len             Frames bis zum Episodenende

Aufruf aus dem Repo-Wurzelverzeichnis:
    python -m src.eval.bildagenten [--episoden 10] [--streuend]
"""
import argparse
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.envs.carrera_2d_env import Carrera2DEnv

TRACK, CAR = 'data/strecke.png', 'data/carrera_car.png'
# Auf der CPU rechnet PyTorch minimal anders als auf der GPU; bei diesen Policies
# reicht das fuer voellig andere Ergebnisse. Siehe rl_erkenntnisse.md Abschnitt 7.
GERAET = 'cuda' if torch.cuda.is_available() else 'cpu'


def kandidaten():
    """Alle Bild-Agenten mit heutiger Physik, aufsteigend nach Datum."""
    aus = []
    for d in sorted(glob.glob('models/*/')):
        cfg = os.path.join(d, 'train_config.json')
        mdl = os.path.join(d, 'best_model.zip')
        if not (os.path.exists(cfg) and os.path.exists(mdl)):
            continue
        c = json.load(io.open(cfg, encoding='utf-8'))
        if c.get('obs_type') != 'vision':
            continue
        if c.get('env', {}).get('longitudinal', {}).get('model') != 'measured_v2':
            continue          # andere Physik, Policy nicht uebertragbar
        hw = c.get('env', {}).get('vision', {}).get('target_hw')
        gs = (int(hw[1]), int(hw[0])) if hw else (166, 100)   # (Breite, Hoehe)
        aus.append(dict(name=os.path.basename(os.path.normpath(d)), pfad=mdl,
                        algo=c.get('algorithm', 'SAC'),
                        n_stack=int(c.get('n_stack', 1) or 1),
                        stride=int(c.get('stride', 1) or 1), gs=gs))
    return aus


def fahre(k, episoden, streuend):
    def f():
        return Carrera2DEnv(TRACK, CAR, obs_type='vision', render_mode='hidden',
                            camera_view='global', global_size=k['gs'],
                            start_streuung=0.9 if streuend else 0.0,
                            start_seite=0.08 if streuend else 0.0)
    env = DummyVecEnv([f])
    if k['n_stack'] > 1:
        env = VecFrameStack(env, n_stack=k['n_stack'], channels_order='first')
    kls = PPO if k['algo'] == 'PPO' else SAC
    m = kls.load(k['pfad'], device=GERAET)

    rundenzeiten, runden_je_ep, laengen, lohn = [], [], [], []
    for _ in range(episoden):
        obs = env.reset()
        seit_linie, zuvor, n_runden, schritte, r = 0, False, 0, 0, 0.0
        erste_gesehen = False
        while True:
            a, _ = m.predict(obs, deterministic=True)
            obs, rew, done, infos = env.step(a)
            schritte += 1
            seit_linie += 1
            r += float(rew[0])
            jetzt = bool(infos[0].get('sf_crossed'))
            if jetzt and not zuvor:
                if erste_gesehen:
                    # eine vollstaendige Runde ist abgeschlossen
                    rundenzeiten.append(seit_linie)
                    n_runden += 1
                else:
                    erste_gesehen = True   # Teilrunde vom Start bis zur Linie
                seit_linie = 0
            zuvor = jetzt
            if done[0]:
                break
        runden_je_ep.append(n_runden)
        laengen.append(schritte)
        lohn.append(r)
    env.close()
    return dict(beste_runde=min(rundenzeiten) if rundenzeiten else None,
                max_runden=max(runden_je_ep), mittel_runden=float(np.mean(runden_je_ep)),
                max_len=max(laengen), mittel_len=float(np.mean(laengen)),
                bester_lohn=max(lohn), mittel_lohn=float(np.mean(lohn)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episoden', type=int, default=10)
    p.add_argument('--streuend', action='store_true',
                   help='mit start_streuung=0.9 und start_seite=0.08 statt festem Start')
    args = p.parse_args()

    ref = Carrera2DEnv(TRACK, CAR, obs_type='lidar', render_mode='hidden')
    dt = ref.dt
    ref.close()

    art = 'gestreute Starts' if args.streuend else 'fester Start'
    print(f'Bild-Agenten, {args.episoden} Episoden je Modell, {art}, Geraet {GERAET}')
    print(f'Zeitlimit 2000 Frames, dt = 1/{round(1/dt)} s\n')
    kopf = (f"{'Modell':<42}{'Aufl.':<10}{'beste Runde':>14}"
            f"{'max Runden':>12}{'Ø Runden':>10}{'max ep_len':>12}{'Ø ep_len':>10}"
            f"{'bester Lohn':>13}")
    print(kopf)
    print('-' * len(kopf))

    zeilen = []
    for k in kandidaten():
        try:
            e = fahre(k, args.episoden, args.streuend)
        except Exception as fehler:
            print(f"{k['name']:<42}uebersprungen: {type(fehler).__name__}: {fehler}")
            continue
        br = (f"{e['beste_runde']} F / {e['beste_runde']*dt:.2f}s"
              if e['beste_runde'] else '- keine -')
        print(f"{k['name']:<42}{k['gs'][0]}x{k['gs'][1]:<6}{br:>14}"
              f"{e['max_runden']:>12}{e['mittel_runden']:>10.1f}"
              f"{e['max_len']:>12}{e['mittel_len']:>10.0f}{e['bester_lohn']:>13.1f}")
        e.update(name=k['name'], aufloesung=f"{k['gs'][0]}x{k['gs'][1]}",
                 n_stack=k['n_stack'], stride=k['stride'])
        zeilen.append(e)

    if zeilen:
        mit = [z for z in zeilen if z['beste_runde']]
        print('-' * len(kopf))
        if mit:
            b = min(mit, key=lambda z: z['beste_runde'])
            print(f"schnellste Runde : {b['name']}  "
                  f"{b['beste_runde']} Frames / {b['beste_runde']*dt:.2f} s")
        w = max(zeilen, key=lambda z: z['max_runden'])
        print(f"meiste Runden    : {w['name']}  {w['max_runden']} in einer Episode")
        ziel = f"models/bildagenten_{'streuend' if args.streuend else 'fest'}.json"
        with io.open(ziel, 'w', encoding='utf-8') as f:
            json.dump({'episoden': args.episoden, 'streuend': args.streuend,
                       'geraet': GERAET, 'ergebnisse': zeilen}, f,
                      indent=2, ensure_ascii=False)
        print(f'Gespeichert: {ziel}')


if __name__ == '__main__':
    main()
