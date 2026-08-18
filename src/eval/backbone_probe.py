"""Stecken in den uebertragenen Faltungsschichten brauchbare Informationen?

Warum diese Messung
-------------------
Vom Vortraining wandern nur `cnn.0`, `cnn.2` und `cnn.4` in die SAC-Policy -
die Augen. Die Linear-Schichten, in denen das Fahren steckt, werden verworfen.

Ob das Netz selbst faehrt, sagt deshalb NICHTS ueber den Nutzen der
Uebertragung aus. Ein Netz kann gute Merkmale gelernt haben und trotzdem
scheitern, weil sein Kopf mit ungewohnten Lagen nicht zurechtkommt.

Die richtige Frage lautet: liegen in den Merkmalen mehr verwertbare
Informationen als in zufaelligen Anfangsgewichten? Genau das misst ein
lineares Abtasten (linear probe): Faltungsschichten einfrieren, darauf einen
linearen Kopf anpassen, der bekannte Groessen vorhersagt, und schauen, wie gut
das gelingt.

Vorhergesagt werden Groessen, die der Agent zum Fahren braucht:

    v            Geschwindigkeit
    dist_l/m/r   Abstand zu den Streckenraendern (die Lidar-Beobachtung)
    sin/cos      Fahrtrichtung

Gemessen wird das Bestimmtheitsmass R2 auf zurueckgehaltenen Daten. R2 = 1
heisst perfekt vorhersagbar, R2 = 0 heisst nicht besser als der Mittelwert.

Aufruf:
    python -m src.eval.backbone_probe
    python -m src.eval.backbone_probe --proben 4000
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

from src.envs.carrera_2d_env import Carrera2DEnv


def faltungsteil():
    """Exakt die drei Schichten, die uebertragen werden."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten(),
    )


def sammle(n, global_size=(250, 150), n_stack=3, stride=1, seed=0):
    """Bildstapel und die zugehoerigen wahren Groessen aufzeichnen.

    Gefahren wird zufaellig, damit die Daten nicht nur die Ideallinie zeigen -
    ein Merkmalsextraktor soll auch abseits davon etwas taugen.
    """
    from stable_baselines3 import PPO
    experten = sorted(glob.glob('models/01_Lidar_Baseline_*.zip'))
    experte = PPO.load(experten[-1], device='cpu') if experten else None

    rng = np.random.default_rng(seed)
    env = Carrera2DEnv('data/strecke.png', 'data/carrera_car.png', obs_type="vision",
                       render_mode="hidden", camera_view="global", global_size=global_size)
    lidar = Carrera2DEnv('data/strecke.png', 'data/carrera_car.png', obs_type="lidar",
                         render_mode="hidden")

    tiefe = (n_stack - 1) * stride + 1
    plaetze = [tiefe - 1 - z * stride for z in range(n_stack - 1, -1, -1)]

    bilder, ziele = [], []
    obs, _ = env.reset()
    verlauf = np.zeros((tiefe,) + obs.shape[1:], dtype=np.uint8)
    verlauf[-1] = obs[0]

    while len(bilder) < n:
        x, y, v, theta, _ = env.state
        sa, _ = lidar.sensor_suite.get_lidar_observation(
            x * env.pixels_per_meter, y * env.pixels_per_meter, theta, v)
        bilder.append(verlauf[plaetze].copy())
        ziele.append([v, sa[1], sa[2], sa[3], np.sin(theta), np.cos(theta)])

        # Mischung aus Experte und Zufall, damit auch schiefe Lagen vorkommen
        if experte is not None and rng.random() < 0.7:
            lidar_obs = np.array([v / env.max_speed, sa[1], sa[2], sa[3]], dtype=np.float32)
            a, _ = experte.predict(lidar_obs, deterministic=False)
        else:
            a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        verlauf[:-1] = verlauf[1:]
        verlauf[-1] = obs[0]
        if term or trunc:
            obs, _ = env.reset()
            verlauf[:] = 0
            verlauf[-1] = obs[0]

    env.close(); lidar.close()
    return np.array(bilder), np.array(ziele, dtype=np.float64)


def merkmale(cnn, bilder, device, batch=64):
    aus = []
    with torch.no_grad():
        for i in range(0, len(bilder), batch):
            x = torch.tensor(bilder[i:i + batch], dtype=torch.float32, device=device) / 255.0
            aus.append(cnn(x).cpu().numpy())
    return np.concatenate(aus).astype(np.float64)


def probe(X, Y, anteil_test=0.25, anteil_val=0.2):
    """Lineares Abtasten mit Ridge, R2 auf zurueckgehaltenen Daten.

    Drei Fallstricke, die eine erste Fassung dieses Tests unbrauchbar machten:

    1. Es gibt weit mehr Merkmale (25920) als Proben. Ohne kraeftige
       Regularisierung passt sich der lineare Kopf perfekt an das Training an
       und liefert auf Testdaten sinnlose Werte (gemessen R2 = -2303).
    2. Viele Filter sind stellenweise tot, ihre Ausgabe ist konstant null. Beim
       Normieren wird durch ihre Streuung geteilt - das sprengt die Rechnung.
       Solche Merkmale werden verworfen.
    3. Der Regularisierungsgrad darf nicht geraten werden. Er wird auf einer
       eigenen Teilmenge gewaehlt, nicht auf den Testdaten.

    Geloest wird in dualer Form, weil nur eine n x n Matrix noetig ist.
    """
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    n_te = int(anteil_test * n)
    n_va = int(anteil_val * n)
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]

    sd_alle = X[tr].std(0)
    behalten = sd_alle > 1e-6
    X = X[:, behalten]
    mu, sd = X[tr].mean(0), X[tr].std(0)

    def norm(a):
        return (a - mu) / sd

    Xtr, Xva, Xte = norm(X[tr]), norm(X[va]), norm(X[te])
    ytr, yva, yte = Y[tr], Y[va], Y[te]
    ym = ytr.mean(0)
    K = Xtr @ Xtr.T
    Kva, Kte = Xva @ Xtr.T, Xte @ Xtr.T
    I = np.eye(len(K))

    bestes, bestes_lam = None, None
    for lam in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8):
        alpha = np.linalg.solve(K + lam * I, ytr - ym)
        pv = Kva @ alpha + ym
        fehler = ((yva - pv) ** 2).mean()
        if bestes is None or fehler < bestes:
            bestes, bestes_lam = fehler, lam

    alpha = np.linalg.solve(K + bestes_lam * I, ytr - ym)
    pred = Kte @ alpha + ym
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot, int(behalten.sum()), bestes_lam


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proben', type=int, default=2500)
    p.add_argument('--backbone', default=None)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pfad = args.backbone or sorted(glob.glob('models/pretrained_vision_*.pth'))[-1]

    print(f"Sammle {args.proben} Proben (70 % Experte, 30 % zufaellig) ...")
    bilder, ziele = sammle(args.proben)
    print(f"   Bilder {bilder.shape}, Ziele {ziele.shape}\n")

    namen = ['v', 'dist_l', 'dist_m', 'dist_r', 'sin(theta)', 'cos(theta)']
    ergebnisse = {}

    for bez, laden in (('zufaellig', False), ('vortrainiert', True)):
        torch.manual_seed(0)
        cnn = faltungsteil().to(device).eval()
        if laden:
            g = torch.load(pfad, map_location=device)
            with torch.no_grad():
                for i in (0, 2, 4):
                    cnn[i].weight.copy_(g[f'cnn.{i}.weight'])
                    cnn[i].bias.copy_(g[f'cnn.{i}.bias'])
        X = merkmale(cnn, bilder, device)
        ergebnisse[bez], n_merk, lam = probe(X, ziele)
        print(f'   {bez:<14} {n_merk:>6} nutzbare Merkmale, Regularisierung {lam:g}')
        del cnn, X
        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f"Backbone: {os.path.basename(pfad)}")
    print(f"Bestimmtheitsmass R2 auf zurueckgehaltenen Daten "
          f"(1 = perfekt, 0 = nutzlos)\n")
    print(f"{'Groesse':<12}{'zufaellig':>12}{'vortrainiert':>14}{'Gewinn':>10}")
    print('-' * 48)
    for i, n in enumerate(namen):
        z, v = ergebnisse['zufaellig'][i], ergebnisse['vortrainiert'][i]
        print(f"{n:<12}{z:>12.3f}{v:>14.3f}{v-z:>+10.3f}")
    z, v = ergebnisse['zufaellig'].mean(), ergebnisse['vortrainiert'].mean()
    print('-' * 48)
    print(f"{'Mittel':<12}{z:>12.3f}{v:>14.3f}{v-z:>+10.3f}")
    print()
    if v - z > 0.05:
        print("Das Vortraining bringt messbar mehr verwertbare Information in die")
        print("Faltungsschichten als zufaellige Anfangsgewichte.")
    else:
        print("Kein nennenswerter Vorsprung gegenueber zufaelligen Anfangsgewichten.")
        print("Der Vortrainingsschritt waere dann verzichtbar.")


if __name__ == '__main__':
    main()
