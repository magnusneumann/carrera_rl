"""Rauchtest: ein trainiertes Modell einmal fahren lassen und drei Zahlen notieren.

Zweck ist NICHT, die Qualitaet vor und nach der Physikumstellung zu vergleichen –
die alten Modelle haben mit dem alten Auto gelernt und werden zwangslaeufig
schlechter fahren. Zweck ist zu sehen, ob die Umgebung nach dem Eingriff
mechanisch noch funktioniert: bewegt sich das Auto, wird gelenkt, werden Runden
gezaehlt.

Ein Durchlauf genuegt: die Umgebung ist deterministisch (fester Startzustand,
kein Seed-Rauschen) und mit deterministic=True liefert die Policy jedes Mal
dasselbe. Genau deshalb sind auch die 5 Eval-Episoden in den evaluations.npz
fuenfmal identisch.

Aufruf:
    python -m src.eval.rauchtest
    python -m src.eval.rauchtest --model models/<lauf>/best_model.zip
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.envs.carrera_2d_env import Carrera2DEnv

STANDARD = "models/09_Vision_SAC_Finetuned_20260724_1432/best_model.zip"


def lade_modell(pfad):
    """Algorithmus aus der train_config.json neben dem Modell bestimmen."""
    cfg_pfad = Path(pfad).parent / "train_config.json"
    algo = "SAC"
    if cfg_pfad.exists():
        algo = json.loads(cfg_pfad.read_text(encoding="utf-8")).get("algorithm", "SAC")
    klasse = {"SAC": SAC, "PPO": PPO}[algo.upper()]
    return klasse.load(pfad, device="cpu"), algo


def groesse_aus_modell(model, n_stack=3):
    """Aufloesung aus dem Beobachtungsraum des Modells ableiten.

    Zuverlaessiger als eine Voreinstellung: ein bei 250x150 trainiertes Modell
    laeuft sonst gegen eine 166x100-Umgebung und bricht mit einem Formfehler ab.
    Der Raum ist (n_stack, H, W), global_size erwartet (Breite, Hoehe).
    """
    form = model.observation_space.shape
    if len(form) != 3:
        return (166, 100)
    return (int(form[2]), int(form[1]))


def baue_env(camera_view="global", n_stack=3, longitudinal_model="measured_v2",
             global_size=(166, 100)):
    """Derselbe Wrapper-Aufbau wie im Training – sonst passt die
    Beobachtungsform nicht (Modell erwartet 3 gestapelte Bilder)."""
    def make():
        return Carrera2DEnv('data/strecke.png', 'data/carrera_car.png',
                            obs_type="vision", render_mode="hidden",
                            camera_view=camera_view,
                            longitudinal_model=longitudinal_model,
                            global_size=global_size)
    return VecFrameStack(DummyVecEnv([make]), n_stack=n_stack)


def fahren(model, env, max_steps=3000):
    roh = env.envs[0].unwrapped          # fuer Zugriff auf state und Rundenzaehler
    obs = env.reset()

    tempo, gas_werte, brems_werte = [], [], []
    runden = 0

    # Runden ueber den Streckenfortschritt zaehlen, NICHT ueber die Ziellinie.
    # Der Startpunkt (1.91, 0.35) m liegt bei 236 px/m nur rund 1 Pixel vor der
    # Ziellinie (462..468 px) - das Auto ueberfaehrt sie sofort nach dem Start
    # und bei langsamer Fahrt mehrfach. Der Fortschritt entlang der Streckenlinie
    # ist dagegen eindeutig.
    def fortschritt():
        p = roh.state[:2] * roh.pixels_per_meter
        i = int(np.argmin(np.linalg.norm(roh.outer_points - p, axis=1)))
        return i / len(roh.outer_points)

    vorher = fortschritt()
    schritte = 0
    crash = False

    while schritte < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        schritte += 1

        tempo.append(float(roh.state[2]))
        gas_werte.append(float(action[0][0]))
        brems_werte.append(float(action[0][1]))

        jetzt = fortschritt()
        if vorher > 0.8 and jetzt < 0.2:      # vorwaerts ueber den Umlauf
            runden += 1
        elif vorher < 0.2 and jetzt > 0.8:    # rueckwaerts
            runden -= 1
        vorher = jetzt

        if done[0]:
            crash = True
            break

    v = np.array(tempo)
    return {
        "frames": schritte,
        "crash": crash,
        "runden": runden,
        "v_mittel": float(v.mean()),
        "v_max": float(v.max()),
        "v_ende": float(v[-1]),
        "gas_mittel": float(np.mean(gas_werte)),
        "bremse_mittel": float(np.mean(brems_werte)),
        "strecke_m": float(np.abs(v).sum() * roh.dt),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=STANDARD)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--physik", default="measured_v2",
                   choices=["measured_v2", "force_drag_v1"])
    args = p.parse_args()

    model, algo = lade_modell(args.model)
    gs = groesse_aus_modell(model)
    env = baue_env(longitudinal_model=args.physik, global_size=gs)

    roh = env.envs[0].unwrapped
    lang = roh.get_env_config()["longitudinal"]

    print("=" * 66)
    print("RAUCHTEST")
    print("=" * 66)
    print(f"  Modell     : {args.model}")
    print(f"  Algorithmus: {algo}")
    print(f"  Kamera     : {gs[0]}x{gs[1]}  (aus dem Modell abgeleitet)")
    print(f"  Physik     : {lang['model']}")
    for k, wert in lang.items():
        if k != "model":
            print(f"      {k:<16} {wert}")

    r = fahren(model, env, args.max_steps)
    env.close()

    print("\n  Ergebnis:")
    print(f"    Frames bis Ende   : {r['frames']}"
          f"{'  (Crash)' if r['crash'] else '  (Limit erreicht, kein Crash)'}")
    print(f"    Runden            : {r['runden']}")
    print(f"    Geschwindigkeit   : Mittel {r['v_mittel']:.3f} | "
          f"max {r['v_max']:.3f} | Ende {r['v_ende']:.3f}  m/s")
    print(f"    zurueckgelegt     : {r['strecke_m']:.2f} m")
    print(f"    Eingaben (Mittel) : Gas {r['gas_mittel']:.2f} | "
          f"Bremse {r['bremse_mittel']:.2f}")
    print()
    return r


if __name__ == "__main__":
    main()
