"""Lidar-Experten unter der neuen Physik trainieren.

Entspricht der Notebook-Zelle 28, aber als Skript und mit Konfigurations-
protokoll. Der Experte dient anschliessend als Lehrer fuer den Bilddatensatz
(Notebook-Zelle 52), auf dem das CNN vortrainiert wird.

Warum neu: die Lidar-Beobachtung enthaelt v / max_speed. Dieser Nenner ist
mit measured_v2 von 1.6 auf 1.9 gewechselt, und die Laengsdynamik darunter
ist eine andere. Ein unter force_drag_v1 trainierter Experte sieht dieselbe
Geschwindigkeit jetzt als kleinere Zahl und faehrt entsprechend falsch.

Aufruf:
    python -m src.train.lidar_experte
    python -m src.train.lidar_experte --steps 150000
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.envs.carrera_2d_env import Carrera2DEnv
from src.utils.reward_func import RewardCalculator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80000)
    p.add_argument("--physik", default="measured_v2",
                   choices=["measured_v2", "force_drag_v1"])
    args = p.parse_args()

    stempel = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"01_Lidar_Baseline_{stempel}"
    log_dir = "./tensorboard_logs/"

    env = Monitor(Carrera2DEnv('data/strecke.png', 'data/carrera_car.png',
                               obs_type="lidar", render_mode="hidden",
                               longitudinal_model=args.physik))

    cfg = {
        "run_name": run_name,
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "obs_type": "lidar",
        "total_timesteps": args.steps,
        "n_envs": 1,
        "zweck": "Lehrer fuer den Bilddatensatz (Behaviour Cloning des CNN)",
        "reward": RewardCalculator().to_dict(),
        "env": env.unwrapped.get_env_config(),
    }

    print("=" * 66)
    print(f"  {run_name}")
    print(f"  Physik : {cfg['env']['longitudinal']['model']}  "
          f"(v_max {cfg['env']['longitudinal']['max_speed_ms']})")
    print(f"  Reward : {cfg['reward']['version']}")
    print(f"  Steps  : {args.steps:,}")
    print("=" * 66)

    # device="cpu": SB3 empfiehlt das ausdruecklich fuer MlpPolicy, auf der GPU
    # waere es hier langsamer. tb_log_name=run_name, damit das Tensorboard-Log
    # dem Lauf zuzuordnen ist - mit dem voreingestellten "PPO" entstehen sonst
    # nur durchnummerierte Ordner ohne Bezug.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                device="cpu")
    model.learn(total_timesteps=args.steps, tb_log_name=run_name)

    ziel = Path("models") / f"{run_name}.zip"
    model.save(str(ziel.with_suffix("")))
    Path("models").mkdir(exist_ok=True)
    (Path("models") / f"{run_name}_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env.close()
    print(f"\nModell : {ziel}")
    print(f"Config : models/{run_name}_config.json")
    print(f"\nFuer die Datensammlung in Notebook-Zelle 52 eintragen:")
    print(f'    expert_model = PPO.load("models/{run_name}.zip")')


if __name__ == "__main__":
    main()
