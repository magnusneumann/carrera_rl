"""
Sequenzielles Training aller Inputvarianten.

Starten:
    DRY_RUN (Test, kein echtes Training):
        python run_trainings.py --dry-run

    Echtes Training (alle Varianten nacheinander):
        python run_trainings.py
"""

import os
import sys
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack

from src.envs.carrera_2d_env import Carrera2DEnv

# ─── Konfiguration ────────────────────────────────────────────────────────────

N_ENVS = 12
TOTAL_TIMESTEPS = 300_000 * N_ENVS
DRY_RUN_STEPS = 200          # Nur zum Testen ob alles startet

TRAININGS = [
    {
        "name": "04_Vision_Crop_1Frame",
        "policy": "CnnPolicy",
        "obs_type": "vision",
        "camera_view": "crop",
        "frame_stack": 1,
        "desc": "1 Bild, Autofeste Vogelperspektive (84x84)",
    },
    {
        "name": "05_Vision_Global_1Frame",
        "policy": "CnnPolicy",
        "obs_type": "vision",
        "camera_view": "global",
        "frame_stack": 1,
        "desc": "1 Bild, globale Ansicht (166x100)",
    },
    {
        "name": "06_Vision_Crop_3Stack",
        "policy": "CnnPolicy",
        "obs_type": "vision",
        "camera_view": "crop",
        "frame_stack": 3,
        "desc": "3 Bilder als Stack (crop), 3 Kanaele",
    },
    {
        "name": "07_Vision_Global_3Stack",
        "policy": "CnnPolicy",
        "obs_type": "vision",
        "camera_view": "global",
        "frame_stack": 3,
        "desc": "3 Bilder als Stack, globale Ansicht (166x100, 3 Kanaele)",
    },
    {
        "name": "08_Hybrid_Multi",
        "policy": "MultiInputPolicy",
        "obs_type": "multi",
        "camera_view": "crop",
        "frame_stack": 1,
        "desc": "Hybrid: Bild + Geschwindigkeit + Lenkwinkel",
    },
]

# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def make_env_fn(obs_type, camera_view):
    def _init():
        return Carrera2DEnv(
            'data/strecke.png',
            'data/carrera_car.png',
            obs_type=obs_type,
            camera_view=camera_view,
            render_mode="hidden",
        )
    return _init


def run_training(cfg, dry_run: bool):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{cfg['name']}_{timestamp}"
    log_dir = f"./tensorboard_logs/{run_name}/"
    checkpoint_dir = f"./models/checkpoints/{run_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {cfg['desc']}")
    print(f"  Run: {run_name}")
    print(f"  {'[DRY RUN - nur ' + str(DRY_RUN_STEPS) + ' Steps]' if dry_run else str(TOTAL_TIMESTEPS) + ' Steps'}")
    print(f"{'='*60}")

    n_envs = N_ENVS
    vec_cls = SubprocVecEnv
    print(f"  Envs: {n_envs} (SubprocVecEnv)")

    env = make_vec_env(
        make_env_fn(cfg['obs_type'], cfg['camera_view']),
        n_envs=n_envs,
        vec_env_cls=vec_cls,
    )

    if cfg['frame_stack'] > 1:
        env = VecFrameStack(env, n_stack=cfg['frame_stack'])

    model = PPO(
        cfg['policy'],
        env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=64 if dry_run else 512,
        batch_size=64 if dry_run else 256,
        n_epochs=2 if dry_run else 10,
    )

    callbacks = []
    if not dry_run:
        callbacks.append(CheckpointCallback(
            save_freq=50_000,
            save_path=checkpoint_dir,
            name_prefix="ppo",
            verbose=1,
        ))

    steps = DRY_RUN_STEPS if dry_run else TOTAL_TIMESTEPS
    model.learn(total_timesteps=steps, tb_log_name="PPO", callback=callbacks or None)

    if not dry_run:
        save_path = f"models/{run_name}"
        model.save(save_path)
        print(f"  Gespeichert: {save_path}.zip")

    env.close()
    print(f"  Fertig: {cfg['desc']}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Nur testen, kein echtes Training")
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else f"ECHTES TRAINING ({N_ENVS} Envs, {TOTAL_TIMESTEPS:,} Steps je Variante)"
    print(f"\nModus: {mode}")
    print(f"Varianten: {len(TRAININGS)}")

    for i, cfg in enumerate(TRAININGS):
        print(f"\n[{i+1}/{len(TRAININGS)}] Starte: {cfg['desc']}")
        try:
            run_training(cfg, dry_run=args.dry_run)
        except Exception as e:
            print(f"\nFEHLER bei {cfg['name']}: {e}")
            print("Naechste Variante wird trotzdem gestartet...")
            continue

    print(f"\nAlle Trainings abgeschlossen.")
