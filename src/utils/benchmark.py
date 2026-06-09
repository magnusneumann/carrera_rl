import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from thop import profile
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from src.envs.carrera_2d_env import Carrera2DEnv
from IPython.display import display

class MultiInputWrapper(torch.nn.Module):
    """Wrapper, damit thop (FLOP-Profiler) mit Dict-Spaces umgehen kann."""
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
        
    def forward(self, image, proprioception):
        return self.extractor({"image": image, "proprioception": proprioception})

def calculate_training_time_from_name(zip_path):
    """
    Extrahiert den Startzeitpunkt aus dem Dateinamen (Format: ..._YYYYMMDD_HHMM.zip)
    und vergleicht ihn mit der Erstellungs-/Änderungszeit der Datei.
    """
    try:
        basename = os.path.basename(zip_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        parts = name_without_ext.split('_')
        if len(parts) < 2:
            return 0.0
            
        time_str = f"{parts[-2]}_{parts[-1]}"
        start_time = datetime.strptime(time_str, "%Y%m%d_%H%M")
        
        end_time_ts = os.path.getmtime(zip_path)
        end_time = datetime.fromtimestamp(end_time_ts)
        
        duration_minutes = (end_time - start_time).total_seconds() / 60.0
        return max(0.0, duration_minutes)
        
    except ValueError:
        print(f"  [Warnung] Zeitstempel in {basename} entspricht nicht '%Y%m%d_%H%M'.")
        return 0.0
    except Exception as e:
        print(f"  [Warnung] Zeitberechnung für {basename} fehlgeschlagen: {e}")
        return 0.0

def run_benchmark(eval_models, render_mode="hidden", max_frames=700, track_path='data/strecke.png', car_path='data/carrera_car.png'):
    """
    Führt den Benchmark für eine Liste von Modellen durch und liefert einen DataFrame zurück.
    """
    results = []
    print(f"Starte Evaluation für {len(eval_models)} Modelle...\n")

    for cfg in eval_models:
        name = cfg["name"]
        zip_path = cfg["model_path"]
        obs_type = cfg["obs_type"]
        is_stacked = cfg.get("is_stacked", False)
        camera_view = cfg.get("camera_view", "crop")
        
        if not os.path.exists(zip_path):
            print(f"Überspringe '{name}': Datei {zip_path} nicht gefunden.")
            continue
            
        print(f"--> Evaluiere: {name}")
        
        # --- A) Trainingszeit ---
        train_time_min = calculate_training_time_from_name(zip_path)

        # Modell laden
        model = PPO.load(zip_path)
        device = model.device
        
        # Trainings-Metadaten für die GFLOPs-Schätzung extrahieren
        total_steps = getattr(model, "num_timesteps", 0)
        n_epochs = getattr(model, "n_epochs", 10)

        # --- B) FLOPs (Inferenz & Training) ---
        macs = 0
        try:
            if obs_type == "lidar":
                obs_shape = model.observation_space.shape
                dummy = torch.randn(1, *obs_shape).to(device)
                macs, _ = profile(model.policy.features_extractor, inputs=(dummy,), verbose=False)
                
            elif obs_type == "vision":
                obs_shape = model.observation_space.shape
                dummy = torch.randn(1, *obs_shape).to(device)
                macs, _ = profile(model.policy.features_extractor, inputs=(dummy,), verbose=False)
                
            elif obs_type == "multi":
                img_shape = model.observation_space.spaces["image"].shape
                prop_shape = model.observation_space.spaces["proprioception"].shape
                
                img_dummy = torch.randn(1, *img_shape).to(device)
                prop_dummy = torch.randn(1, *prop_shape).to(device)
                
                wrapper = MultiInputWrapper(model.policy.features_extractor)
                macs, _ = profile(wrapper, inputs=(img_dummy, prop_dummy), verbose=False)
                
        except Exception as e:
            print(f"  [Warnung] FLOPs konnten nicht berechnet werden: {e}")

        # Inferenz-GFLOPs berechnen (1 MAC = 2 FLOPs)
        inference_gflops = (macs * 2) / 1e9 if macs > 0 else 0.0
        
        # Gesamtaufwand Training schätzen: Rollout + Optimierung
        # Formel: Steps * (1 Forward + 3 Backward * n_epochs)
        training_gflops = inference_gflops * total_steps * (1 + 3 * n_epochs)

        # --- C) Simulation ---
        def make_env():
            return Carrera2DEnv(track_path, car_path, obs_type=obs_type, render_mode=render_mode, camera_view=camera_view)

        if is_stacked:
            env = VecFrameStack(DummyVecEnv([make_env]), n_stack=3)
            obs = env.reset()
        else:
            env = make_env()
            obs, _ = env.reset()

        total_reward = 0.0
        status = "DNF (Crashed/Timeout)"
        laps_crossed = 0
        lap_start_frame = -1
        best_lap_frames = -1
        in_sf_zone = False

        for step in range(max_frames):
            action, _ = model.predict(obs, deterministic=False)
            
            if is_stacked:
                obs, rewards, dones, infos = env.step(action)
                r, d, inf = rewards[0], dones[0], infos[0]
                env.render()
            else:
                obs, r, term, trunc, inf = env.step(action)
                d = term or trunc
                env.render()

            total_reward += float(r)
            curr_sf = inf.get('sf_crossed', False)
            
            if curr_sf and not in_sf_zone:
                laps_crossed += 1
                if laps_crossed == 1:
                    lap_start_frame = step
                elif laps_crossed == 2:
                    best_lap_frames = step - lap_start_frame
                    status = "Finished"
                    break
                    
            in_sf_zone = curr_sf

            if d or inf.get('is_crashing', False):
                break

        env.close()

        # --- D) Speichern ---
        results.append({
            "Model Name": name,
            "Type": obs_type.upper() + (" (Stack)" if is_stacked else ""),
            "Train Time (min)": round(train_time_min, 1) if train_time_min > 0 else "-",
            "~GFLOPs Inferenz": round(inference_gflops, 6),
            "~GFLOPs Training": round(training_gflops, 2) if training_gflops > 0 else "-",
            "Test Reward": round(total_reward, 1),
            "Lap Time (Frames)": best_lap_frames if best_lap_frames > 0 else "-",
            "Status": status
        })

    print("\n--- EVALUATION ABGESCHLOSSEN ---")
    df = pd.DataFrame(results)
    return df