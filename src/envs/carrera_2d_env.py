import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
import math
from collections import deque
from shapely import LineString, Polygon
from src.utils.reward_func import RewardCalculator
from src.utils.model_free import SensorSuite

class Carrera2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, track_image_path, car_image_path):
        super().__init__()
        
        self.track_image_path = track_image_path
        self.car_image_path = car_image_path
        
        # --- Physikalische Parameter (SI-Einheiten) ---
        self.dt = 1/30.0  
        self.L = 0.058  # 58 mm Radstand
        self.mass = 0.080  # 80 g Masse
        self.max_steer_angle = np.radians(30) # 30 Grad Lenkwinkel
        self.max_steer_change = 0.2 # Wie viel sich der Lenkwert pro Step ändern darf (0.0 bis 2.0 Bereich)
        self.max_speed = 1.6  # m/s
        self.mu = 0.3  # Reibwert für Grip-Limit (Untersteuern)
        
        # --- MASTER SKALIERUNGS-FAKTOR (basiert auf 830px Streckenbild) ---
        # 1 Meter = 236 Pixel
        self.pixels_per_meter = 236.0  

        # --- Gym Spaces ---
        self.action_space = spaces.Box(low=np.array([0, 0, -1]), #Gas Bremse 0 bis 1 und Lenken -1 bis 1
                                       high=np.array([1, 1, 1]), 
                                       dtype=np.float32)
        # [v_norm, dist_l, dist_r]
        # Geschwindigkeit darf von -1.0 bis 1.0 gehen. Raycasts von 0.0 bis 1.0.
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
)
        # Rendering Setup
        self.screen = None
        self.clock = pygame.time.Clock()    
        self.isopen = True
        self.frame_count = 0

        # Reward Calculator
        # --- Shapely Track Limits laden ---
        self.outer_points = np.load('data/outer_raw_spline.npy')
        inner_points = np.load('data/inner_raw_spline.npy')
        
        self.outer_wall = LineString(self.outer_points).buffer(3.0, cap_style=1, join_style=1)
        self.inner_wall = LineString(inner_points).buffer(3.0, cap_style=1, join_style=1)
        self.sf_line = LineString([(462, 55), (468, 114)])
        
        # Reward & Status
        self.reward_calculator = RewardCalculator()
        self.sf_crossed_last_frame = False
        self.sensor_suite = SensorSuite(self.outer_wall, self.inner_wall) #maximale Raycastlänge in model_free.py definiert
        self.episode_reward = 0.0
        self.current_steer = 0.0
        self.last_steer_val = 0.0
        self.steer_delta_history = deque(maxlen=30)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Startzustand (in Metern!). Passe X und Y an, damit das Auto auf der Strecke steht.
        # Beispiel: 0.5m * 236 = 118 Pixel auf dem Bildschirm.
        self.state = np.array([2.1, 0.4, 0.0, 0.0, 0.0]) # x, y, v, theta, omega hier ist der spawn
        
        # Initialisiere Pygame und lade Bilder
        self._init_render()
        self.frame_count = 0
        self.episode_reward = 0.0
        self.current_steer = 0.0
        return self._get_obs(), {}

    def step(self, action):
        terminated = False
        gas, brake, target_steer = action # Wir nennen es target_steer
        x, y, v, theta, omega = self.state

        # --- 1. Lenk-Trägheit (Rate Limiting) ---
        # Die KI will 'target_steer', aber wir erlauben nur eine kleine Änderung
        self.last_steer_val = self.current_steer
        steer_diff = target_steer - self.current_steer #geforderte Lenkwinkel änderung
        # Wir begrenzen die Änderung
        steer_diff = np.clip(steer_diff, -self.max_steer_change, self.max_steer_change)
        
        # Der neue tatsächliche Lenkwert
        self.current_steer += steer_diff
        
# --- 2. Physik-Engine Logik (mit self.current_steer) ---
        force = gas * 1.0 - brake * 0.4 
        dv_dt = (force - (0.5 * v)) / self.mass 
        v_new = v + dv_dt * self.dt
        v_new = np.clip(v_new, -self.max_speed, self.max_speed)

        # Wir nutzen jetzt den trägen Lenkwert für die Geometrie
        delta = self.current_steer * self.max_steer_angle 
        
        if np.abs(v_new) > 0.01:
            # 1. Theoretische Winkelgeschwindigkeit ohne Grip-Limit berechnen
            omega_theoretisch = (v_new * np.tan(delta)) / self.L
            
            # 2. Grip-Limit definieren (Reibwert mu z.B. 0.6 für glatte Carrera-Schiene)
            a_max = self.mu * 9.81  # Maximale Querbeschleunigung in m/s² VEREINFACHTER KAMMSCHERKREIS
            
            # 3. Maximal erlaubtes Omega berechnen basierend auf v_new
            omega_max = a_max / np.abs(v_new)
            
            # 4. Echtes Omega clippen -> Simuliert das Untersteuern!
            omega_new = np.clip(omega_theoretisch, -omega_max, omega_max)
        else:
            omega_new = 0.0 

        theta_new = theta + omega_new * self.dt
        dx_dt = v_new * np.cos(theta_new)
        dy_dt = v_new * np.sin(theta_new)
        x_new = x + dx_dt * self.dt
        y_new = y + dy_dt * self.dt

        # Zustand speichern
        self.state = np.array([x_new, y_new, v_new, theta_new, omega_new])

        # --- 3. Hitbox (car_poly) berechnen ---
        cx = x_new * self.pixels_per_meter
        cy = y_new * self.pixels_per_meter
        
        HITBOX_L, HITBOX_W = 20.0, 7.0
        dx_f = (HITBOX_L / 2) * math.cos(theta_new)
        dy_f = (HITBOX_L / 2) * math.sin(theta_new)
        dx_s = (HITBOX_W / 2) * -math.sin(theta_new)
        dy_s = (HITBOX_W / 2) * math.cos(theta_new)

        self.car_corners = [
            (cx + dx_f + dx_s, cy + dy_f + dy_s),
            (cx + dx_f - dx_s, cy + dy_f - dy_s),
            (cx - dx_f - dx_s, cy - dy_f - dy_s),
            (cx - dx_f + dx_s, cy - dy_f + dy_s)
        ]
        self.car_poly = Polygon(self.car_corners)

        # --- 4. Zustände prüfen (Schiedsrichter) ---
        is_crashing = self.car_poly.intersects(self.outer_wall) or self.car_poly.intersects(self.inner_wall)
        sf_crossed = self.car_poly.intersects(self.sf_line)
        
        dists = np.linalg.norm(self.outer_points - [cx, cy], axis=1)
        nearest_idx = np.argmin(dists)
        next_idx = (nearest_idx + 5) % len(self.outer_points)
        
        p1, p2 = self.outer_points[nearest_idx], self.outer_points[next_idx]
        track_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        angle_diff = (theta_new - track_angle + math.pi) % (2 * math.pi) - math.pi
        correct_direction = abs(angle_diff) < (math.pi / 2)

        # --- 5. Reward berechnen ---
        is_new_lap = sf_crossed and not self.sf_crossed_last_frame
        self.sf_crossed_last_frame = sf_crossed

        # 1. Delta & History berechnen
        steer_delta = abs(steer_diff)
        self.steer_delta_history.append(steer_delta)
        
        # 2: Das Tuple sauber entpacken (mit , terminated_from_calc)
        reward, terminated_from_calc = self.reward_calculator.calculate(
            v=v_new,
            is_crashing=is_crashing,
            sf_crossed=sf_crossed,
            is_new_lap=is_new_lap,
            correct_direction=correct_direction,
            steer_delta=steer_delta,
            steer_delta_history_sum=sum(self.steer_delta_history)
        )
        
        if terminated_from_calc:
            terminated = True

        # Jetzt ist 'reward' wieder ein reiner Float!
        self.episode_reward += reward

        if self.episode_reward <= -3000:
            terminated = True
            reward -= 100.0

        # Frame-Counter und Truncation
        self.frame_count += 1
        truncated = self.frame_count >= 2000

        obs = self._get_obs()
        
        # Info für Debugging
        info = {
            'is_crashing': is_crashing,
            'correct_direction': correct_direction,
            'sf_crossed': sf_crossed,
            'actual_steer': self.current_steer # Sehr hilfreich zum Debuggen!
        }
        reward = float(reward) # Sicherstellen, dass es ein Float ist
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        x_m, y_m, v, theta, _ = self.state
        cx = x_m * self.pixels_per_meter
        cy = y_m * self.pixels_per_meter
        
        # sensor_array enthält jetzt [v, dist_links, dist_rechts]
        sensor_array, self.last_ray_lines = self.sensor_suite.get_lidar_observation(cx, cy, theta, v)
        
        dist_links = sensor_array[1]
        dist_rechts = sensor_array[2]
        
        # Wir geben [v_normiert, dist_links, dist_rechts] zurück
        obs = np.array([v / self.max_speed, dist_links, dist_rechts], dtype=np.float32)
        return obs

    def _init_render(self):
        if self.screen is None:
            pygame.init()
            self.track_img = pygame.image.load(self.track_image_path)
            self.screen_width, self.screen_height = self.track_img.get_size()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Carrera RL - WASD Prototyp")
            
            # Lade .png für Transparenz
            car_surface = pygame.image.load(self.car_image_path).convert_alpha()
            
            # --- Grafik 180° drehen ---
            # Das Originalbild zeigt nach oben (UP). Rotieren um 180°. Auto zeigt jetzt nach unten.
            self.rotated_car_initial_state = pygame.transform.rotate(car_surface, 180)
            
            # --- Skalieren auf physikalisch korrekte Größe ---
            # Länge: 0.098m * 236 px/m = 23 Pixel (vertical Axis y)
            # Breite: 0.039m * 236 px/m = 9 Pixel (horizontal Axis x)
            self.car_img = pygame.transform.scale(self.rotated_car_initial_state, (23, 9))

    def render(self):
        if self.screen is None:
            return

        # Hintergrund zeichnen
        self.screen.blit(self.track_img, (0, 0))
        
        x_m, y_m, _, theta, _ = self.state
        
        # Meter in Pixel umrechnen
        pixel_x = int(x_m * self.pixels_per_meter)
        pixel_y = int(y_m * self.pixels_per_meter)
        
        # --- Rotations-Logik ---
        # Unser Initialbild self.car_img zeigt nach UNTEN (Physics 90°).
        # Pygame rotation ist anti-clockwise.
        
        desired_angle_deg = np.degrees(theta)
        pygame_rotation = 180 - desired_angle_deg

        rotated_car = pygame.transform.rotate(self.car_img, pygame_rotation)
        
        # Rotationszentrum korrigieren, damit es nicht eiert
        rect = rotated_car.get_rect(center=(pixel_x, pixel_y))
        self.screen.blit(rotated_car, rect.topleft)
        
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False