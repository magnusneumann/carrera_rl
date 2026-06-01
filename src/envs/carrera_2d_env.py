import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from collections import deque
from shapely import LineString, Polygon

from src.utils.reward_func import RewardCalculator
from src.utils.model_free import SensorSuite

# WICHTIG: Die neue Virtuelle Kamera importieren
from src.utils.virtual_camera import VirtualCamera

class Carrera2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "hidden"], "render_fps": 30}

    def __init__(self, track_image_path, car_image_path, obs_type="lidar", render_mode="hidden"):
        super().__init__()
        
        self.track_image_path = track_image_path
        self.car_image_path = car_image_path
        self.obs_type = obs_type # "lidar", "vision", oder "multi"
        
        # --- Physikalische Parameter (SI-Einheiten) ---
        self.dt = 1/30.0  
        self.L = 0.058  # 58 mm Radstand
        self.mass = 0.080  # 80 g Masse
        self.max_steer_angle = np.radians(30) # 30 Grad Lenkwinkel
        self.max_steer_change = 1.0 # Wie viel sich der Lenkwert pro Step ändern darf
        self.max_speed = 1.6  # m/s
        self.mu = 0.3  # Reibwert für Grip-Limit (Untersteuern)
        
        # 1 Meter = 236 Pixel
        self.pixels_per_meter = 236.0  

        # --- Gym Spaces: Actions ---
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1]), # Gas, Bremse, Lenken
            high=np.array([1, 1, 1]), 
            dtype=np.float32
        )

        # --- Gym Spaces: Observations (Dynamisch!) ---
        if self.obs_type == "lidar":
            # [v_norm, dist_l, dist_m, dist_r]
            self.observation_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32), 
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), 
                dtype=np.float32
            )
        elif self.obs_type == "vision":
            # high=255 und dtype=np.uint8
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
            )
            self.camera = VirtualCamera(crop_size=(150, 150), target_size=(84, 84))
            
        elif self.obs_type == "multi":
            self.observation_space = spaces.Dict({
                # Hier ebenfalls uint8 und 255
                "image": spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8),
                "speed": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            })
            self.camera = VirtualCamera(crop_size=(150, 150), target_size=(84, 84))
        else:
            raise ValueError("obs_type muss 'lidar', 'vision' oder 'multi' sein.")

        # Rendering Setup
        self.screen = None
        self.clock = pygame.time.Clock()    
        self.isopen = True
        self.frame_count = 0
        self.render_mode = render_mode

        # --- Shapely Track Limits laden ---
        self.outer_points = np.load('data/outer_raw_spline.npy')
        inner_points = np.load('data/inner_raw_spline.npy')
        
        self.outer_wall = LineString(self.outer_points).buffer(3.0, cap_style=1, join_style=1)
        self.inner_wall = LineString(inner_points).buffer(3.0, cap_style=1, join_style=1)
        self.sf_line = LineString([(462, 55), (468, 114)])
        
        # Reward & Status
        self.reward_calculator = RewardCalculator()
        self.sf_crossed_last_frame = False
        self.sensor_suite = SensorSuite(self.outer_wall, self.inner_wall) 
        self.episode_reward = 0.0
        self.current_steer = 0.0
        self.last_steer_val = 0.0
        self.steer_delta_history = deque(maxlen=30)
        self.frames_since_lap = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Startzustand: x, y, v, theta, omega
        self.state = np.array([2.1, 0.4, 0.0, 0.0, 0.0]) 
        
        self._init_render()
        
        # WICHTIG FÜR VISION: 
        # Wenn die KI ein Bild sehen soll, MUSS die Welt vor dem allerersten _get_obs() 
        # einmal gezeichnet werden, sonst ist das Bild komplett schwarz!
        if self.obs_type in ["vision", "multi"]:
            self.render()

        self.frame_count = 0
        self.episode_reward = 0.0
        self.current_steer = 0.0
        self.frames_since_lap = 0
        
        return self._get_obs(), {}

    def step(self, action):
        terminated = False
        gas, brake, target_steer = action
        x, y, v, theta, omega = self.state

        # --- 1. Lenk-Trägheit ---
        self.last_steer_val = self.current_steer
        steer_diff = target_steer - self.current_steer
        steer_diff = np.clip(steer_diff, -self.max_steer_change, self.max_steer_change)
        self.current_steer += steer_diff
        
        # --- 2. Physik-Engine Logik ---
        force = gas * 1.0 - brake * 0.4 
        dv_dt = (force - (0.5 * v)) / self.mass 
        v_new = v + dv_dt * self.dt
        v_new = np.clip(v_new, -self.max_speed, self.max_speed)

        delta = self.current_steer * self.max_steer_angle 
        
        if np.abs(v_new) > 0.01:
            omega_theoretisch = (v_new * np.tan(delta)) / self.L
            a_max = self.mu * 9.81 
            omega_max = a_max / np.abs(v_new)
            omega_new = np.clip(omega_theoretisch, -omega_max, omega_max)
        else:
            omega_new = 0.0 

        theta_new = theta + omega_new * self.dt
        dx_dt = v_new * np.cos(theta_new)
        dy_dt = v_new * np.sin(theta_new)
        x_new = x + dx_dt * self.dt
        y_new = y + dy_dt * self.dt

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

        # --- 4. Zustände prüfen ---
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
        self.frames_since_lap +=1 #frame zählen, für Rundenzeiten
        
        is_new_lap = sf_crossed and not self.sf_crossed_last_frame
        self.sf_crossed_last_frame = sf_crossed

        steer_delta = abs(steer_diff)
        self.steer_delta_history.append(steer_delta)
        
        reward, terminated_from_calc = self.reward_calculator.calculate(
            v=v_new,
            is_crashing=is_crashing,
            sf_crossed=sf_crossed,
            is_new_lap=is_new_lap,
            correct_direction=correct_direction,
            steer_delta=steer_delta,
            steer_delta_history_sum=sum(self.steer_delta_history),
            lap_frames=self.frames_since_lap
        )
        
        if is_new_lap:
            self.frames_since_lap = 0

        if terminated_from_calc:
            terminated = True

        self.episode_reward += reward

        if self.episode_reward <= -3000:
            terminated = True
            reward -= 100.0

        self.frame_count += 1
        truncated = self.frame_count >= 2000

        # WICHTIG: Wenn wir mit Bildern arbeiten, müssen wir den Screen VOR _get_obs() zeichnen!
        if self.obs_type in ["vision", "multi"]:
            self.render()

        obs = self._get_obs()
        
        info = {
            'is_crashing': is_crashing,
            'correct_direction': correct_direction,
            'sf_crossed': sf_crossed,
            'actual_steer': self.current_steer 
        }
        
        reward = float(reward) 
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        x_m, y_m, v, theta, _ = self.state
        cx = x_m * self.pixels_per_meter
        cy = y_m * self.pixels_per_meter
        
        # --- Modus 1: Lidar ---
        if self.obs_type == "lidar":
            sensor_array, self.last_ray_lines = self.sensor_suite.get_lidar_observation(cx, cy, theta, v)
            dist_links = sensor_array[1]
            dist_mitte = sensor_array[2] 
            dist_rechts = sensor_array[3]
            return np.array([v / self.max_speed, dist_links, dist_mitte, dist_rechts], dtype=np.float32)
        
        # --- Modus 2 & 3: Vision / Multi ---
        # In carrera_2d_env.py -> _get_obs()
        
        # --- Modus 2 & 3: Vision / Multi ---
        else:
            # Fallback, falls screen leer ist (MUSS AUCH uint8 sein!)
            if self.screen is None:
                rl_image = np.zeros((1, 84, 84), dtype=np.uint8)
            else:
                rl_image, _ = self.camera.get_car_centric_observation(self.screen, cx, cy)
            if self.obs_type == "vision":
                return rl_image
            elif self.obs_type == "multi":
                return {
                    "image": rl_image,
                    "speed": np.array([v / self.max_speed], dtype=np.float32)
                }

    def _init_render(self):
        if self.screen is None:
            pygame.init()
            self.track_img = pygame.image.load(self.track_image_path)
            self.screen_width, self.screen_height = self.track_img.get_size()
            
            # WICHTIG: Wir nutzen pygame.HIDDEN, damit nicht beim Training 100 Fenster aufpoppen!
            # Wenn du es sehen willst, machst du das beim Trainieren über env.render("human")
            if self.render_mode == "hidden":
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.HIDDEN)
            else:
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Carrera RL")
            
            car_surface = pygame.image.load(self.car_image_path).convert_alpha()
            self.rotated_car_initial_state = pygame.transform.rotate(car_surface, 180)
            self.car_img = pygame.transform.scale(self.rotated_car_initial_state, (23, 9))

    def render(self):
        if self.screen is None:
            return

        self.screen.blit(self.track_img, (0, 0))
        x_m, y_m, _, theta, _ = self.state
        
        pixel_x = int(x_m * self.pixels_per_meter)
        pixel_y = int(y_m * self.pixels_per_meter)
        
        desired_angle_deg = np.degrees(theta)
        pygame_rotation = 180 - desired_angle_deg

        rotated_car = pygame.transform.rotate(self.car_img, pygame_rotation)
        rect = rotated_car.get_rect(center=(pixel_x, pixel_y))
        self.screen.blit(rotated_car, rect.topleft)
        
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False