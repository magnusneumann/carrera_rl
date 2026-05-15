import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class Carrera2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, track_image_path, car_image_path):
        super().__init__()
        
        self.track_image_path = track_image_path
        self.car_image_path = car_image_path
        
        # --- Echte physikalische Parameter (SI-Einheiten) ---
        self.dt = 1/30.0  
        self.L = 0.058  # 58 mm Radstand
        self.mass = 0.080  # 80 g Masse
        self.max_steer_angle = np.radians(30) 
        self.max_speed = 3.0  # m/s
        
        # --- Skalierung ---
        self.pixels_per_meter = 1064.0  

        # --- Gym Spaces ---
        # Action: [Gas (0-1), Bremse (0-1), Lenkung (-1 bis 1)]
        self.action_space = spaces.Box(low=np.array([0, 0, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        # Observation: [x, y, v, theta, omega]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # Rendering Setup
        self.screen = None
        self.clock = pygame.time.Clock()
        self.isopen = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Startzustand (in Metern!). Passe X und Y an, damit das Auto auf der Strecke steht.
        # Beispiel: 0.1m * 1064 = 106 Pixel auf dem Bildschirm.
        self.state = np.array([0.2, 0.2, 0.0, 0.0, 0.0]) 
        self.is_drifting = False
        
        self._init_render()
        
        return self._get_obs(), {}

    def step(self, action):
        gas, brake, steer = action
        x, y, v, theta, omega = self.state

        # 1. Antrieb und Bremse
        force = gas * 1.5 - brake * 2.0 
        dv_dt = (force - (0.5 * v)) / self.mass 
        v_new = v + dv_dt * self.dt
        v_new = np.clip(v_new, -self.max_speed, self.max_speed)

        # 2. Kinematisches Einspurmodell
        delta = steer * self.max_steer_angle 
        
        if np.abs(v_new) > 0.01:
            omega_new = (v_new * np.tan(delta)) / self.L
        else:
            omega_new = 0.0 

        # 3. Grip-Verlust
        radius = v_new / omega_new if np.abs(omega_new) > 0.001 else np.inf
        centripetal_force = self.mass * (v_new**2) / np.abs(radius)
        max_friction_force = self.mass * 9.81 * 1.2 
        
        if centripetal_force > max_friction_force:
            self.is_drifting = True
            omega_new *= 0.2 
        else:
            self.is_drifting = False

        # 4. Position updaten
        theta_new = theta + omega_new * self.dt
        x_new = x + (v_new * np.cos(theta_new)) * self.dt
        y_new = y + (v_new * np.sin(theta_new)) * self.dt

        self.state = np.array([x_new, y_new, v_new, theta_new, omega_new])
        
        # Dummy Reward/Done für den WASD-Test
        reward = 0.0
        terminated = False
        truncated = False
        info = {"is_drifting": self.is_drifting}

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return self.state.astype(np.float32)

    def _init_render(self):
        if self.screen is None:
            pygame.init()
            self.track_img = pygame.image.load(self.track_image_path)
            self.screen_width, self.screen_height = self.track_img.get_size()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Carrera RL - WASD Prototyp")
            
            # Autobild laden und anhand der Skalierung anpassen (104x41 Pixel)
            car_surface = pygame.image.load(self.car_image_path).convert_alpha()
            self.car_img = pygame.transform.scale(car_surface, (104, 41))

    def render(self):
        if self.screen is None:
            return

        # Hintergrund zeichnen
        self.screen.blit(self.track_img, (0, 0))
        
        x_m, y_m, _, theta, _ = self.state
        
        # Meter in Pixel umrechnen
        pixel_x = int(x_m * self.pixels_per_meter)
        pixel_y = int(y_m * self.pixels_per_meter)
        
        # Auto rotieren (Pygame rotiert gegen den Uhrzeigersinn, also Winkel negativ machen)
        # Pygame 0 Grad ist rechts, was zu unserer Mathematik passt.
        angle_deg = np.degrees(-theta)
        rotated_car = pygame.transform.rotate(self.car_img, angle_deg)
        
        # Rotationszentrum korrigieren, damit es nicht eiert
        rect = rotated_car.get_rect(center=(pixel_x, pixel_y))
        self.screen.blit(rotated_car, rect.topleft)
        
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.isopen = False