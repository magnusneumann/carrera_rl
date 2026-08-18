import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
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

    def __init__(self, track_image_path, car_image_path, obs_type="lidar", render_mode="hidden", camera_view="crop", seed=None, longitudinal_model="measured_v2", global_size=(166, 100), max_steer_change=0.5, start_streuung=0.0):
        super().__init__()

        self.track_image_path = track_image_path
        self.car_image_path = car_image_path
        self.obs_type = obs_type # "lidar", "vision", oder "multi"
        self.render_mode = render_mode # "human" oder "hidden"
        self.camera_view = camera_view # "crop" oder "full"
        self.seed_value = seed # nur zur Protokollierung; gesetzt wird er in reset()
        # Groesse der Vogelperspektive (Breite, Hoehe). Bestimmt zugleich, ab
        # welcher Geschwindigkeit sich das Auto ueberhaupt um mehr als einen
        # Pixel je Frame verschiebt - siehe rl_erkenntnisse.md, Abschnitt 13d:
        #     Schwelle [m/s] = 30 / (236 * Breite / 830)
        #     166 px -> 0.64      250 px -> 0.42      300 px -> 0.35
        self.global_size = tuple(global_size)
        # Wie weit der Lenkeinschlag je Frame springen darf. Am 28.07. von 1.0
        # auf 0.5 gesetzt, zusammen mit dem gemessenen Fahrzeugmodell. Als
        # Parameter, damit sich aeltere Laeufe originalgetreu nachfahren
        # lassen - alles bis 1432 lief mit 1.0.
        self._max_steer_change_arg = max_steer_change
        # Wie weit der Startpunkt nach HINTEN streuen darf, in Metern entlang
        # der Geraden vor der Ziellinie. 0.0 = fester Startpunkt wie bisher.
        #
        # Ohne Streuung laeuft jede Episode identisch ab: gleicher Start,
        # deterministische Physik, bei der Auswertung deterministische Policy.
        # Fuenf Auswertungsepisoden liefern dann fuenfmal dasselbe Ergebnis,
        # erkennbar an der Streuung von exakt +/- 0.00 in den Logs. Wir zahlen
        # fuenffachen Aufwand fuer eine einzige Messung und koennen nicht
        # unterscheiden, ob ein Wert typisch oder ein Gluecksfall ist.
        #
        # Die Gerade traegt rund 0.94 m: Mittellinie konstant bei y = 82 px
        # von x = 220 bis zur Ziellinie bei x = 465. Der feste Startpunkt
        # liegt bei x = 451, also 14 px davor.
        self.start_streuung = float(start_streuung)
        self._gesaet = False   # siehe reset(): der Seed wirkt nur einmal
        
        # --- Physikalische Parameter (SI-Einheiten) ---
        self.dt = 1/30.0
        self.L = 0.058  # 58 mm Radstand
        self.max_steer_angle = np.radians(30) # 30 Grad Lenkwinkel
        # Wie viel sich der Lenkwert (-1..1) pro Step ändern darf. Der Weg von
        # Anschlag zu Anschlag beträgt 2.0, bei 0.5 also 4 Frames = 133 ms.
        # Vorher 1.0, was 67 ms entsprach - schneller als ein echter Servo.
        # Noch nicht am Auto gemessen, 0.5 ist eine plausible Schätzung.
        self.max_steer_change = self._max_steer_change_arg
        self.mu = 0.3  # Reibwert für Grip-Limit (Untersteuern)

        # --- Längsdynamik ---
        # Umschaltbar. Diese Werte werden von get_env_config() ausgelesen und
        # in train_config.json geschrieben. Sie MÜSSEN Attribute bleiben und
        # dürfen in step() nicht als Literal wiederholt werden, sonst driftet
        # das Log vom tatsächlichen Verhalten weg.
        #
        #   "force_drag_v1"  altes Modell, F = gas*1.0 - brake*0.4, dv/dt = (F - 0.5v)/m
        #   "measured_v2"    aus Videomessung, siehe
        #                    "Other tools/Auto Parameter messen/Code/fahrzeugmodell.py"
        # Über den Konstruktor umschaltbar, damit A/B-Vergleiche ohne
        # Codeänderung möglich sind.
        if longitudinal_model not in ("force_drag_v1", "measured_v2"):
            raise ValueError(
                f"longitudinal_model muss 'force_drag_v1' oder 'measured_v2' "
                f"sein, nicht {longitudinal_model!r}")
        self.longitudinal_model = longitudinal_model

        # -- force_drag_v1 --
        self.mass = 0.080       # kg   Fahrzeugmasse
        self.f_gas = 1.0        # N    Antriebskraft bei gas = 1
        self.f_brake = 0.4      # N    Bremskraft bei brake = 1
        self.c_drag = 0.5       # Ns/m geschwindigkeitsproportionaler Widerstand

        # -- measured_v2 --
        # dv/dt = gas*A_drive(v) - brake*A_BRAKE - (R0 + R1*v)
        # A_drive(v) = min(unten haftungsbegrenzt, oben motorbegrenzt)
        # Bei Vollgas ergibt das min(0.25 + 0.17v, 2.00*(1 - v/1.90)),
        # bei Vollbremsung -(0.217 + 0.336v). Beides deckt sich mit der Messung.
        # Gemessene Vollgas-Kennlinie: a(v) = min(g0 + g1*v, gm*(1 - v/v_max))
        self._g0, self._g1 = 0.25, 0.17   # gemessen (Haftungsast)
        self._gm = 2.00                   # ANNAHME (Motorast, definiert v_max)
        self.v_max_measured = 1.90        # m/s   Gleichgewicht Antrieb/Widerstand
        self.r1 = 0.336                   # 1/s   Luftwiderstand (gemessen)
        self.r0 = 0.05                    # m/s^2 Rollwiderstand (geschätzt;
                                          #       wirkt NUR aufs Ausrollen, siehe unten)
        # Vollbremsung soll 0.217 + 0.336*v ergeben -> a_brake = 0.217 - r0.
        # Bei Vollgas und Vollbremsung kürzt sich r0 exakt heraus, es beeinflusst
        # deshalb nur das Rollen ohne Eingabe.
        self.a_brake = 0.217 - self.r0
        # Abgeleitet, nicht gerundet hingeschrieben: sonst trifft die Kennlinie
        # bei v_max nicht exakt die Null.
        self.drive_lo = (self._g0 + self.r0, self._g1 + self.r1)
        self.drive_hi = (self._gm + self.r0,
                         self._gm / self.v_max_measured - self.r1)

        if self.longitudinal_model == "measured_v2":
            self.max_speed = self.v_max_measured
            self.allow_reverse = False  # ohne Rückwärtsgang kein Rückwärtsschub
        else:
            self.max_speed = 1.6
            self.allow_reverse = True   # v wird symmetrisch geklemmt

        # 1 Meter = 236 Pixel
        self.pixels_per_meter = 236.0

        # --- Geometrie / Startbedingungen ---
        self.hitbox_l_px = 20.0
        self.hitbox_w_px = 7.0
        self.start_state = [1.91, 0.35, 0.0, 0.0, 0.0]  # x, y, v, theta, omega

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
            if self.camera_view == "global":
                # EINE Quelle fuer die Groesse: sie geht in den Beobachtungsraum
                # und in die Kamera. Vorher stand sie an beiden Stellen getrennt.
                gw, gh = self.global_size
                self.observation_space = spaces.Box(low=0, high=255, shape=(1, gh, gw), dtype=np.uint8)
                self.camera = VirtualCamera(global_size=self.global_size)
            elif self.camera_view == "crop":
                self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
                self.camera = VirtualCamera(crop_size=(150, 150), target_size=(84, 84))
            
        elif self.obs_type == "multi":
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8),
                # Aus "speed" machen wir "proprioception" (Geschwindigkeit und aktueller Lenkwinkel)
                "proprioception": spaces.Box(low=np.array([-1.0, -1.0]), 
                                             high=np.array([1.0, 1.0]), dtype=np.float32)
            })
            self.camera = VirtualCamera(crop_size=(150, 150), target_size=(84, 84))
        else:
            raise ValueError("obs_type muss 'lidar', 'vision' oder 'multi' sein.")

        # Rendering Setup
        self.camera = getattr(self, "camera", None)  # bei obs_type="lidar" nicht vorhanden
        self.screen = None
        self.isopen = True
        self.frame_count = 0
        

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

    def _get_longitudinal_config(self):
        """Nur die Parameter des tatsaechlich aktiven Modells protokollieren.

        Die Werte des inaktiven Modells mitzuschreiben waere irrefuehrend -
        genau die Falle, die 'mass_kg' bisher war: geloggt, aber fuer sich
        allein nicht aussagekraeftig.
        """
        gemeinsam = {
            "model": self.longitudinal_model,
            "max_speed_ms": self.max_speed,
            "allow_reverse": self.allow_reverse,
        }
        if self.longitudinal_model == "measured_v2":
            return {**gemeinsam,
                    "source": "analyse_v10 / kinematik_v10_104532.png",
                    "drive_lo": list(self.drive_lo),
                    "drive_hi": list(self.drive_hi),
                    "a_brake": self.a_brake,
                    "r0": self.r0,
                    "r1": self.r1}
        return {**gemeinsam,
                "mass_kg": self.mass,
                "f_gas_n": self.f_gas,
                "f_brake_n": self.f_brake,
                "c_drag": self.c_drag}

    def _laengsbeschleunigung(self, v, gas, brake):
        """Beschleunigung in m/s^2 nach dem eingestellten Laengsdynamikmodell."""
        if self.longitudinal_model == "measured_v2":
            # Antrieb: unten haftungs-, oben motorbegrenzt
            haftung = self.drive_lo[0] + self.drive_lo[1] * v
            motor   = self.drive_hi[0] - self.drive_hi[1] * v
            antrieb = max(min(haftung, motor), 0.0)
            # Bremse und Widerstand wirken der Bewegung entgegen
            bremsend = brake * self.a_brake + (self.r0 + self.r1 * abs(v))
            richtung = 1.0 if v >= 0.0 else -1.0
            return gas * antrieb - richtung * bremsend

        # force_drag_v1 (Ausgangsmodell)
        force = gas * self.f_gas - brake * self.f_brake
        return (force - (self.c_drag * v)) / self.mass

    def _get_vision_config(self):
        """Parameter der Bilderzeugung.

        Gehört ins Log, weil eine Änderung hier die Beobachtung verändert,
        ohne dass sich Form oder Wertebereich ändern – ein Modell läuft dann
        weiter und fährt nur schlechter, ohne dass irgendetwas warnt.
        Die Umstellung von Pygame auf cv2/numpy am 26.07.2026 (nötig, weil
        12 parallele Envs ebenso viele Fenster-Handles brauchten) war
        nachgemessen beobachtungsneutral, aber das war Glück, nicht Garantie.
        """
        if self.camera is None:
            return {"model": "none", "renderer": self.render_mode}

        cfg = {
            "model": "cv2_numpy_v1" if self.render_mode == "hidden" else "pygame_v1",
            "renderer": self.render_mode,
            "camera_view": self.camera_view,
            "gray_levels": self.camera.levels,
        }
        if self.camera_view == "global":
            cfg["target_hw"] = [self.camera.target_height_global,
                                self.camera.target_width_global]
        else:
            cfg["target_hw"] = [self.camera.target_height,
                                self.camera.target_width]
            cfg["crop_hw"] = [self.camera.crop_height, self.camera.crop_width]
        return cfg

    def get_env_config(self):
        """Vollständiger Zustand der Umgebung für train_config.json.

        Alle Werte werden aus Attributen gelesen, nie als Literal wiederholt –
        sonst kann das Log vom tatsächlichen Verhalten abweichen.

        Der Block "longitudinal" trägt eine Modellkennung. Ohne sie ließen
        sich Läufe mit unterschiedlichen Fahrphysik-Modellen nachträglich
        nicht mehr auseinanderhalten: die Schlüssel unterscheiden sich zwar,
        aber nichts sagt, welches Modell gemeint war.
        """
        return {
            "dt": self.dt,
            "L_m": self.L,
            "max_steer_angle_deg": float(np.degrees(self.max_steer_angle)),
            "max_steer_change": self.max_steer_change,
            "mu": self.mu,
            "pixels_per_meter": self.pixels_per_meter,

            "longitudinal": self._get_longitudinal_config(),

            "geometry": {
                "hitbox_px": [self.hitbox_l_px, self.hitbox_w_px],
                "start_state": list(self.start_state),
                "start_streuung_m": self.start_streuung,
            },

            "vision": self._get_vision_config(),

            "assets": {
                "track": self.track_image_path,
                "car": self.car_image_path,
            },

            "seed": self.seed_value,

            # Rückwärtskompatibel: ältere Auswertungen lesen env["max_speed_ms"]
            # direkt. "mass_kg" gilt nur für longitudinal.model ==
            # "force_drag_v1" und steht deshalb nur dort.
            "max_speed_ms": self.max_speed,
        }

    def reset(self, seed=None, options=None):
        # Ein beim Konstruktor übergebener Seed gilt, solange der Aufrufer
        # keinen eigenen mitgibt. So landet derselbe Wert in reset() und in
        # get_env_config() – sonst protokollierten wir etwas, das nie wirkt.
        if seed is None and self.seed_value is not None and not self._gesaet:
            # Nur EINMAL saeen. Wuerde bei jedem reset() derselbe Seed gesetzt,
            # zoege der Zufallsgenerator jedes Mal dieselbe Zahl - zufaellige
            # Startpunkte waeren dann in jeder Episode identisch. Der Lauf
            # bleibt trotzdem reproduzierbar, weil die Kette der Ziehungen
            # deterministisch ist.
            seed = self.seed_value
        if seed is not None:
            self.seed_value = seed
            self._gesaet = True
            super().reset(seed=seed)
        else:
            super().reset()

        # Startzustand: x, y, v, theta, omega
        self.state = np.array(self.start_state, dtype=float)
        if self.start_streuung > 0.0:
            # Nur nach hinten verschieben, damit die Ziellinie noch vor dem
            # Auto liegt und die Rundenzaehlung ihren Sinn behaelt. Richtung
            # und Geschwindigkeit bleiben unangetastet.
            self.state[0] -= self.np_random.uniform(0.0, self.start_streuung)
        
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
        self.reward_calculator.neue_episode()
        
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
        dv_dt = self._laengsbeschleunigung(v, gas, brake)
        v_new = v + dv_dt * self.dt
        # Bremse und Widerstand duerfen das Auto nicht durch die Null hindurch
        # rueckwaerts ziehen - ohne diese Klemme rollt es ohne Gas rueckwaerts an.
        if v > 0.0 and v_new < 0.0 and gas <= 0.0:
            v_new = 0.0
        v_min = -self.max_speed if self.allow_reverse else 0.0
        v_new = np.clip(v_new, v_min, self.max_speed)

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
        
        HITBOX_L, HITBOX_W = self.hitbox_l_px, self.hitbox_w_px
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
        else:
            if self.screen is None:
                rl_image = np.zeros((1, 84, 84), dtype=np.uint8)
            else:
                if self.camera_view == "global":
                    rl_image, _ = self.camera.get_global_observation(self.screen)
                else: # Default: "crop"
                    rl_image, _ = self.camera.get_car_centric_observation(self.screen, cx, cy, theta)
            
            if self.obs_type == "vision":
                return rl_image
            elif self.obs_type == "multi":
                return {
                    "image": rl_image,
                    "proprioception": np.array([v / self.max_speed, self.current_steer], dtype=np.float32)
                }

    def _init_render(self):
        if self.screen is None:
            if self.render_mode == "hidden":
                # Kein pygame, kein SDL, kein Windows-Handle-Problem.
                # Reines cv2/numpy-Rendering fuer alle Hidden-Mode-Subprozesse.
                track_bgr = cv2.imread(self.track_image_path)
                self.screen_height, self.screen_width = track_bgr.shape[:2]
                self.track_bg = track_bgr                          # unveraenderlicher Hintergrund
                self.screen = np.zeros_like(track_bgr)             # Arbeitspuffer (BGR)

                car_raw = cv2.imread(self.car_image_path, cv2.IMREAD_UNCHANGED)
                car_scaled = cv2.resize(car_raw, (23, 9), interpolation=cv2.INTER_AREA)
                self.car_img_np = cv2.rotate(car_scaled, cv2.ROTATE_180)
            else:
                pygame.init()
                self.track_img = pygame.image.load(self.track_image_path)
                self.screen_width, self.screen_height = self.track_img.get_size()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Carrera RL")
                car_surface = pygame.image.load(self.car_image_path)
                self.rotated_car_initial_state = pygame.transform.rotate(car_surface, 180)
                self.car_img = pygame.transform.scale(self.rotated_car_initial_state, (23, 9))

    def render(self):
        if self.screen is None:
            return

        x_m, y_m, _, theta, _ = self.state
        pixel_x = int(x_m * self.pixels_per_meter)
        pixel_y = int(y_m * self.pixels_per_meter)

        if self.render_mode == "hidden":
            np.copyto(self.screen, self.track_bg)

            angle_deg = float(180 - np.degrees(theta))
            car_h, car_w = self.car_img_np.shape[:2]
            n_ch = self.car_img_np.shape[2]

            # Quadratischer Canvas gross genug fuer jede Rotation (Diagonale + Puffer)
            diag = int(np.ceil(np.sqrt(car_h ** 2 + car_w ** 2))) + 2
            canvas = np.zeros((diag, diag, n_ch), dtype=np.uint8)
            y_off = (diag - car_h) // 2
            x_off = (diag - car_w) // 2
            canvas[y_off:y_off + car_h, x_off:x_off + car_w] = self.car_img_np

            M = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), angle_deg, 1.0)
            rotated = cv2.warpAffine(canvas, M, (diag, diag),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            x1 = pixel_x - diag // 2
            y1 = pixel_y - diag // 2
            sx1 = max(0, x1);  sy1 = max(0, y1)
            sx2 = min(self.screen_width, x1 + diag)
            sy2 = min(self.screen_height, y1 + diag)
            cx1 = sx1 - x1;  cy1 = sy1 - y1
            cx2 = cx1 + (sx2 - sx1);  cy2 = cy1 + (sy2 - sy1)

            if sx2 > sx1 and sy2 > sy1:
                patch = rotated[cy1:cy2, cx1:cx2]
                if n_ch == 4:   # BGRA mit Alpha-Kanal
                    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
                    bg = self.screen[sy1:sy2, sx1:sx2].astype(np.float32)
                    fg = patch[:, :, :3].astype(np.float32)
                    self.screen[sy1:sy2, sx1:sx2] = (bg * (1 - alpha) + fg * alpha).astype(np.uint8)
                else:           # BGR ohne Alpha
                    self.screen[sy1:sy2, sx1:sx2] = patch
        else:
            self.screen.blit(self.track_img, (0, 0))
            pygame_rotation = 180 - np.degrees(theta)
            rotated_car = pygame.transform.rotate(self.car_img, pygame_rotation)
            rect = rotated_car.get_rect(center=(pixel_x, pixel_y))
            self.screen.blit(rotated_car, rect.topleft)
            pygame.display.flip()

    def close(self):
        if self.screen is not None:
            if self.render_mode != "hidden":
                pygame.quit()
            self.screen = None
            self.isopen = False
