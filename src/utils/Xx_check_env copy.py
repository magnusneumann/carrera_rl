import pygame
import numpy as np
import math
from shapely.geometry import Polygon, LineString, Point
from src.envs.carrera_2d_env import Carrera2DEnv 

# --- 1. Shapely Geometrien initialisieren ---
# 1. Rohe Splines laden
outer_points = np.load('data/outer_raw_spline.npy')
inner_points = np.load('data/inner_raw_spline.npy')

# 2. Zu Shapely-Linien machen und DIE GRENZEN AUFBLÄHEN (um 3 Pixel)
outer_wall = LineString(outer_points).buffer(3.0, cap_style=1, join_style=1)
inner_wall = LineString(inner_points).buffer(3.0, cap_style=1, join_style=1)

# Start-/Ziellinie als LineString
sf_line = LineString([(462, 55), (468, 114)])

# Hitbox-Größe (Auto-Grafik war 23x9 Pixel. Hitbox etwas kleiner: 20x7)
HITBOX_L = 20.0
HITBOX_W = 7.0

# --- 2. Environment starten ---
env = Carrera2DEnv('data/strecke.png', 'data/carrera_car.png')
obs, _ = env.reset()

pygame.init()
font = pygame.font.SysFont('Arial', 20, bold=True)
clock = pygame.time.Clock()

running = True
print("🚗 WASD zum Fahren. ESC zum Beenden.")

while running:
    # --- Pygame Events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    keys = pygame.key.get_pressed()
    
    # Drosselung (Gas = 0.5) wie früher besprochen für besseres Testen
    gas = 0.5 if keys[pygame.K_w] else 0.0
    brake = 1.0 if keys[pygame.K_s] else 0.0
    
    steer = 0.0
    if keys[pygame.K_a]: steer = -1.0
    if keys[pygame.K_d]: steer = 1.0

    action = np.array([gas, brake, steer])
    
    # --- Environment Schritt ---
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Koordinaten aus der Physik abrufen
    x_m, y_m, v, theta, _ = env.state
    
    # In Pixel umrechnen für Shapely!
    cx = x_m * env.pixels_per_meter
    cy = y_m * env.pixels_per_meter
    
    # --- 3. Shapely Hitbox berechnen (Rotierte Ecken) ---
    # Vektoren für die Eckpunkte basierend auf dem physikalischen Winkel (theta)
    dx_f = (HITBOX_L / 2) * math.cos(theta)
    dy_f = (HITBOX_L / 2) * math.sin(theta)
    dx_s = (HITBOX_W / 2) * -math.sin(theta) # Orthogonaler Vektor
    dy_s = (HITBOX_W / 2) * math.cos(theta)
    
    car_corners = [
        (cx + dx_f + dx_s, cy + dy_f + dy_s), # Vorne Links
        (cx + dx_f - dx_s, cy + dy_f - dy_s), # Vorne Rechts
        (cx - dx_f - dx_s, cy - dy_f - dy_s), # Hinten Rechts
        (cx - dx_f + dx_s, cy - dy_f + dy_s)  # Hinten Links
    ]
    car_poly = Polygon(car_corners)
    
    # --- 4. Auswertungen (Check Env) ---
    
    # A. Track Limits Check (KORRIGIERT: Jetzt hier unten, nachdem car_poly existiert)
    crash = car_poly.intersects(outer_wall) or car_poly.intersects(inner_wall)
    border_ok = not crash
    
    # B. Start/Finish Check
    sf_crossed = car_poly.intersects(sf_line)
    
    # C. Direction Check (KORRIGIERT: Nutzt jetzt outer_points als Referenz)
    # Finde den am nächsten gelegenen Punkt auf der Außenlinie
    dists = np.linalg.norm(outer_points - [cx, cy], axis=1)
    nearest_idx = np.argmin(dists)
    
    # Nimm einen Punkt ca. 5 Indices weiter vorne, um den Vektor zu bilden
    next_idx = (nearest_idx + 5) % len(outer_points)
    p1 = outer_points[nearest_idx]
    p2 = outer_points[next_idx]
    
    # Winkel der Strecke an dieser Stelle berechnen
    track_vec = p2 - p1
    track_angle = math.atan2(track_vec[1], track_vec[0])
    
    # Differenz zum Auto-Winkel (Wertebereich -Pi bis Pi)
    angle_diff = (theta - track_angle + math.pi) % (2 * math.pi) - math.pi
    correct_direction = abs(angle_diff) < (math.pi / 2) # Weniger als 90 Grad Abweichung
    
    # --- 5. Rendering & HUD ---
    env.render()
    
    # Hitbox zeichnen (Optional, super zum Debuggen!)
    pygame.draw.polygon(env.screen, (0, 255, 0) if border_ok else (255, 0, 0), car_corners, 1)
    
    # HUD Texte generieren
    text_data = [
        (f"Border OK: {border_ok}", (0, 255, 0) if border_ok else (255, 0, 0)),
        (f"Start/Finish: {sf_crossed}", (255, 255, 0) if sf_crossed else (200, 200, 200)),
        (f"Direction OK: {correct_direction}", (0, 255, 0) if correct_direction else (255, 0, 0)),
        (f"Nearest Idx: {nearest_idx} / {len(outer_points)}", (255, 255, 255))
    ]
    
    # HUD auf den Screen blitten
    y_offset = 10
    for text_str, color in text_data:
        text_surface = font.render(text_str, True, color)
        # Kleiner schwarzer Hintergrund für bessere Lesbarkeit
        bg_rect = text_surface.get_rect(topleft=(10, y_offset))
        pygame.draw.rect(env.screen, (0, 0, 0), bg_rect)
        env.screen.blit(text_surface, (10, y_offset))
        y_offset += 25
        
    pygame.display.flip()
    clock.tick(30)

env.close()