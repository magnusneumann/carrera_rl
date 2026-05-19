import pygame
import numpy as np
from src.envs.carrera_2d_env import Carrera2DEnv 

# --- 1. Environment starten ---
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
    
    # Steuerung abfragen
    gas = 0.5 if keys[pygame.K_w] else 0.0
    brake = 1.0 if keys[pygame.K_s] else 0.0
    
    steer = 0.0
    if keys[pygame.K_a]: steer = -1.0
    if keys[pygame.K_d]: steer = 1.0

    action = np.array([gas, brake, steer])
    
    # --- Environment Schritt ---
    # Die 2D-Env rechnet JETZT ALLES (Physik, Trägheit, Integral, Crash) intern!
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Wenn das Env sagt "Kollision" oder "Punkte-Limit", setzen wir zurück
    if terminated or truncated:
        print(f"🏁 Episode beendet! Letzter Frame-Reward: {reward:.2f}")
        obs, _ = env.reset()

    # --- 2. Rendering & HUD ---
    env.render()
    
    # Wir holen uns die Ecken DIREKT aus dem Environment, um die Hitbox zu zeichnen
    if hasattr(env, 'car_corners'):
        # Grün wenn kein Crash, rot wenn Crash
        color = (255, 0, 0) if info['is_crashing'] else (0, 255, 0)
        pygame.draw.polygon(env.screen, color, env.car_corners, 1)
    
    # HUD Texte DIREKT aus dem 'info'-Paket der 2D-Env lesen!
    border_ok = not info['is_crashing']
    correct_direction = info['correct_direction']
    sf_crossed = info['sf_crossed']
    
    text_data = [
        (f"Border OK: {border_ok}", (0, 255, 0) if border_ok else (255, 0, 0)),
        (f"Start/Finish: {sf_crossed}", (255, 255, 0) if sf_crossed else (200, 200, 200)),
        (f"Direction OK: {correct_direction}", (0, 255, 0) if correct_direction else (255, 0, 0)),
        (f"Frame Reward: {reward:.4f}", (255, 255, 255)),
        (f"Actual Steer: {info.get('actual_steer', steer):.2f}", (0, 191, 255)) # Zeigt die Trägheit!
    ]
    
    # HUD auf den Screen blitten
    y_offset = 10
    for text_str, color in text_data:
        text_surface = font.render(text_str, True, color)
        bg_rect = text_surface.get_rect(topleft=(10, y_offset))
        pygame.draw.rect(env.screen, (0, 0, 0), bg_rect)
        env.screen.blit(text_surface, (10, y_offset))
        y_offset += 25
        
    pygame.display.flip()
    clock.tick(30)

env.close()