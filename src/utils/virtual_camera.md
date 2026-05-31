## Virtuelle Kamera einbinden

# Oben bei den Imports:
from src.utils.virtual_camera import VirtualCamera
import cv2

# Vor der Schleife:
camera = VirtualCamera(crop_size=(150, 150), target_size=(84, 84))

# ... in der Schleife, NACHDEM env.render() aufgerufen wurde! ...
env.render()

if env.screen is not None:
    x_m, y_m = env.state[0], env.state[1]
    car_x_px = x_m * env.pixels_per_meter
    car_y_px = y_m * env.pixels_per_meter
    
    # Kamera aufrufen
    rl_obs, debug_crop = camera.get_car_centric_observation(env.screen, car_x_px, car_y_px)
    
    # Den originalen (farbigen) Ausschnitt über OpenCV in einem Extra-Fenster anzeigen!
    cv2.imshow("Virtuelle Kamera - Auto Tracker", debug_crop)
    cv2.waitKey(1) # OpenCV Event Loop am Laufen halten