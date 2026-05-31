import pygame
import numpy as np
import cv2

class VirtualCamera:
    """
    Eine virtuelle Kamera, die das Pygame-Fenster abgreift, das Auto zentriert
    und einen vorverarbeiteten Bildausschnitt (Graustufen, skaliert) für RL-Agenten liefert.
    """
    def __init__(self, crop_size=(150, 150), target_size=(84, 84)):
        """
        :param crop_size: Größe des Ausschnitts um das Auto im Originalbild (Breite, Höhe).
        :param target_size: Zielauflösung für den RL-Agenten nach der Skalierung (Breite, Höhe).
        """
        self.crop_width, self.crop_height = crop_size
        self.target_width, self.target_height = target_size

    def get_car_centric_observation(self, screen, car_x_px, car_y_px):
        """
        Extrahiert das Bild um das Auto herum und bereitet es vor.
        
        :param screen: Das Pygame-Display-Surface (env.screen).
        :param car_x_px: X-Koordinate des Autos in Pixeln.
        :param car_y_px: Y-Koordinate des Autos in Pixeln.
        :return: Ein Numpy-Array (target_size, 1 Kanal) mit Werten von 0.0 bis 1.0.
        """
        # 1. Das gesamte Pygame-Bild als Numpy-Array abgreifen
        # Pygame nutzt intern (Breite, Höhe, RGB). OpenCV nutzt (Höhe, Breite, BGR).
        view = pygame.surfarray.array3d(screen)
        
        # Achsen für OpenCV anpassen (Transponieren)
        view = np.transpose(view, (1, 0, 2))
        
        # OpenCV nutzt BGR statt RGB, also Farbräume tauschen
        view_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        
        # 2. Den Ausschnitt (Crop) berechnen
        img_h, img_w = view_bgr.shape[:2]
        car_x, car_y = int(car_x_px), int(car_y_px)
        
        # Grenzen für den Crop definieren
        half_w = self.crop_width // 2
        half_h = self.crop_height // 2
        
        x_min = car_x - half_w
        x_max = car_x + half_w
        y_min = car_y - half_h
        y_max = car_y + half_h
        
        # 3. Padding (Auffüllen), falls das Auto zu nah am Rand ist
        # Damit das Output-Array immer exakt crop_size hat!
        pad_top = max(0, -y_min)
        pad_bottom = max(0, y_max - img_h)
        pad_left = max(0, -x_min)
        pad_right = max(0, x_max - img_w)
        
        # Sichere Grenzen für den Array-Schnitt
        safe_y_min = max(0, y_min)
        safe_y_max = min(img_h, y_max)
        safe_x_min = max(0, x_min)
        safe_x_max = min(img_w, x_max)
        
        # Crop durchführen
        cropped_view = view_bgr[safe_y_min:safe_y_max, safe_x_min:safe_x_max]
        
        # Fehlende Ränder mit Schwarz auffüllen (falls am Rand der Welt)
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            cropped_view = cv2.copyMakeBorder(
                cropped_view, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            
        # 4. In Graustufen umwandeln
        gray_view = cv2.cvtColor(cropped_view, cv2.COLOR_BGR2GRAY)
        
        # 5. Auf Zielauflösung skalieren (z.B. 84x84)
        resized_view = cv2.resize(gray_view, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        
        # 6. Normalisieren (Werte zwischen 0.0 und 1.0) - Wichtig für neuronale Netze!
        normalized_view = resized_view.astype(np.float32) / 255.0
        
        # Um eine Dimension erweitern, damit es das Format (Kanäle, Höhe, Breite) hat
        # Das erwartet Stable Baselines3 oft für Bilder (1, 84, 84)
        final_obs = np.expand_dims(normalized_view, axis=0)
        
        return final_obs, cropped_view # Wir geben cropped_view fürs Debuggen mit zurück