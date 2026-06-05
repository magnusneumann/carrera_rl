import cv2
import numpy as np
import pygame

class VirtualCamera:
    def __init__(self, crop_size=(150, 150), target_size=(84, 84)):
        self.crop_width, self.crop_height = crop_size
        self.target_width, self.target_height = target_size

    def get_car_centric_observation(self, pygame_screen, cx, cy, theta):
        """
        cx, cy: Position des Autos in Pixeln
        theta: Ausrichtung des Autos in Radiant
        """
        # 1. Pygame-Surface in Numpy-Array (RGB) umwandeln
        view = pygame.surfarray.array3d(pygame_screen)
        view = np.transpose(view, (1, 0, 2)) # (X,Y,C) -> (Y,X,C) für OpenCV
        view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

        # 2. Rotationswinkel berechnen (Ziel: Auto zeigt starr nach OBEN)
        # In unserem Pygame-Setup bedeutet 0 Grad "Rechts". 
        # Um das Auto nach "Oben" (-90 Grad) blicken zu lassen, drehen wir die Welt zurück.
        # NEU (KORREKT)
        theta_deg = np.degrees(theta)
        
        # Da Pygame das Auto mit (180 - theta) zeichnet,
        # hebt (+ theta - 90) die Rotation exakt auf, sodass es immer nach oben zeigt.
        rotation_angle = theta_deg - 90

        # 3. Rotationsmatrix für das Zentrum des Autos erstellen
        M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)

        # 4. HIGH-PERFORMANCE TRICK:
        # Wir verschieben die Matrix so, dass der Ziel-Ausschnitt exakt 
        # im Zentrum der neuen (kleinen) Bildmatrix landet.
        M[0, 2] += (self.crop_width / 2) - cx
        M[1, 2] += (self.crop_height / 2) - cy

        # 5. Schneidet und rotiert NUR den kleinen 150x150 Bereich
        cropped_view = cv2.warpAffine(
            view, M, (self.crop_width, self.crop_height), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0) # Alles außerhalb der Strecke wird schwarz
        )

        # 6. Graustufen & Skalierung auf 84x84
        gray_view = cv2.cvtColor(cropped_view, cv2.COLOR_BGR2GRAY)
        resized_view = cv2.resize(gray_view, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        
        # 7. Channel-Dimension für Stable Baselines3 hinzufügen (uint8)
        final_obs = np.expand_dims(resized_view, axis=0).astype(np.uint8)

        return final_obs, cropped_view