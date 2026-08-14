import cv2
import numpy as np
import pygame

class VirtualCamera:
    def __init__(self, crop_size=(150, 150), target_size=(84, 84),
                 global_size=(166, 100)):
        self.crop_width, self.crop_height = crop_size
        self.target_width, self.target_height = target_size
        # Groesse der Vogelperspektive. War frueher hier UND im
        # Beobachtungsraum der Umgebung hartkodiert - liefen die beiden
        # auseinander, brach das Training mit einem Formfehler ab. Jetzt gibt
        # die Umgebung den Wert vor und uebergibt ihn hierher.
        self.target_width_global, self.target_height_global = global_size
        self.levels = 4 # Anzahl der Graustufen, damit es nicht 255 Grautöne gibt

    def get_car_centric_observation(self, screen, cx, cy, theta):
        """
        cx, cy: Position des Autos in Pixeln
        theta: Ausrichtung des Autos in Radiant
        screen: numpy-BGR-Array (hidden mode) oder pygame.Surface (human mode)
        """
        # 1. Screen in BGR-Numpy-Array umwandeln
        if isinstance(screen, np.ndarray):
            view = screen.copy()  # bereits BGR (H,W,3)
        else:
            view = pygame.surfarray.array3d(screen)
            view = np.transpose(view, (1, 0, 2))  # (X,Y,C) -> (Y,X,C)
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
        
        # 7. Reduktion der Graustufen
        # Teilt den Bereich 0-255 in 16 Blöcke ein
        quantized_view = np.floor_divide(resized_view, 256 // self.levels) * (256 // self.levels)
        

        # 8. Channel-Dimension für Stable Baselines3 hinzufügen (uint8)
        final_obs = np.expand_dims(quantized_view, axis=0).astype(np.uint8)

        return final_obs, cropped_view
    
    def get_global_observation(self, screen):
        """
        Nimmt das gesamte Bild auf, wandelt es in Graustufen um
        und skaliert es auf die Zielgröße herunter.
        screen: numpy-BGR-Array (hidden mode) oder pygame.Surface (human mode)
        """
        # 1. Screen in Numpy-Array umwandeln
        if isinstance(screen, np.ndarray):
            view = screen
            gray_view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        else:
            view = pygame.surfarray.array3d(screen)
            view = np.transpose(view, (1, 0, 2))  # Pygame (X,Y,C) -> OpenCV (Y,X,C)
            gray_view = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)

        # 3. Skalierung auf 84x84 (INTER_AREA ist der beste Algorithmus zum Verkleinern)
        resized_view = cv2.resize(gray_view, (self.target_width_global, self.target_height_global), interpolation=cv2.INTER_AREA)
        #resized_view = cv2.resize(gray_view, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)

        # 4. Channel-Dimension für Stable Baselines3 hinzufügen (Shape: 1, 84, 84)
        final_obs = np.expand_dims(resized_view, axis=0).astype(np.uint8)

        # 5. Reduktion der Graustufen
        # Teilt den Bereich 0-255 in 16 Blöcke ein
        quantized_view = np.floor_divide(resized_view, 256 // self.levels) * (256 // self.levels)
        
        final_obs = np.expand_dims(quantized_view.astype(np.uint8), axis=0)

        return final_obs, resized_view