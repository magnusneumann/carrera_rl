import cv2
import numpy as np
from scipy.interpolate import splprep, splev

class TrackProcessor:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    def get_mask(self, color_type):
        if color_type == "blue": # Außenbegrenzung
            lower = np.array([94, 114, 49])
            upper = np.array([107, 255, 255])
            return cv2.inRange(self.hsv, lower, upper)
            
        elif color_type == "red": # Innenbegrenzung
            # Rot ist im HSV-Raum zweigeteilt (nahe 0 und nahe 180)
            lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
            lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
            mask1 = cv2.inRange(self.hsv, lower1, upper1)
            mask2 = cv2.inRange(self.hsv, lower2, upper2)
            return cv2.bitwise_or(mask1, mask2)
        
        return None

    def extract_track_logic(self, save_masks=True):
        blue_mask = self.get_mask("blue")
        red_mask = self.get_mask("red")
        
        # Morphologische Operationen zum Schließen kleiner Lücken
        kernel = np.ones((5,5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        if save_masks:
            import os
            # Stelle sicher, dass der data-Ordner existiert
            os.makedirs('data', exist_ok=True) 
            cv2.imwrite('data/mask_outer.png', blue_mask)
            cv2.imwrite('data/mask_inner.png', red_mask)
            # Optional: Kurzes Feedback in der Konsole
            print("Masken gespeichert: 'data/mask_outer.png' und 'data/mask_inner.png'")

        return blue_mask, red_mask


class TrackAnalyzer:
    def __init__(self, blue_mask, red_mask):
        self.blue_mask = blue_mask
        self.red_mask = red_mask

    def _process_mask_to_splines(self, mask, abstand_parameter):
        """Interne Hilfsfunktion: Wandelt eine Maske in zwei glatte Hüllkurven um."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []

        # 1. Schwerpunkte finden
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append([cX, cY])

        if len(centroids) < 3:
            return []

        centroids = np.array(centroids)

        # 2. Nearest Neighbor Sortierung
        unvisited = list(centroids)
        ordered_points = [unvisited.pop(0)]

        while len(unvisited) > 0:
            last_point = ordered_points[-1]
            distances = np.linalg.norm(unvisited - last_point, axis=1)
            nearest_idx = np.argmin(distances)
            ordered_points.append(unvisited.pop(nearest_idx))

        ordered_points.append(ordered_points[0]) # Kreis schließen
        ordered_points = np.array(ordered_points)

        # 3. Spline-Approximation
        x = ordered_points[:, 0]
        y = ordered_points[:, 1]
        tck, u = splprep([x, y], s=0, per=True) 
        u_new = np.linspace(0, 1, 1000)
        x_spline, y_spline = splev(u_new, tck)
        raw_spline_points = np.vstack((x_spline, y_spline)).T.astype(np.int32)
        

        # 4. Mittellinie zeichnen und aufblähen
        midline_img = np.zeros_like(mask)
        cv2.polylines(midline_img, [raw_spline_points], isClosed=True, color=255, thickness=1)

        kernel_size = 1 + (2 * abstand_parameter) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated_midline = cv2.dilate(midline_img, kernel, iterations=1)

        # 5. Exakte Hüllkurven extrahieren
        final_contours, _ = cv2.findContours(dilated_midline, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = sorted(final_contours, key=cv2.contourArea, reverse=True)[:2]
        
        return final_contours, raw_spline_points

    def get_all_track_limits(self, abstand_parameter=3):
        """
        Gibt die 4 geschlossenen Linien und ein kombiniertes Array aller Pixel zurück.
        """
        # Verarbeite blaue (außen) und rote (innen) Maske
        outer_contours, outer_raw_spline = self._process_mask_to_splines(self.blue_mask, abstand_parameter)
        inner_contours, inner_raw_spline = self._process_mask_to_splines(self.red_mask, abstand_parameter)
        # Alle 4 Konturen in einer Liste
        all_4_contours = outer_contours + inner_contours
        
        # Alle Pixel in ein einziges flaches (N, 2) Array zusammenführen (für die Physik-Engine)
        all_pixels = []
        for cnt in all_4_contours:
            all_pixels.extend(cnt.reshape(-1, 2))
            
        combined_pixels_array = np.array(all_pixels)

        np.save('data/outer_raw_spline.npy', outer_raw_spline)
        np.save('data/inner_raw_spline.npy', inner_raw_spline)
        
        return all_4_contours, combined_pixels_array, outer_raw_spline, inner_raw_spline

    @staticmethod
    def detect_finish_line(image_gray):
        laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, thresh = cv2.threshold(laplacian, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((10,10), np.uint8)
        finish_area = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return finish_area
    
    def calculate_centerline(self, outer_points, inner_points, smoothing=10.0):
        """
        Berechnet die Mittellinie mittels Nearest-Neighbor und Spline-Glättung.
        Setzt den Startpunkt (Index 0) exakt auf die Start-/Ziellinie.
        """
        from scipy.spatial import KDTree
        
        # 1. KDTree für die innere Kurve aufbauen (für extrem schnelle Nachbarsuche)
        inner_tree = KDTree(inner_points)
        midpoints = []
        
        # 2. Hüllkurve entlanggehen und halbe Strecke berechnen
        # (Wir gehen die äußere Kurve ab, da sie länger ist und eine höhere Auflösung hat)
        for p_out in outer_points:
            # Finde den Index des nächsten Nachbarn auf der inneren Kurve
            _, idx = inner_tree.query(p_out)
            p_in = inner_points[idx]
            
            # Halbe Strecke Punkt setzen
            midpoint = (p_out + p_in) / 2.0
            midpoints.append(midpoint)
            
        midpoints = np.array(midpoints)
        
        # 3. Alle neuen Punkte glätten (Der Fix gegen Zick-Zack)
        # Doppelte Punkte vor dem Spline-Fitting entfernen
        _, unique_indices = np.unique(midpoints, axis=0, return_index=True)
        midpoints = midpoints[np.sort(unique_indices)]
        
        x = midpoints[:, 0]
        y = midpoints[:, 1]
        
        # Spline fitten (per=True, da es ein Rundkurs ist)
        tck, u = splprep([x, y], s=smoothing, per=True)
        u_new = np.linspace(0, 1, 1000) # 1000 Punkte für eine hochauflösende Mittellinie
        x_spline, y_spline = splev(u_new, tck)
        
        centerline = np.vstack((x_spline, y_spline)).T
        
        # --- 4. Start-/Ziellinie als Nullpunkt (Index 0) setzen ---
        # Koordinaten deiner definierten Ziellinie
        start_finish_p1 = np.array([462, 55])
        start_finish_p2 = np.array([468, 114])
        
        # Mittelpunkt der Ziellinie
        start_finish_mid = (start_finish_p1 + start_finish_p2) / 2.0
        
        # Finde den Punkt auf der berechneten Mittellinie, der der echten Ziellinie am nächsten ist
        distances = np.linalg.norm(centerline - start_finish_mid, axis=1)
        start_idx = np.argmin(distances)
        
        # Rotiere das Array so, dass der Ziellinien-Punkt an Position 0 rutscht
        centerline_aligned = np.roll(centerline, -start_idx, axis=0)
        
        return centerline_aligned