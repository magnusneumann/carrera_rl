import numpy as np
import math
from shapely.geometry import LineString, Point

class SensorSuite:
    def __init__(self, outer_wall, inner_wall, max_ray_length=400.0):
        # Wir speichern die Wände als ein MultiPolygon oder eine Liste
        self.walls = [outer_wall, inner_wall]
        self.max_ray_length = max_ray_length
        # Sensor-Winkel relativ zur Auto-Ausrichtung (z.B. -30°, 0°, +30°)
        self.angles_deg = [-30.0, 30.0] 

    def get_lidar_observation(self, cx, cy, theta, v):
        """
        Berechnet die Distanzen der Raycasts und gibt die Observation zurück.
        """
        distances = []
        ray_lines = [] # Für die Visualisierung im Pygame-Fenster

        for angle_deg in self.angles_deg:
            # Winkel in Bogenmaß umrechnen und zum Auto-Winkel addieren
            ray_angle = theta + math.radians(angle_deg)
            
            # Endpunkt des Strahls berechnen
            end_x = cx + self.max_ray_length * math.cos(ray_angle)
            end_y = cy + self.max_ray_length * math.sin(ray_angle)
            ray = LineString([(cx, cy), (end_x, end_y)])
            
            min_dist = self.max_ray_length
            
            # Schnittpunkte mit Innen- und Außenwand prüfen
            for wall in self.walls:
                # BUG FIX 1: Wir schneiden mit der Kante (boundary) der Wand!
                intersection = ray.intersection(wall.boundary)
                
                if not intersection.is_empty:
                    # BUG FIX 2: Robuste Extraktion der Koordinaten, egal was Shapely zurückgibt
                    if intersection.geom_type == 'Point':
                        pts = [intersection.coords[0]]
                    elif intersection.geom_type == 'MultiPoint':
                        pts = [pt.coords[0] for pt in intersection.geoms]
                    elif intersection.geom_type == 'LineString':
                        pts = list(intersection.coords)
                    elif intersection.geom_type == 'MultiLineString':
                        pts = [coord for line in intersection.geoms for coord in line.coords]
                    else:
                        pts = []
                        
                    for p in pts:
                        # Schnelle Distanzberechnung mit Pythagoras
                        d = math.hypot(p[0] - cx, p[1] - cy)
                        if d < min_dist:
                            min_dist = d
            
            # Normalisieren (0.0 = Wand berührt, 1.0 = Freie Fahrt)
            norm_dist = min_dist / self.max_ray_length
            distances.append(norm_dist)
            
            # Fürs Zeichnen speichern (Sichtbar im Pygame)
            actual_end_x = cx + min_dist * math.cos(ray_angle)
            actual_end_y = cy + min_dist * math.sin(ray_angle)
            ray_lines.append(((cx, cy), (actual_end_x, actual_end_y)))

        # Observation Array: [v, dist_left, dist_right]
        obs = np.array([v] + distances, dtype=np.float32)
        return obs, ray_lines