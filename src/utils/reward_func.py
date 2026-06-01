class RewardCalculator:
    def __init__(self):
        self.crash_penalty = -300.0
        self.lap_bonus = 200.0
        self.best_lap_bonus = 300.0
        self.slow_lap_penalty = -300.0
        self.wrong_way_penalty = -30.0
        self.standstill_penalty = -15.0
        self.w_speed = 6.0
        self.w_smooth = 0.5 
        self.w_integral = 0.0
        self.w_reverse = 10.0 # Neuer, kontrollierbarer Faktor für Rückwärts
        self.best_lap_frames = float('inf') # Für zukünftige Rundenzeit-Belohnung

    def calculate(self, v, is_crashing, sf_crossed, is_new_lap, correct_direction, steer_delta, steer_delta_history_sum, lap_frames):
        reward = 0.0
        terminated = False

        # 1. Todesbedingung (Crash)
        if is_crashing:
            # Wir geben die Strafe und beenden sofort. PPO lernt das am schnellsten.
            return self.crash_penalty, True 

        # 2. Richtungs- und Geschwindigkeits-Reward
        if not correct_direction:
            reward += self.wrong_way_penalty
            if v > 0.1: # Strafe für Gas in falsche Richtung
                reward -= (v * 5.0) 
        else:
            # Richtig herum unterwegs
            if -0.1 <= v <= 0.1: 
                reward += self.standstill_penalty
            elif v > 0.1: 
                # Progressiver Reward: Schneller = exponentiell besser
                reward += (v**2) * self.w_speed*2.0
            elif v < -0.1: 
                # Harte, aber nicht explodierende Strafe fürs Rückwärtsfahren
                reward += v * self.w_reverse 
        
        # 3. Runden-Bonus
        if sf_crossed and is_new_lap:
            reward += self.lap_bonus

        # 4. Smoothness (Strafe)
        reward -= abs(steer_delta) * self.w_smooth
        # 5. Oszillations-Schutz (Integral)
        if steer_delta_history_sum > 2.0:
            # Jetzt bestrafen wir wirklich nur den Überschuss!
            reward -= (steer_delta_history_sum - 2.0) * self.w_integral
        # 6. Bestzeiten Bonus und Slow Lap Penality
        # Hat er die Bestzeit geschlagen?
            if lap_frames < self.best_lap_frames:
                # Differenz berechnen (Wie viele Frames war er schneller?)
                frame_improvement = self.best_lap_frames - lap_frames
                
                # Wenn es nicht die allererste Runde überhaupt ist, gibt es den fetten Bonus
                if self.best_lap_frames != float('inf'):
                    reward += frame_improvement * 5.0 # Z.B. 5 Punkte pro gespartem Frame!
                
                # Neue Bestzeit speichern
                self.best_lap_frames = lap_frames
            else:
                # Er war langsamer als seine Bestzeit -> Leichte Strafe
                frame_delay = lap_frames - self.best_lap_frames
                reward -= frame_delay * 2.0

        return float(reward), terminated