class RewardCalculator:
    def __init__(self):
        # Hier kannst du später Gewichtungen einstellen
        self.crash_penalty = -100.0
        self.lap_bonus = 20.0
        self.speed_multiplier = 1.33
        self.speed_reward = 5.0
        self.wrong_way_penalty = -25.0

    def calculate(self, v, is_crashing, sf_crossed, is_new_lap, correct_direction):
        """
        Nimmt die aktuellen Zustände des Environments und berechnet den Reward.
        """
        reward = 0.0
        terminated = False

        # 1. Todesbedingung (Crash)
        if is_crashing:
            reward = self.crash_penalty
            terminated = False #True #grade auskommentiert, mal probieren ob man ohne abbrechen lernen kann
            return reward, terminated # Sofort abbrechen

        # 2. Richtungs- und Geschwindigkeits-Reward
        if not correct_direction:
            # KI steht/fährt in die falsche Richtung
            reward += self.wrong_way_penalty
            
            # Strafe, wenn sie auch noch Gas in die falsche Richtung gibt!
            if v > 0.1:
                reward -= v * 2 * self.speed_multiplier * self.speed_reward
                
        else:
            # KI steht richtig herum (Normales Verhalten)
            if -0.1 <= v <= 0.1: # Auto steht (fast)
                reward += -6.0
            elif v > 0.11: # Auto fährt vorwärts
                reward += v * self.speed_multiplier * self.speed_reward
            elif v < -0.1: # Auto fährt rückwärts
                reward += v * 2 * self.speed_multiplier * self.speed_reward # v ist negativ = Minuspunkte
        # 3. Runden-Bonus
        if sf_crossed and is_new_lap:
            reward += self.lap_bonus

        return reward, terminated