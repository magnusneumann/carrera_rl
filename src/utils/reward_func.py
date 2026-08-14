class RewardCalculator:
    """Belohnungsfunktion.

    version kennzeichnet die FORMEL, nicht nur die Gewichte. Ohne diese
    Kennung lassen sich Läufe nicht unterscheiden, bei denen die Gewichte
    gleich blieben, aber die Berechnung sich geändert hat.

    v3_rundenzeit
    -------------
    1) Gefälle statt Klippe im langsamen Bereich (aus v2 übernommen).
       Grund war das neue Fahrzeugmodell measured_v2: mit rund 0.25 m/s²
       Anfahrbeschleunigung entstand zwischen 0.1 und 0.5 m/s eine tote
       Zone ohne nennenswerte Belohnung oder Strafe - genau dort landet
       das Auto beim Bremsen in Kurven, und zurück auf Tempo dauert es
       Sekunden.

    2) Die Strafe für langsamere Runden ist entfallen. Vorher wurde jede
       Runde gegen die bisherige Bestzeit der Episode gerechnet, und zwar
       jedes Mal aufs Neue. Eine einzelne schnelle Runde hob damit die
       Messlatte und kostete über die restliche Episode mehr, als sie
       einbrachte: sieben Runden mit einer schnellen darunter ergaben
       -250, sieben gleichmäßig langsame dagegen 0. Schnellfahren war
       unterm Strich bestraft.

    3) Alle Faktoren sind jetzt benannte Parameter. best_lap_bonus und
       slow_lap_penalty standen zwar in der Konfiguration, wurden aber
       nie verwendet - die echten Werte waren hartcodiert.
    """

    def __init__(self):
        self.version = "v3_rundenzeit"

        # --- Ereignisse ---
        self.crash_penalty = -300.0
        self.lap_bonus = 50.0
        self.wrong_way_penalty = -30.0

        # --- laufende Bewertung pro Frame ---
        self.w_speed = 1.0
        self.w_reverse = 10.0   # derzeit unerreichbar, siehe calculate()

        # Gefälle im langsamen Bereich (ersetzt die frühere standstill_penalty)
        self.v_slow = 0.6       # m/s   darunter wird es teuer
        self.w_slow = 4.0       #       Strafe pro Frame bei v = 0

        # --- Lenkung ---
        self.w_smooth = 0.5     # bestraft jede einzelne Lenkänderung
        # Oszillationsschutz: bestraft die Summe der Lenkänderungen der
        # letzten 30 Frames, soweit sie osc_threshold übersteigt.
        #
        # ABGESCHALTET (w_integral = 0.0). Die Zerlegung eines Laufs zeigte,
        # dass die Schwelle 2.0 nicht Zappeln von normalem Lenken trennt,
        # sondern jedes Lenken besteuert: der gut fahrende Lidar-Experte zahlte
        # damit -625 über 2000 Frames, mehr als seine gesamten Rundenboni
        # (+600). Sein Mittelwert der Lenksumme liegt bei rund 5.9.
        # Verschärfend kommt hinzu, dass die Strafe absolut ist, der
        # Speed-Reward aber mit v² skaliert - unterhalb von etwa 0.54 m/s
        # kostet Lenken mehr als Fahren einbringt, was langsame Agenten
        # geradeaus in die Wand treibt.
        # Falls wieder aktiviert: Schwelle auf 6-8 setzen, nicht auf 2.
        self.osc_threshold = 2.0
        self.w_integral = 0.0

        # --- Rundenzeit ---
        # Bonus für das Unterbieten der eigenen Bestzeit innerhalb der Episode.
        # Die Boni teleskopieren: in Summe immer (erste Runde - beste Runde)
        # mal diesem Faktor, unabhängig davon, wie die Verbesserung verteilt war.
        self.w_lap_improvement = 5.0

        self.best_lap_frames = float('inf')   # beste Runde DIESER Episode

    def to_dict(self):
        exclude = {"best_lap_frames"}
        return {k: v for k, v in vars(self).items() if k not in exclude}

    def calculate(self, v, is_crashing, sf_crossed, is_new_lap, correct_direction,
                  steer_delta, steer_delta_history_sum, lap_frames):
        reward = 0.0
        terminated = False

        # 1. Todesbedingung (Crash)
        if is_crashing:
            # Wir geben die Strafe und beenden sofort. PPO lernt das am schnellsten.
            return self.crash_penalty, True

        # 2. Richtungs- und Geschwindigkeits-Reward
        if not correct_direction:
            reward += self.wrong_way_penalty
            if v > 0.1:   # Strafe für Gas in falsche Richtung
                reward -= (v * 5.0)
        else:
            if v < -0.1:
                # Harte, aber nicht explodierende Strafe fürs Rückwärtsfahren.
                # Mit longitudinal_model="measured_v2" derzeit unerreichbar
                # (v >= 0 wird physikalisch erzwungen); bleibt erhalten, weil
                # Rückwärtsfahren später wieder möglich sein soll.
                reward += v * self.w_reverse
            else:
                # Progressiver Reward: Schneller = exponentiell besser
                reward += (v ** 2) * self.w_speed * 2.0

                # Gefälle statt Klippe: bei v_slow kostet es nichts,
                # bei v = 0 volle w_slow, dazwischen linear.
                if v < self.v_slow:
                    reward -= self.w_slow * (self.v_slow - v) / self.v_slow

        # 3. Runden-Bonus
        if sf_crossed and is_new_lap and v > 0:
            reward += self.lap_bonus
        # Strafe für rückwärts über die Ziellinie fahren
        if sf_crossed and v < 0:
            reward -= self.lap_bonus

        # 4. Smoothness (Strafe pro Lenkänderung)
        reward -= abs(steer_delta) * self.w_smooth

        # 5. Oszillations-Schutz (Integral über die letzten 30 Frames)
        if steer_delta_history_sum > self.osc_threshold:
            reward -= (steer_delta_history_sum - self.osc_threshold) * self.w_integral

        # 6. Rundenzeit
        if sf_crossed and is_new_lap:
            if lap_frames < self.best_lap_frames:
                # Die erste Runde setzt nur die Referenz und gibt keinen Bonus,
                # sonst wäre die Verbesserung gegenüber "unendlich" unendlich.
                if self.best_lap_frames != float('inf'):
                    reward += (self.best_lap_frames - lap_frames) * self.w_lap_improvement
                self.best_lap_frames = lap_frames
            # Kein Gegenstück für langsamere Runden: eine Runde über der
            # Bestzeit kostet nichts extra. Dass Langsamfahren schlecht ist,
            # regelt bereits der Speed-Reward in jedem einzelnen Frame.

        return float(reward), terminated
