Projektziele

Phase 1 :: Python
-Ziel: Herausfinden welchen Ansatz man verfolgen will, Architektur/Konzept Input für RL.
--Testen: RL mit: nur Bild, Bildfolge, Bildvorverarbeitung - dann: Bild & ermittelte Werte, Position, eigene Output-Geschwindigkeit, Abstand zum Rand etc. 
--Teststrecke als Bild, darauf muss ein Auto. Vorwärtsbewegbar, träge, lenkbar (von v abhängig), mit wasd im Notebook steuerbar.
--Hinterlegte Kosten/Betrafungsfunktion damit man sehen kann wie gut eigene Fahrten sind/was bestraft wird.
--Fahrbahn 25cm breit
21mm Fahrbahnrand rechts und links
Autolänge 98mm 39mm breit. von hinten das vorderrad 71mm von hinten bis hinterrad 23mm

State of the Art RL- Wie geht man aktuell dem Fahrenlenen um, wie werden Zeitliche zusammenhänge gelernt?
Welche RL Netze gibt es und welche sind die besten. Wie sieht das Weltrekordnetz von Trackmania aus?
wie bringt man in der Industrie Robotern fahren/laufen bei?
ändert man während dem Training die betrafung ab, sobald grundlagen erlernt sind? erst fahren lernen, dann richtig dem Kursnachfahren?
gibt es dafür vortrainerte Netze?
Kann man bei einem RL Ansatz eine Runde vorfahren? wie zeigt man der KI was gewollt ist? sodass es nur als Optimierung zu sehen ist?
gibt es eine Arbeit an der sich orientiert werden kann?
Wie ist die Architektur von RL Netzen, gibt es ein "Netz" oder Unterschiede?

ToDo:
Quellen Dokument lesen in Googledrive

Vanishing Gradient in deep networks, only visual input - Autonomous reinforcement learning on raw visual
input data in a real world application Sascha Lange, Martin Riedmiller Arne Voigtl¨ander


Das sind krasse Ideen:




RL for an RC Car on a Racetrack with Third‑Person Camera: What Fits and What’s Cutting‑Edge?

For a vision‑based RC car on a fixed track, research points to two main families: model‑free deep RL (actor–critic) for direct control from pixels, and hybrid imitation+RL / model‑based RL for sample efficiency and safety.
Core RL Methods That Fit Your Setup
Model‑Free, End‑to‑End from Images

    On‑policy actor–critic (A3C, PPO)
        A3C learns steering, brake, and throttle from front camera images in a realistic rally game, and generalizes to unseen tracks
        1.
        PPO from raw pixels achieves professional‑level lap times in a racing simulator, with continuous steering/throttle actions
        2.
    Off‑policy actor–critic (DDPG/TD3 and variants)
        DDPG with a vision encoder in a deep imitation RL (DIRL) setup successfully learns self‑driving policies in TORCS 3.
        TD3 outperforms DQN for FSAE‑style racing on a Turtlebot‑based platform and transfers from sim to real racetrack
        4.

These map closely to an RC car task: continuous control, racetrack geometry, and high‑frequency decisions from image streams.
Value‑Based DRL

    DQN and variants
        DQN (with suitable reward shaping and ε‑decay) drives a car around a 2D CarRacing track, outperforming PPO and DDPG in that setting
        5.
        Deep Q‑learning also controls cars in simple 3D game tracks from visual inputs 6.

DQN is attractive for simplicity but is less natural for fine, continuous control at race pace than actor–critic methods.

Methods on the Edge of Knowledge (Racing‑Focused)

    Imitative + Model‑Based RL for RC Cars
        Deep Imitative RL (DIRL) unifies imitation learning + model‑based RL: learn from human demonstrations, then safely self‑improve using an offline learned world model. Validated in sim and on a real 1/20‑scale RC car from vision, outperforming pure IL and pure RL in both sample efficiency and lap performance
        7.
    Champion‑Level Vision‑Only Agents (Gran Turismo series)
        Super‑human agents in Gran Turismo use deep RL with course‑progress rewards to beat 50,000+ human players and built‑in AIs 9.
        Newer work achieves champion‑level racing using only ego‑centric camera and onboard sensors via an asymmetric actor–critic: actor sees only local vision, critic has global info during training
        10
        11.
    Uncertainty‑Aware and Model‑Based RL
        Ensemble, uncertainty‑aware world models with adaptive truncation improve both efficiency and performance over model‑free baselines in end‑to‑end autonomous driving 12.
        Deductive RL (DeRL) adds a learned environment model (“deduction reasoner”) for predicting future trajectories, improving safety and success in urban driving navigation 13.

These frontier methods suggest that for an RC racetrack with third‑person camera, a vision encoder + off‑policy actor–critic, optionally combined with imitation pretraining and a learned world model / asymmetric critic, aligns well with state‑of‑the‑art research while remaining grounded in setups already proven on RC‑scale platforms.
