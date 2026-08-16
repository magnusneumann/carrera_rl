"""Framestapel mit zeitlichem Abstand: t, t-s, t-2s statt t, t-1, t-2.

Warum
-----
Der Stapel existiert nur, damit der Agent seine Geschwindigkeit ablesen kann.
Das geht nur, wenn sich zwischen den gestapelten Bildern etwas messbar
verschiebt. Bei 250x150 und den tatsaechlich gefahrenen Tempi sind die
Unterschiede, auf die es ankommt, kleiner als ein Pixel:

    1.20 gegen 1.41 m/s   0.50 px      (normale Fahrt gegen Anbremspunkt)
    1.04 gegen 1.11 m/s   0.17 px      (Kurve geht gerade noch / nicht mehr)

Mit Abstand s vervielfacht sich die Verschiebung um s, ohne dass mehr Bilder
gespeichert oder verarbeitet werden. Bei s=4 werden daraus 1.99 und 0.66 px.

Siehe rl_erkenntnisse.md, Abschnitt 13d.

Zwei Teile muessen zusammenpassen
---------------------------------
1. `VecFrameStackGespreizt` liefert der Policy den gespreizten Stapel.
2. `FramestapelReplayBuffer(stride=s)` setzt beim Ziehen denselben Stapel
   zusammen.

Laufen die beiden auseinander, trainiert der Agent auf etwas anderem, als er
spaeter sieht - ein Fehler, der still bleibt. `pruefe_stride()` vergleicht sie.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


class VecFrameStackGespreizt(VecEnvWrapper):
    """Wie VecFrameStack, aber mit Abstand zwischen den gestapelten Bildern.

    :param venv: die zu umhuellende VecEnv, Beobachtung (C, H, W) oder (H, W, C)
    :param n_stack: Anzahl gestapelter Bilder
    :param stride: Abstand in Frames. 1 entspricht dem gewoehnlichen
        VecFrameStack.
    :param channels_order: "first" (C, H, W) oder "last" (H, W, C)
    """

    def __init__(self, venv: VecEnv, n_stack: int, stride: int = 1,
                 channels_order: str = "first"):
        self.n_stack = int(n_stack)
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("stride muss mindestens 1 sein")
        self.channels_order = channels_order

        raum = venv.observation_space
        if not isinstance(raum, spaces.Box):
            raise ValueError("Es wird ein Box-Beobachtungsraum erwartet.")

        self._achse = 0 if channels_order == "first" else -1
        einzel = raum.shape
        self.kanaele = einzel[self._achse]

        # So viele Bilder muessen vorgehalten werden, damit t-(n-1)*s erreichbar ist
        self.tiefe = (self.n_stack - 1) * self.stride + 1

        if self._achse == 0:
            voll = (self.kanaele * self.n_stack,) + einzel[1:]
        else:
            voll = einzel[:-1] + (self.kanaele * self.n_stack,)

        neuer_raum = spaces.Box(
            low=np.repeat(raum.low, self.n_stack, axis=self._achse),
            high=np.repeat(raum.high, self.n_stack, axis=self._achse),
            shape=voll, dtype=raum.dtype)

        super().__init__(venv, observation_space=neuer_raum)

        self._verlauf = np.zeros((venv.num_envs, self.tiefe) + einzel, dtype=raum.dtype)
        # Positionen im Verlauf, die in den Stapel wandern - von alt nach neu
        self._plaetze = [self.tiefe - 1 - z * self.stride
                         for z in range(self.n_stack - 1, -1, -1)]

    # ------------------------------------------------------------------ #
    def _stapel(self) -> np.ndarray:
        gewaehlt = self._verlauf[:, self._plaetze]        # (n_envs, n_stack, *einzel)
        n = gewaehlt.shape[0]
        if self._achse == 0:
            return gewaehlt.reshape((n,) + self.observation_space.shape)
        return np.moveaxis(gewaehlt, 1, -2).reshape((n,) + self.observation_space.shape)

    def _schiebe(self, obs: np.ndarray, envs=None) -> None:
        if envs is None:
            self._verlauf[:, :-1] = self._verlauf[:, 1:]
            self._verlauf[:, -1] = obs
        else:
            self._verlauf[envs, :-1] = self._verlauf[envs, 1:]
            self._verlauf[envs, -1] = obs[envs]

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self._verlauf[:] = 0          # wie VecFrameStack: mit Nullen beginnen
        self._verlauf[:, -1] = obs
        return self._stapel()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._schiebe(obs)
        for i, fertig in enumerate(dones):
            if fertig:
                # Die terminale Beobachtung gehoert noch zur alten Episode und
                # steckt in infos; obs enthaelt hier bereits den Reset.
                self._verlauf[i] = 0
                self._verlauf[i, -1] = obs[i]
        return self._stapel(), rewards, dones, infos

    def close(self) -> None:
        self.venv.close()


def pruefe_stride(n_stack: int = 3, stride: int = 4, laenge: int = 400,
                  groesse: int = 300, h: int = 6, w: int = 8) -> bool:
    """Liefert der Buffer dieselben Stapel wie der Wrapper?

    Baut eine Bildfolge mit Episodengrenzen, bildet einmal das Verhalten des
    Wrappers nach und vergleicht mit dem, was der Buffer rekonstruiert.
    """
    from stable_baselines3.common.buffers import ReplayBuffer
    from src.utils.framestapel_buffer import FramestapelReplayBuffer

    rng = np.random.default_rng(0)
    voll = spaces.Box(0, 255, (n_stack, h, w), dtype=np.uint8)
    akt = spaces.Box(-1, 1, (2,), dtype=np.float32)

    std = ReplayBuffer(groesse, voll, akt, n_envs=1, handle_timeout_termination=True)
    neu = FramestapelReplayBuffer(groesse, voll, akt, n_envs=1, n_stack=n_stack,
                                  stride=stride, channels_order="first",
                                  handle_timeout_termination=True)

    bilder = [rng.integers(0, 255, (h, w), dtype=np.uint8) for _ in range(laenge + 1)]
    dones = np.zeros(laenge, dtype=bool)
    t = 0
    while t < laenge:
        t += int(rng.integers(20, 60))
        if t < laenge:
            dones[t] = True

    tiefe = (n_stack - 1) * stride + 1
    plaetze = [tiefe - 1 - z * stride for z in range(n_stack - 1, -1, -1)]
    verlauf = np.zeros((tiefe, h, w), dtype=np.uint8)
    verlauf[-1] = bilder[0]

    for i in range(laenge):
        obs = verlauf[plaetze][None]
        folge = np.concatenate([verlauf[1:], bilder[i + 1][None]], axis=0)[plaetze][None]
        std.add(obs, folge, np.zeros((1, 2), np.float32), np.zeros(1, np.float32),
                np.array([dones[i]]), [{}])
        neu.add(obs, folge, np.zeros((1, 2), np.float32), np.zeros(1, np.float32),
                np.array([dones[i]]), [{}])
        verlauf[:-1] = verlauf[1:]
        verlauf[-1] = bilder[i + 1]
        if dones[i]:
            verlauf[:] = 0
            verlauf[-1] = bilder[i + 1]

    # Ist der Ring uebergelaufen, sind die (n_stack-1)*stride Indizes ab pos
    # nicht rekonstruierbar - ihre Vorgaenger wurden ueberschrieben. Genau die
    # schliesst sample() aus, also gehoeren sie auch hier nicht in den Vergleich.
    alle = np.arange(min(laenge, groesse))
    if neu.full:
        kaputt = {(neu.pos + k) % groesse for k in range((n_stack - 1) * stride)}
        idx = np.array([i for i in alle if i not in kaputt])
    else:
        kaputt, idx = set(), alle
    env_idx = np.zeros(len(idx), dtype=int)
    ok_obs = np.array_equal(std.observations[idx, env_idx],
                            neu._stapel_bauen(idx, env_idx, folge=False))
    ok_next = np.array_equal(std.next_observations[idx, env_idx],
                             neu._stapel_bauen(idx, env_idx, folge=True))
    print(f"stride={stride}, n_stack={n_stack}, {int(dones.sum())} Episodengrenzen, "
          f"Ring {'uebergelaufen' if neu.full else 'nicht voll'}, "
          f"{len(idx)} von {len(alle)} Indizes vergleichbar")
    print(f"   Beobachtungsstapel identisch: {ok_obs}")
    print(f"   Folgestapel        identisch: {ok_next}")
    return ok_obs and ok_next


if __name__ == "__main__":
    import sys
    alles = all(pruefe_stride(stride=s) for s in (1, 2, 4, 8))
    print("BESTANDEN" if alles else "FEHLGESCHLAGEN")
    sys.exit(0 if alles else 1)
