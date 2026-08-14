"""Replay-Buffer, der Einzelbilder speichert statt fertiger Framestapel.

Problem
-------
`VecFrameStack` umhüllt die Umgebung, deshalb sieht SB3s ReplayBuffer den
fertig gestapelten Zustand als *die* Beobachtung und legt ihn komplett ab.
Benachbarte Übergänge teilen sich aber n_stack-1 ihrer Bilder — jedes
Einzelbild liegt dadurch rund n_stack-mal im Speicher.

    3 Bilder à 100x166 uint8, 500.000 Übergänge  ->  23.2 GB
    dieselben Daten als Einzelbilder             ->   7.7 GB

Lösung
------
Beim Speichern nur das neueste Bild ablegen, den Stapel beim Ziehen aus den
Nachbarindizes zusammensetzen. Beim Ziehen wird dieselbe Datenmenge bewegt,
nur über n_stack Indizes statt einen — die Ersparnis kostet also keine
nennenswerte Rechenzeit.

So arbeiten die DQN-Implementierungen von DeepMind, Dopamine und rlpyt.
SB3 bringt es nicht mit.

Zwei Feinheiten
---------------
1. *Episodengrenzen.* `VecFrameStack` setzt den Stapel bei jedem reset() auf
   null und schiebt die neuen Bilder hinten an. Am Episodenanfang ist er also
   mit Nullen aufgefüllt. Genau das muss die Rekonstruktion nachbilden, sonst
   stapelt sie Bilder aus zwei verschiedenen Episoden zusammen.

2. *Der Umlauf des Rings.* Ist der Buffer voll, liegt bei `pos` der älteste
   Eintrag und bei `pos-1` der neueste. Wer von `pos` aus zurückgreift, holt
   Bilder, die buffer_size Schritte später aufgenommen wurden. Deshalb schließt
   `sample()` die n_stack-1 Indizes ab `pos` aus. SB3s eigene Auswahl tut das
   nur bei `optimize_memory_usage=True` und ist hier daher nicht ausreichend.

Verwendung
----------
    from src.utils.framestapel_buffer import FramestapelReplayBuffer

    model = SAC("CnnPolicy", stacked_env,
                replay_buffer_class=FramestapelReplayBuffer,
                replay_buffer_kwargs=dict(n_stack=3),
                ...)

Wichtig: `optimize_memory_usage` NICHT zusätzlich setzen. Dieser Buffer
speichert ohnehin nur einmal und leitet die Folgebeobachtung aus dem
Nachbarindex ab.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class FramestapelReplayBuffer(ReplayBuffer):
    """Speichert Einzelbilder, baut die Stapel beim Ziehen zusammen.

    :param n_stack: Anzahl der gestapelten Bilder, muss zu VecFrameStack passen.
    :param channels_order: "first" (C,H,W) oder "last" (H,W,C).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_stack: int = 3,
        channels_order: str = "first",
    ):
        if optimize_memory_usage:
            raise ValueError(
                "optimize_memory_usage wird von diesem Buffer nicht unterstützt "
                "und ist auch nicht nötig: er speichert ohnehin nur einmal."
            )
        if not isinstance(observation_space, spaces.Box):
            raise ValueError("Dieser Buffer erwartet einen Box-Beobachtungsraum.")

        self.n_stack = int(n_stack)
        self.channels_order = channels_order
        voll = observation_space.shape

        if channels_order == "first":
            if voll[0] % self.n_stack:
                raise ValueError(
                    f"Kanalzahl {voll[0]} ist nicht durch n_stack={n_stack} teilbar."
                )
            self.kanaele = voll[0] // self.n_stack
            einzel = (self.kanaele,) + tuple(voll[1:])
            self._achse = 0
        else:
            if voll[-1] % self.n_stack:
                raise ValueError(
                    f"Kanalzahl {voll[-1]} ist nicht durch n_stack={n_stack} teilbar."
                )
            self.kanaele = voll[-1] // self.n_stack
            einzel = tuple(voll[:-1]) + (self.kanaele,)
            self._achse = -1

        # Der Elternkonstruktor legt die Arrays anhand des Raums an. Wir geben
        # ihm den EINZELBILD-Raum, damit observations entsprechend klein wird.
        einzelraum = spaces.Box(
            low=observation_space.low.reshape(voll).take(
                range(self.kanaele), axis=self._achse
            ),
            high=observation_space.high.reshape(voll).take(
                range(self.kanaele), axis=self._achse
            ),
            shape=einzel,
            dtype=observation_space.dtype,
        )
        super().__init__(
            buffer_size,
            einzelraum,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=handle_timeout_termination,
        )

        # next_observations speichert nur das jeweils neueste Folgebild.
        # Der Rest des Folgestapels ist mit dem aktuellen Stapel identisch
        # (um eins verschoben) und wird beim Ziehen daraus gebaut.
        self._voller_raum = observation_space

    # ------------------------------------------------------------------ #
    # Speichern
    # ------------------------------------------------------------------ #
    def _neuestes(self, gestapelt: np.ndarray) -> np.ndarray:
        """Aus (n_envs, n_stack*C, H, W) das jeweils neueste Bild schneiden."""
        if self.channels_order == "first":
            return gestapelt[:, -self.kanaele:, ...]
        return gestapelt[..., -self.kanaele:]

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(
            self._neuestes(np.asarray(obs).reshape((self.n_envs,) + self._voller_raum.shape)),
            self._neuestes(np.asarray(next_obs).reshape((self.n_envs,) + self._voller_raum.shape)),
            action,
            reward,
            done,
            infos,
        )

    # ------------------------------------------------------------------ #
    # Ziehen
    # ------------------------------------------------------------------ #
    def _stapel_bauen(
        self, batch_inds: np.ndarray, env_inds: np.ndarray, folge: bool
    ) -> np.ndarray:
        """Setzt für jeden Index den Framestapel aus den Vorgängern zusammen.

        `folge=True` baut den Stapel der FOLGEbeobachtung: dessen neuestes Bild
        ist next_observations[i], die älteren sind observations[i], [i-1], ...

        Episodengrenzen: liegt zwischen zwei Bildern ein Episodenende, wird ab
        dort mit Nullen aufgefüllt — genauso wie VecFrameStack nach einem
        reset().
        """
        bilder = []                                   # von neu nach alt
        gueltig = np.ones(len(batch_inds), dtype=bool)

        if folge:
            # Neuestes Bild des Folgestapels ist die Folgebeobachtung selbst.
            bilder.append(self.next_observations[batch_inds, env_inds])
            # Darunter liegt die aktuelle Beobachtung - sie gehört immer zur
            # selben Episode wie der Übergang, hier ist keine Prüfung nötig.
            bilder.append(self.observations[batch_inds, env_inds])
        else:
            bilder.append(self.observations[batch_inds, env_inds])

        # Ab hier rückwärts durch die gespeicherten Bilder laufen.
        zurueck = 1
        while len(bilder) < self.n_stack:
            idx = (batch_inds - zurueck) % self.buffer_size
            # dones[idx] == 1 bedeutet: der Übergang bei idx beendete die
            # Episode, das Bild dort gehört also zur vorherigen. Ab dann ist
            # alles Ältere ebenfalls ungültig, deshalb kumulativ.
            gueltig &= ~(self.dones[idx, env_inds] > 0)
            bild = self.observations[idx, env_inds].copy()
            bild[~gueltig] = 0
            bilder.append(bild)
            zurueck += 1

        # bilder liegt von neu nach alt vor, VecFrameStack ordnet alt -> neu
        achse = -1 if self._achse == -1 else 1
        return np.concatenate(bilder[::-1], axis=achse)

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        """Zieht Indizes und laesst dabei die aus, deren Vorgaenger fehlen.

        Ist der Ring voll, zeigt `pos` auf den AELTESTEN Eintrag - sein
        Vorgaenger wurde bereits ueberschrieben und enthaelt jetzt das
        neueste Bild. Wer bei `pos` zurueckgreift, stapelt Bilder aus zwei
        weit auseinanderliegenden Episoden zusammen.

        Ungueltig sind genau pos ... pos+n_stack-2:
            pos     braucht pos-1, pos-2   -> beide ueberschrieben
            pos+1   braucht pos (gut), pos-1 -> eines ueberschrieben
            pos+2   braucht pos+1, pos      -> beide gueltig
        """
        if not self.full:
            # Noch nie umgelaufen: Index 0 greift auf die genullten Enden des
            # Arrays zurueck - das entspricht genau dem Reset-Zustand von
            # VecFrameStack und ist damit korrekt.
            return self._get_samples(
                np.random.randint(0, self.pos, size=batch_size), env=env
            )

        tot = self.n_stack - 1
        batch_inds = (
            np.random.randint(0, self.buffer_size - tot, size=batch_size)
            + self.pos + tot
        ) % self.buffer_size
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        env_inds = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        obs = self._stapel_bauen(batch_inds, env_inds, folge=False)
        next_obs = self._stapel_bauen(batch_inds, env_inds, folge=True)

        daten = (
            self._normalize_obs(obs, env),
            self.actions[batch_inds, env_inds, :],
            self._normalize_obs(next_obs, env),
            (self.dones[batch_inds, env_inds]
             * (1 - self.timeouts[batch_inds, env_inds])).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_inds].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, daten)))
