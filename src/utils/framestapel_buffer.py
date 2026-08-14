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
Nachbarindizes zusammensetzen.

Was das kostet
--------------
Gemessen bei der echten Konfiguration (batch_size=512, GPU, 600 Updates):

    Standard-Buffer     49.1 ms/Update    2849.6 MB
    dieser Buffer       54.9 ms/Update     949.9 MB
                        +11.8 %           Faktor 3.00 weniger

Vorsicht beim Faktor 3: er gilt gegen einen Standard-Buffer mit
`optimize_memory_usage=False`, der `observations` UND `next_observations`
anlegt. Die Zelle lief vorher aber mit `optimize_memory_usage=True`, und
das legt `next_observations` gar nicht erst an. Gegen den TATSAECHLICHEN
Vorzustand betraegt die Ersparnis nur Faktor 1.5:

    vorher  optimize_memory_usage=True, gestapelt   3 * B * W * H
    jetzt   Einzelbilder, obs + next_obs            2 * B * W * H

    166x100, 500k:   23.2 GB  ->  15.5 GB

Halbieren liesse sich das noch, indem next_observations nur fuer die
Episodenenden abgelegt und sonst aus observations[i+1] abgeleitet wird -
genau der Trick von optimize_memory_usage. Nicht umgesetzt, weil er die
Behandlung der Endzustaende verkompliziert.

Die Zeit ist also NICHT umsonst. Der Aufschlag steckt allein im Ziehen
(8.6 -> 13.9 ms je Batch), weil der Stapel dort erst entstehen muss.

Wichtig für eigene Messungen: bei kleinem batch_size verschwindet der
Unterschied im Rauschen. Eine frühere Messung mit batch_size=32 ergab
12.9 gegen 12.8 s und war damit irreführend.

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
    def _ordnen(self, gestapelt: np.ndarray) -> np.ndarray:
        """(N, n_stack, *Einzelbild) -> die Form, die VecFrameStack liefert."""
        n = len(gestapelt)
        if self._achse == 0:
            # (N, n, C, H, W) -> (N, n*C, H, W), reine Umdeutung ohne Kopie
            return gestapelt.reshape((n,) + self._voller_raum.shape)
        # (N, n, H, W, C) -> (N, H, W, n*C)
        return np.moveaxis(gestapelt, 1, -2).reshape((n,) + self._voller_raum.shape)

    def _stapel_paar(self, batch_inds: np.ndarray, env_inds: np.ndarray):
        """Baut Beobachtungs- UND Folgestapel mit je einem Zugriff.

        Beide überlappen sich um n_stack-1 Bilder:

            obs   = [ f_{n-1} ... f_1 , f_0 ]        f_z = observations[i-z]
            next  = [ f_{n-2} ... f_0 , g   ]        g   = next_observations[i]

        Statt die Bilder einzeln zu holen und zusammenzusetzen, wird EIN
        Indexfeld der Form (N, n_stack) aufgebaut und in einem Zugriff
        ausgelesen. Das Ergebnis liegt bereits in der richtigen Reihenfolge im
        Speicher, die Umformung zum Stapel ist danach kostenlos. Der
        Folgestapel entsteht aus demselben Ergebnis, um eine Stelle versetzt.

        Gemessen bei batch_size=512: 9.5 ms statt 22 ms für den stückweisen
        Aufbau, praktisch gleichauf mit SB3s Standard-Buffer (8.6 ms).

        Episodengrenzen: liegt zwischen zwei Bildern ein Episodenende, wird ab
        dort mit Nullen aufgefüllt — genauso wie VecFrameStack nach einem
        reset().
        """
        n, N = self.n_stack, len(batch_inds)

        # Platz k im Stapel (0 = ältestes) zeigt auf batch_inds - (n-1-k).
        zurueck = np.arange(n - 1, -1, -1)
        idx = (batch_inds[:, None] - zurueck[None, :]) % self.buffer_size

        # Gültigkeit: Platz k ist nur erreichbar, wenn zwischen ihm und dem
        # aktuellen Bild kein Episodenende liegt. dones[i-z] == 1 heißt, der
        # Übergang bei i-z beendete die Episode; alles Ältere ist damit
        # ebenfalls ungültig, deshalb das kumulative Produkt.
        gueltig = np.ones((N, n), dtype=bool)
        if n > 1:
            zur = np.arange(1, n)
            d = self.dones[(batch_inds[:, None] - zur[None, :]) % self.buffer_size,
                           env_inds[:, None]] > 0
            kum = np.cumprod(~d, axis=1)                    # (N, n-1)
            gueltig[:, : n - 1] = kum[:, ::-1]              # Platz 0 = weiteste Sicht

        # Ein einziger Zugriff über das flache Feld: (N, n, *Einzelbild)
        flach = self.observations.reshape((-1,) + self.observations.shape[2:])
        bilder = flach[idx * self.n_envs + env_inds[:, None]]
        bilder[~gueltig] = 0

        # Folgestapel: dieselben Bilder um eine Stelle versetzt, oben die
        # gespeicherte Folgebeobachtung.
        folge = np.empty_like(bilder)
        folge[:, : n - 1] = bilder[:, 1:]
        folge[:, n - 1] = self.next_observations[batch_inds, env_inds]

        return self._ordnen(bilder), self._ordnen(folge)

    def _stapel_bauen(
        self, batch_inds: np.ndarray, env_inds: np.ndarray, folge: bool
    ) -> np.ndarray:
        """Einzelner Stapel — nur für Tests, das Training nutzt _stapel_paar."""
        obs, nxt = self._stapel_paar(batch_inds, env_inds)
        return nxt if folge else obs

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

        obs, next_obs = self._stapel_paar(batch_inds, env_inds)

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
