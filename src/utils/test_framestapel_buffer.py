"""Prüft, ob der Einzelbild-Buffer dieselben Stapel liefert wie VecFrameStack.

Der Vergleich läuft gegen SB3s Standard-ReplayBuffer, der die von
VecFrameStack erzeugten Stapel unverändert ablegt. Beide bekommen dieselbe
Folge von Übergängen; anschließend werden für JEDEN gültigen Index die
gezogenen Stapel verglichen.

Der kritische Fall sind Episodengrenzen — dort füllt VecFrameStack mit Nullen
auf, und genau das muss die Rekonstruktion nachbilden.

Aufruf:
    python -m src.utils.test_framestapel_buffer
"""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer

from src.utils.framestapel_buffer import FramestapelReplayBuffer

H, W = 6, 8
N_STACK = 3
GROESSE = 200


def stapel_wie_vecframestack(bilder, dones):
    """Bildet nach, was VecFrameStack aus einer Bildfolge macht.

    Beim Reset wird der Stapel genullt, danach werden Bilder hinten
    angeschoben. `dones[t]` bedeutet: Übergang t hat die Episode beendet,
    Bild t+1 gehört zur nächsten.
    """
    stapel = np.zeros((N_STACK, H, W), dtype=np.uint8)
    aus = []
    for t, bild in enumerate(bilder):
        stapel = np.roll(stapel, shift=-1, axis=0)
        stapel[-1] = bild[0]
        aus.append(stapel.copy())
        if t < len(dones) and dones[t]:
            stapel = np.zeros_like(stapel)
    return aus


def main():
    rng = np.random.default_rng(0)

    voller_raum = spaces.Box(0, 255, (N_STACK, H, W), dtype=np.uint8)
    aktionsraum = spaces.Box(-1, 1, (2,), dtype=np.float32)

    standard = ReplayBuffer(GROESSE, voller_raum, aktionsraum,
                            n_envs=1, handle_timeout_termination=True)
    neu = FramestapelReplayBuffer(GROESSE, voller_raum, aktionsraum, n_envs=1,
                                  n_stack=N_STACK, channels_order="first",
                                  handle_timeout_termination=True)

    print(f"Speicher: Standard {standard.observations.nbytes + standard.next_observations.nbytes:>8} B"
          f" | neu {neu.observations.nbytes + neu.next_observations.nbytes:>8} B")

    # Bildfolge mit Episoden unterschiedlicher Länge erzeugen
    N = 150
    bilder = [rng.integers(0, 255, (1, H, W), dtype=np.uint8) for _ in range(N + 1)]
    dones = np.zeros(N, dtype=bool)
    t = 0
    while t < N:
        t += int(rng.integers(2, 9))       # kurze Episoden, damit Grenzen haeufig sind
        if t < N:
            dones[t] = True

    gestapelt = stapel_wie_vecframestack(bilder, dones)

    for t in range(N):
        obs = gestapelt[t][None]
        if dones[t]:
            # Bei einem Episodenende ist next_obs die TERMINALE Beobachtung,
            # nicht der Reset. VecFrameStack liefert sie vor dem Zuruecksetzen.
            folge_stapel = np.roll(gestapelt[t], -1, axis=0)
            folge_stapel[-1] = bilder[t + 1][0]
            next_obs = folge_stapel[None]
        else:
            next_obs = gestapelt[t + 1][None]

        a = rng.random((1, 2)).astype(np.float32)
        r = rng.random(1).astype(np.float32)
        d = np.array([dones[t]])
        infos = [{}]
        standard.add(obs, next_obs, a, r, d, infos)
        neu.add(obs, next_obs, a, r, d, infos)

    # Jeden gueltigen Index vergleichen
    idx = np.arange(min(N, GROESSE))
    env_idx = np.zeros(len(idx), dtype=int)

    soll_obs = standard.observations[idx, env_idx]
    ist_obs = neu._stapel_bauen(idx, env_idx, folge=False)
    soll_next = standard.next_observations[idx, env_idx]
    ist_next = neu._stapel_bauen(idx, env_idx, folge=True)

    ok_obs = np.array_equal(soll_obs, ist_obs)
    ok_next = np.array_equal(soll_next, ist_next)

    print(f"\nBeobachtungsstapel  identisch: {ok_obs}")
    print(f"Folgestapel         identisch: {ok_next}")

    if not ok_obs:
        schlecht = np.where((soll_obs != ist_obs).any(axis=(1, 2, 3)))[0]
        print(f"   abweichende Indizes: {schlecht[:15]} (von {len(idx)})")
        print(f"   davon direkt nach einem Episodenende: "
              f"{sum(1 for i in schlecht if i > 0 and dones[i-1])}")
    if not ok_next:
        schlecht = np.where((soll_next != ist_next).any(axis=(1, 2, 3)))[0]
        print(f"   abweichende Indizes: {schlecht[:15]} (von {len(idx)})")
        print(f"   davon an einem Episodenende: "
              f"{sum(1 for i in schlecht if dones[i])}")

    print(f"\nEpisodengrenzen im Test: {int(dones.sum())}")
    print("BESTANDEN" if (ok_obs and ok_next) else "FEHLGESCHLAGEN")
    return ok_obs and ok_next


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
