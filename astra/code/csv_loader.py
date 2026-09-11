import pandas as pd
from pathlib import Path
from typing import Union, Any, Optional
import matplotlib.pyplot as plt


def load_csv(
    file_path: Union[str, Path],
    **kwargs: Any
) -> pd.DataFrame:
    """
    Liest eine CSV-Datei ein, benennt die erste Spalte automatisch in "t_s" um
    und gibt sie als pandas DataFrame zurueck.

    Parameter:
    -----------
    file_path : str oder Path
        Pfad zur CSV-Datei (absolut oder relativ zum aufrufenden Skript/Notebook).
    **kwargs : beliebig
        Zusaetzliche Argumente, die an `pandas.read_csv()` weitergegeben werden
        (z. B. `sep=';'`, `decimal=','`, `dtype=str`, `encoding='utf-8'`, etc.).

    Rueckgabe:
    ---------
    pd.DataFrame
        Eingelesener DataFrame, wobei die erste Spalte als "t_s" vorliegt.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Die Datei '{file_path}' wurde nicht gefunden.")

    try:
        df = pd.read_csv(file_path, **kwargs)
        # Erste Spalte umbenennen zu "t_s"
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "t_s"})
        return df
    except Exception as e:
        raise RuntimeError(f"Fehler beim Einlesen von '{file_path}': {e}")
    


def plot_csv_data(df: pd.DataFrame, title: str = "Geschwindigkeitsprofile") -> None:
    """
    Erstellt zwei Subplots:

    1. Oben: v_ms (Geschwindigkeit) ueber t_s (Zeit)
    2. Unten: v_ms (Geschwindigkeit) ueber s_m (Weg)

    Parameter:
    -----------
    df : pd.DataFrame
        DataFrame mit den Spalten `t_s`, `s_m`, `v_ms`.
    title : str, optional
        Titel fuer die gesamte Figur. Standard: "Geschwindigkeitsprofile".
    """
    required_cols = {"t_s", "s_m", "v_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame fehlen erforderliche Spalten: {missing}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=False)

    # --- Oben: v vs. t ---
    ax1.plot(df["t_s"], df["v_ms"], color="tab:blue", label="v_ms (Geschwindigkeit)", linewidth=1.5)
    ax1.set_ylabel("Geschwindigkeit v [m/s]")
    ax1.set_title("Geschwindigkeit ueber Zeit")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # --- Unten: v vs. s ---
    ax2.plot(df["s_m"], df["v_ms"], color="tab:red", label="v_ms (Geschwindigkeit)", linewidth=1.5)
    ax2.set_xlabel("Weg s [m]")
    ax2.set_ylabel("Geschwindigkeit v [m/s]")
    ax2.set_title("Geschwindigkeit ueber Weg")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Platz fuer Ueberschrift lassen
    plt.show()