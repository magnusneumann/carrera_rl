"""Angaben, die einen Trainingslauf nachvollziehbar machen.

Gedacht für die train_config.json. Der wichtigste Teil ist der Git-Stand:
mit Commit-Hash lässt sich der exakte Code eines Laufs wiederherstellen,
unabhängig davon, ob eine Einstellung im Log auftaucht oder nicht. Das
entschärft die ganze Fehlerklasse, bei der ein Wert im Code steht und
getrennt davon von Hand ins Log geschrieben wird.

Beispiel:

    from src.utils.laufinfo import laufinfo
    train_config = {..., **laufinfo(model_sac)}
"""
import subprocess
import sys


def git_stand():
    """Commit, Branch und ob unversionierte Änderungen vorliegen.

    'dirty' ist entscheidend: bei nicht committeten Änderungen beschreibt der
    Hash den Lauf nur unvollständig.
    """
    def lauf(*args):
        try:
            return subprocess.run(("git",) + args, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    commit = lauf("rev-parse", "HEAD")
    if not commit:
        return {"verfuegbar": False}
    status = lauf("status", "--porcelain")
    return {
        "verfuegbar": True,
        "commit": commit,
        "commit_kurz": commit[:9],
        "branch": lauf("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "geaenderte_dateien": len(status.splitlines()) if status else 0,
    }


def modell_info(model):
    """Parameterzahl und Netzstruktur des Agenten."""
    try:
        policy = model.policy
        gesamt = sum(p.numel() for p in policy.parameters())
        info = {"parameter_gesamt": int(gesamt),
                "policy_klasse": type(policy).__name__}
        for teil in ("actor", "critic"):
            if hasattr(policy, teil):
                info[f"parameter_{teil}"] = int(
                    sum(p.numel() for p in getattr(policy, teil).parameters()))
        fe = getattr(policy, "features_extractor", None)
        if fe is not None:
            info["features_extractor"] = type(fe).__name__
            info["features_dim"] = int(getattr(fe, "features_dim", 0))
        return info
    except Exception as e:
        return {"fehler": str(e)}


def hardware_info():
    """Womit gerechnet wurde. Ohne das sind Laufzeitangaben nicht einzuordnen."""
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_verfuegbar"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_speicher_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    except Exception:
        pass
    try:
        import psutil
        info["cpu_kerne"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    try:
        import stable_baselines3
        info["stable_baselines3"] = stable_baselines3.__version__
    except Exception:
        pass
    return info


def laufinfo(model=None):
    """Alles zusammen, fertig zum Einfügen in train_config."""
    d = {"git": git_stand(), "hardware": hardware_info()}
    if model is not None:
        d["modell"] = modell_info(model)
    return d


if __name__ == "__main__":
    import json
    print(json.dumps(laufinfo(), indent=2, ensure_ascii=False))
