from pathlib import Path


def windows_to_wsl_path(windows_path: Path):
    windows_path = str(windows_path)
    wsl_path = windows_path.replace("\\", "/").replace(":", "").replace(" ", "\\ ")
    wsl_path = "/mnt/" + wsl_path
    wsl_path = wsl_path.replace("/C/", "/c/")
    return Path(wsl_path)
