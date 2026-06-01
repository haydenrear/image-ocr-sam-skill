from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

from .utils import ensure_parent


def take_screenshot(out_path: str | Path, window: str | None = None) -> dict[str, Any]:
    out = ensure_parent(out_path)
    system = platform.system().lower()
    tried: list[str] = []
    if system == "darwin":
        cmd = ["screencapture", "-x", str(out)]
        tried.append(" ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            return {"status": "ok", "path": str(out), "engine": "screencapture"}
    elif system == "linux":
        candidates = [
            ["gnome-screenshot", "-f", str(out)],
            ["import", "-window", "root", str(out)],
            ["scrot", str(out)],
        ]
        for cmd in candidates:
            tried.append(" ".join(cmd))
            try:
                rc = subprocess.run(cmd).returncode
            except FileNotFoundError:
                continue
            if rc == 0:
                return {"status": "ok", "path": str(out), "engine": cmd[0]}
    elif system == "windows":
        escaped_out = str(out).replace("'", "''")
        ps = [
            "powershell", "-NoProfile", "-Command",
            "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
            "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height; "
            "$g=[System.Drawing.Graphics]::FromImage($bmp); "
            "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size); "
            f"$bmp.Save('{escaped_out}')"
        ]
        tried.append("powershell screenshot")
        try:
            rc = subprocess.run(ps).returncode
            if rc == 0:
                return {"status": "ok", "path": str(out), "engine": "powershell"}
        except FileNotFoundError:
            pass
    return {"status": "unavailable", "path": str(out), "tried": tried, "reason": "no supported screenshot command succeeded"}
