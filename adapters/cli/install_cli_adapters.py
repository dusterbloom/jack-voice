#!/usr/bin/env python3
from __future__ import annotations

import os
import pathlib
import stat
import textwrap

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
ADAPTER_BIN_DIR = ROOT_DIR / "adapters" / "bin"

POSIX_COMMANDS = [
    "jv-doctor",
    "jv-chat",
    "jv-stt",
    "jv-tts",
    "codex-voice",
    "claude-voice",
    "opencode-voice",
    "droid-voice",
    "pi-voice",
]

WINDOWS_COMMANDS = {
    "jv-doctor": ["doctor"],
    "jv-chat": ["chat"],
    "jv-stt": ["stt-mic"],
    "jv-tts": ["tts"],
    "codex-voice": ["ask", "--target", "codex"],
    "claude-voice": ["ask", "--target", "claude"],
    "opencode-voice": ["ask", "--target", "opencode"],
    "droid-voice": ["ask", "--target", "droid"],
    "pi-voice": ["ask", "--target", "pi"],
}


def main() -> int:
    install_dir = default_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        install_windows_wrappers(install_dir)
    else:
        install_posix_symlinks(install_dir)

    print(f"Installed CLI adapter commands to {install_dir}:")
    print("  jv-doctor, jv-chat, jv-stt, jv-tts")
    print("  codex-voice, claude-voice, opencode-voice, droid-voice, pi-voice")
    print()
    print("If this directory is not on PATH, add:")
    if os.name == "nt":
        print(f"  set PATH={install_dir};%PATH%")
    else:
        print(f'  export PATH="{install_dir}:$PATH"')

    return 0


def default_install_dir() -> pathlib.Path:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return pathlib.Path(local_app_data) / "jack-voice" / "bin"
        return pathlib.Path.home() / "AppData" / "Local" / "jack-voice" / "bin"
    return pathlib.Path.home() / ".local" / "bin"


def install_posix_symlinks(install_dir: pathlib.Path) -> None:
    for name in POSIX_COMMANDS:
        src = ADAPTER_BIN_DIR / name
        dst = install_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def install_windows_wrappers(install_dir: pathlib.Path) -> None:
    cli_path = ROOT_DIR / "adapters" / "cli_voice.py"
    cli_win = str(cli_path)
    for name, args in WINDOWS_COMMANDS.items():
        bat = install_dir / f"{name}.cmd"
        arg_string = " ".join(args)
        body = textwrap.dedent(
            f"""\
            @echo off
            where py >nul 2>nul
            if %ERRORLEVEL%==0 (
              py -3 "{cli_win}" {arg_string} %*
            ) else (
              python "{cli_win}" {arg_string} %*
            )
            """
        )
        bat.write_text(body, encoding="utf-8")
        bat.chmod(bat.stat().st_mode | stat.S_IEXEC)


if __name__ == "__main__":
    raise SystemExit(main())
