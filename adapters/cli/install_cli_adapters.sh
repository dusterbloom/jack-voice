#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${HOME}/.local/bin"

mkdir -p "${BIN_DIR}"

for name in \
  jv-doctor \
  jv-chat \
  jv-stt \
  jv-tts \
  codex-voice \
  claude-voice \
  opencode-voice \
  droid-voice \
  pi-voice; do
  ln -sfn "${ROOT_DIR}/adapters/bin/${name}" "${BIN_DIR}/${name}"
done

echo "Installed CLI adapter commands to ${BIN_DIR}:"
echo "  jv-doctor, jv-chat, jv-stt, jv-tts"
echo "  codex-voice, claude-voice, opencode-voice, droid-voice, pi-voice"
echo
echo "If ${BIN_DIR} is not in PATH, add:"
echo "  export PATH=\"${BIN_DIR}:\$PATH\""
