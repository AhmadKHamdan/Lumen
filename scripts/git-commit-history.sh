#!/usr/bin/env bash
# Replays the Sprint 1 build as ~10 logical commits so `git log` shows
# the order in which the system was constructed.
#
# Run from the repo root:
#   bash scripts/git-commit-history.sh
#
# Prerequisites: a fresh `git init` (no commits yet) and all Sprint 1 files
# present on disk. The script `git add`s in groups and commits each group.

set -euo pipefail

if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo "Error: not in a git repo. Run 'git init -b main' first."
  exit 1
fi

if git rev-parse HEAD > /dev/null 2>&1; then
  echo "Error: repo already has commits. This script expects a fresh repo."
  echo "Either delete existing history or just run 'git add . && git commit -m \"...\"' instead."
  exit 1
fi

commit() {
  local message="$1"
  shift
  git add "$@"
  git commit -m "$message"
}

# 1. Repo skeleton
commit "Initial repo structure: protocol contract, README, gitignore" \
  .gitignore README.md docs/protocol.md docs/Intro.pdf docs/Lumen_Implementation_Plan.pdf

# 2. Backend skeleton
commit "Backend: FastAPI skeleton with /health and /ws endpoints" \
  backend/requirements.txt backend/main.py backend/__init__.py

# 3. Backend infrastructure
commit "Backend: Session, message router, frame ingestion handler" \
  backend/api/

# 4. FSM
commit "Backend: 5-state FSM with transition rules and subscribers" \
  backend/fsm/

# 5. TTS service
commit "Voice: tts_service with gTTS and bounded LRU cache" \
  backend/services/tts_service.py

# 6. Speech service
commit "Voice: speech_service with Whisper STT and WebM/Opus decode" \
  backend/services/speech_service.py

# 7. Command parser
commit "Voice: command_parser with rapidfuzz and COCO noun list" \
  backend/services/command_parser.py backend/services/__init__.py

# 8. Task stubs
commit "Backend: stub entry hooks for object allocation and navigation" \
  backend/services/object_allocation.py backend/services/navigation.py

# 9. Tests
commit "Backend: FSM and command_parser unit tests" \
  backend/tests/

# 10. Frontend
commit "Frontend: HTML shell, WS client, media capture, audio queue, app coordinator" \
  frontend/

# 11. Scripts and final touches
commit "Scripts: git setup docs and helper scripts" \
  scripts/

echo
echo "Done. Run 'git log --oneline' to inspect."
