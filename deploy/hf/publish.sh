#!/usr/bin/env bash
# Assemble the Space payload (backend only: no caches, no secrets) and upload it.
# Usage: HF_TOKEN=hf_... ./deploy/hf/publish.sh <hf-username> <space-name>
set -euo pipefail

USER_NAME="${1:?hf username}"
SPACE="${2:?space name}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(mktemp -d)/space"
mkdir -p "$STAGE"

rsync -a \
  --exclude '__pycache__' --exclude '.ruff_cache' --exclude '.pytest_cache' \
  --exclude '*.pyc' --exclude '.env' --exclude '*.db' --exclude '*.sqlite3' \
  --exclude '.cache' --exclude 'data/generated' --exclude 'reports/**' \
  "$ROOT/backend/" "$STAGE/"

cp "$ROOT/deploy/hf/Dockerfile" "$STAGE/Dockerfile"
cp "$ROOT/deploy/hf/README.md" "$STAGE/README.md"

python3 -m pip install --quiet "huggingface_hub[cli]>=0.34"

python3 - "$USER_NAME" "$SPACE" <<'PY'
import sys
from huggingface_hub import HfApi

api = HfApi()
user, space = sys.argv[1], sys.argv[2]
repo_id = f"{user}/{space}"
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
print("repo ready:", repo_id)
PY

python3 - "$USER_NAME" "$SPACE" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi

user, space, stage = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi()
api.upload_folder(
    folder_path=stage,
    repo_id=f"{user}/{space}",
    repo_type="space",
    commit_message="VoxSentinel AI backend",
)
print("uploaded")
PY

echo "Space: https://huggingface.co/spaces/$USER_NAME/$SPACE"
echo "API:   https://$USER_NAME-$SPACE.hf.space"
