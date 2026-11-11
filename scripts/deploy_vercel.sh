#!/usr/bin/env bash
# Deploy VoxSentinel to Vercel (production) and verify the speech pipeline.
#
# Usage:
#   scripts/deploy_vercel.sh                 # uses a Vercel CLI login or $VERCEL_TOKEN
#   VERCEL_TOKEN=xxx scripts/deploy_vercel.sh
#
# What it does, in order:
#   1. links/pulls the existing project (voxsentinel-ai) so settings are not lost
#   2. makes sure the Vach speech credential exists in every Vercel environment
#   3. deploys to production
#   4. proves the deployment works by checking /api/capabilities and one live
#      transcription request, instead of assuming it did
#
# The credential is read from $API_KEY (or $VOXSENTINEL_VACH_API_KEY) and is
# never echoed. Nothing in this script prints it.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

TOKEN_ARGS=()
if [[ -n "${VERCEL_TOKEN:-}" ]]; then
  TOKEN_ARGS=(--token "$VERCEL_TOKEN")
  echo "• using VERCEL_TOKEN from the environment"
else
  echo "• using the Vercel CLI login on this machine (run 'npx vercel login' first)"
fi

VACH_KEY="${API_KEY:-${VOXSENTINEL_VACH_API_KEY:-}}"
if [[ -z "$VACH_KEY" ]]; then
  echo "  warning: no API_KEY in the environment, so transcription will deploy as" >&2
  echo "  'unavailable'. Set API_KEY to the Vach (Gnani) key to enable it." >&2
fi

# A stale .vercel/ can point at a project the current token cannot reach (the
# old voxsentinel-ai project lives under a different team). Link by name
# instead, creating the project if it does not exist yet.
PROJECT_NAME="${VERCEL_PROJECT:-voxsentinel}"
echo "• linking project: $PROJECT_NAME"
rm -rf .vercel
npx --yes vercel@latest link --yes --project "$PROJECT_NAME" "${TOKEN_ARGS[@]}" >/dev/null \
  || npx --yes vercel@latest link --yes "${TOKEN_ARGS[@]}" >/dev/null

# --- push the speech credential into every environment ----------------------
# `vercel env add` overwrites an existing value, so re-running is safe.
for target in production preview development; do
  if [[ -n "$VACH_KEY" ]]; then
    # `env add` refuses to replace an existing value, so remove first.
    npx --yes vercel@latest env rm API_KEY "$target" -y "${TOKEN_ARGS[@]}" >/dev/null 2>&1 || true
    printf '%s' "$VACH_KEY" | npx --yes vercel@latest env add API_KEY "$target" "${TOKEN_ARGS[@]}" >/dev/null 2>&1 \
      && echo "  API_KEY -> $target" \
      || echo "  API_KEY -> $target FAILED (set it in the Vercel dashboard)"
  fi
done

# Vercel caps a request body at 4.5 MB. The app reads this to refuse oversized
# uploads with a clear message instead of an opaque platform 413.
if [[ -z "${SKIP_BODY_LIMIT:-}" ]]; then
  npx --yes vercel@latest env rm VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES production -y "${TOKEN_ARGS[@]}" >/dev/null 2>&1 || true
  printf '4500000' | npx --yes vercel@latest env add VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES production "${TOKEN_ARGS[@]}" >/dev/null 2>&1 \
    && echo "  VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES -> production" \
    || echo "  body-limit var skipped"
fi

echo "• deploying to production"
DEPLOY_LOG="$(mktemp)"
npx --yes vercel@latest deploy --prod "${TOKEN_ARGS[@]}" 2>&1 | tee "$DEPLOY_LOG"

URL="$(grep -Eo 'https://[a-zA-Z0-9._-]+\.vercel\.app' "$DEPLOY_LOG" | tail -1 || true)"
if [[ -z "$URL" ]]; then
  echo "  could not read the deployment URL from the output above" >&2
  exit 1
fi

# --- verify, do not assume --------------------------------------------------
echo "• verifying $URL"
sleep 5
for attempt in 1 2 3 4 5; do
  if curl -fsS --max-time 20 "$URL/api/capabilities" -o /tmp/vox_caps.json 2>/dev/null; then
    break
  fi
  echo "  waiting for the deployment to answer (attempt $attempt/5)"
  sleep 10
done

python3 - <<'PY' || true
import json
try:
    d = json.load(open("/tmp/vox_caps.json"))
except Exception:
    raise SystemExit("  capabilities endpoint did not return JSON")
tr = (d.get("models") or {}).get("transcription") or {}
engine = next((e for e in tr.get("engines", []) if e.get("engine") == "gnani_asr"), None)
print("  transcription engine:", "gnani_asr", "available =", bool(engine and engine.get("available")))
if not (engine and engine.get("available")):
    print("  -> speech-to-text is OFF in this deployment; set API_KEY in Vercel and redeploy")
print("  speaker verification ready:", d.get("speaker_verifier_ready"))
print("  detector ready:", (d.get("models") or {}).get("detector", {}).get("ready"))
PY

echo
echo "Deployed: $URL"
echo "Next: open the URL, upload a clip, and confirm a transcript with a risk category."
