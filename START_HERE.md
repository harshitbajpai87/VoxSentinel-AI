# VoxSentinel repaired project

This is a repair release of your existing project, not a replacement application.
The original model artifacts, UI, integrations and database are preserved.

It is not a certified complete system and cannot guarantee 100% accuracy. No
new speech-model training or independent accuracy measurement was possible:
the supplied manifests reference external audio that is not in this ZIP.
Historical metrics elsewhere in the project are retained as supplied evidence,
not claimed as fresh test results.

## Start locally on Windows

Install Python 3.11 or 3.12 and Node.js. From this folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup-backend.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\setup-frontend.ps1
```

Then start these in two separate terminals:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\start-backend.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\start-frontend.ps1
```

Open http://localhost:5173. The default backend address is http://127.0.0.1:8000.
The lightweight installation includes the trained synthetic detector. To install
the large optional ECAPA/Whisper runtime, rerun backend setup with `-FullAudio`.
These optional libraries and their models are not the same as the lightweight
detector. Missing capabilities must be reported as unavailable.

FFmpeg is needed for formats the installed libsndfile cannot decode. Follow the
existing backend environment examples for ASR/contextual-provider configuration;
no private API credentials are supplied. Do not put provider keys in the frontend.

## Read before presenting or deploying

- `docs/REPAIR_VERIFICATION.md`: changes, real tests, unresolved work.
- `docs/DATASET_READINESS.md`: missing data and split-validation instructions.
- `docs/OBJECTIVE_INVENTORY.json`: conservative acceptance inventory.
- `verification/`: test logs and measured runtime/data evidence.

Do not expose this backend publicly as-is. Application-level user authentication
and per-user record ownership remain unfinished. Local SQLite history is not a
multi-user security boundary. A Vercel `/tmp` SQLite file is ephemeral; use a
persistent, authenticated backend for production. No production deployment was
changed during this ZIP-only repair.
