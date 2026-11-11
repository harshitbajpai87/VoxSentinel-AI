# VoxSentinel AI — light frontend redesign

## What changed

A frontend-only redesign based on the supplied layout references and LIA interaction ideas, not their dark colour schemes.

- Light ivory surfaces with terracotta, plum, peach, sage and warm neutral accents. No dark theme or blue theme. The favicon now matches.
- A custom animated SVG voice illustration with pointer-responsive tilt. It is decorative and explicitly labelled, not a detection result.
- Large editorial typography, a circular analysis link, scroll reveals, directional link hovers and a layered card deck that fans on hover.
- Four keyboard-operable tool tabs linked to existing analysis, speaker verification, scam protection and recording routes.
- Expandable FAQs, mobile navigation, a decorative-motion pause control, system reduced-motion support and visible keyboard focus.
- Consistent workspace colours, buttons, cards, page headings and navigation, including links to existing documentation, explainability, audit and settings pages.
- Clearer overview copy without the old phase labels. No invented statistics, endorsements or scores.
- Five landing-page regression tests.

No reference-site images, third-party plugins, new runtime dependencies or remote fonts were added.

## Install safely

Keep your original project ZIP as a backup. Work in a separate copy first.

**Frontend-only update ZIP (recommended):** extract it at the project root, where `frontend` and `backend` sit side by side. Merge/replace the `frontend` files. It contains no backend files.

**Complete project ZIP:** extract it into a new folder. It contains the original project with the revised frontend and this handoff.

The archives exclude generated `node_modules`, build output and the temporary responsive QA harness.

From the `frontend` folder, run:

```powershell
npm ci
npm run dev -- --port 5174 --strictPort
```

Open `http://localhost:5174/`. Port 5174 avoids replacing an existing preview on port 5173. If 5174 is already occupied, choose another free port.

For a production build:

```powershell
npm run build
```

Run the existing backend using the project's original instructions if you need live analysis. Keep its configuration and model files unchanged. The Settings page still controls the API address as before.

## Verification

- TypeScript check and Vite production build: passed.
- Route smoke render: all 11 actual application routes passed.
- Vitest: 81 tests passed in 11 files, including the five added landing-page tests.
- Desktop browser review: 1284 × 804 viewport.
- Landing layout checked at 320, 375, 390, 768, 1024 and 1280 CSS-pixel frame widths; no horizontal page overflow. The rotating decorative artwork is intentionally clipped.
- All ten workspace routes rendered at phone width without horizontal page overflow.
- Live tool-tab selection and mobile-menu interaction checked in the browser.
- Source comparisons use SHA-256. All 124 backend files, all nine existing frontend `src/lib` files, and original training/evaluation/docs files were unchanged.
- The API request library, model code, stored model artifacts and app route declarations were not modified.

The original smoke script expected a slogan that has now changed and listed an evaluation route absent from the original app router. Its markers were aligned with the current landing page and the existing recording route. The evaluation component and its tests remain untouched; no new evaluation route was added.

## Limits of this verification

The backend was offline in the preview. This handoff does not claim that live AI inference, model accuracy, microphone hardware, or deployment has been validated. Backend-dependent controls continue to require the original running backend. Automated API tests use controlled test responses.

Existing React Router future-flag notices and an npm environment-setting warning appeared during checks; they did not fail the tests or build. Dependency upgrades were outside this redesign.

The temporary preview was used only for QA. Start the app with the commands above to review the animations and continue iterating on the design.

See `frontend-redesign-verification.json` for the exact source changes and preserved-file checks.
