// Render smoke test: server-renders every route and asserts expected content.
// Not a substitute for browser tests (Phase 8) but catches runtime crashes
// in every page component.
import React from 'react'
import { renderToString } from 'react-dom/server'
import { StaticRouter } from 'react-router-dom/server'
import App from '../src/App'
import { ApiStatusProvider } from '../src/lib/apiStatus'

const ROUTES: [string, string][] = [
  ['/', 'Voice Authenticity. Trusted.'],
  ['/protection', 'Family Emergency &amp; Scam Protection'],
  ['/dashboard', 'Overview'],
  ['/analyze', 'Analyze Audio'],
  ['/verify', 'Speaker Verification'],
  ['/history', 'Analysis History'],
  ['/evaluation', 'Model Evaluation'],
  ['/explainability', 'Explainability'],
  ['/security', 'Security'],
  ['/docs', 'Documentation'],
  ['/settings', 'Settings &amp; Privacy'],
]

let failures = 0
for (const [path, marker] of ROUTES) {
  try {
    const html = renderToString(
      <StaticRouter location={path}>
        <ApiStatusProvider>
          <App />
        </ApiStatusProvider>
      </StaticRouter>,
    )
    if (html.includes(marker)) {
      console.log(`PASS ${path} (${html.length} chars, contains "${marker}")`)
    } else {
      console.error(`FAIL ${path}: marker "${marker}" not found`)
      failures++
    }
  } catch (err) {
    console.error(`FAIL ${path}: threw ${err}`)
    failures++
  }
}
if (failures > 0) {
  console.error(`${failures} route(s) failed`)
  process.exit(1)
}
console.log('All 10 routes rendered without errors.')
