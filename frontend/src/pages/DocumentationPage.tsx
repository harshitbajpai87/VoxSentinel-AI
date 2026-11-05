import {
  Callout,
  Card,
  PageHeader,
  Tag,
} from '../components/ui'

function Code({ children }: { children: string }) {
  return (
    <pre className="overflow-x-auto rounded-md border border-line bg-canvas px-4 py-3 font-mono text-xs leading-relaxed text-accent">
      {children}
    </pre>
  )
}

const ENDPOINTS = [
  ['GET', '/health', 'Backend and model readiness'],
  ['GET', '/api/v1/models', 'Available detection and speaker models'],
  ['POST', '/api/v1/analyze', 'Analyze one audio file (multipart)'],
  ['POST', '/api/v1/speaker-verify', 'Reference vs. test comparison (multipart)'],
  ['GET', '/api/v1/analyses', 'List analyses'],
  ['GET', '/api/v1/analyses/{id}', 'Fetch one analysis'],
  ['GET', '/api/v1/evaluation', 'Evaluation summary and provenance'],
  ['GET', '/api/v1/evaluation/metrics', 'Measured metrics, or evaluated: false'],
]

export default function DocumentationPage() {
  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Documentation"
        subtitle="Everything a teammate needs to run and understand the project. Written for beginners, verified against the actual code."
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Card title="Run the frontend" subtitle="From the repository root:">
            <Code>{`cd frontend
npm install
cp .env.example .env.local   # optional
npm run dev`}</Code>
            <p className="mt-3 text-xs leading-relaxed text-ink-dim">
              Then open <span className="font-mono text-accent">http://localhost:5173</span>. The
              app works without the backend; every backend-dependent panel shows an explicit
              offline state.
            </p>
          </Card>

          <Card title="Configure the backend URL" subtitle="Never hardcode secrets in code.">
            <p className="text-xs leading-relaxed text-ink-dim">
              The frontend reads <span className="font-mono text-accent">VITE_API_BASE_URL</span>{' '}
              from <span className="font-mono text-ink">frontend/.env.local</span> at dev/build
              time. You can also override it at runtime from Settings &amp; Privacy. Default:
              <span className="font-mono text-accent"> http://localhost:8000</span>.
            </p>
          </Card>

          <Card title="API endpoints" subtitle="Full contract: docs/API_CONTRACT.md">
            <table className="w-full text-xs">
              <tbody className="divide-y divide-line/60">
                {ENDPOINTS.map(([method, path, desc]) => (
                  <tr key={path}>
                    <td className="py-2 pr-2 font-mono text-accent">{method}</td>
                    <td className="py-2 pr-3 font-mono text-ink">{path}</td>
                    <td className="py-2 text-ink-dim">{desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>

        <div className="space-y-6">
          <Card title="Architecture" subtitle="Full document: docs/ARCHITECTURE.md">
            <p className="text-xs leading-relaxed text-ink-dim">
              Two applications: a React + TypeScript frontend (this repo) and a Python FastAPI
              backend (Phase 4). They communicate only through the versioned REST API. The
              frontend contains no model logic; the backend contains no UI. The model sits
              behind an adapter so it can be replaced without rewriting the app.
            </p>
            <p className="mt-3 text-xs leading-relaxed text-ink-dim">
              Base44 hosts JavaScript only, so the Python detection backend can never run inside
              Base44. Keeping the two separate is an architectural requirement, not a
              limitation we worked around silently.
            </p>
          </Card>

          <Card title="Team workflow" subtitle="Full document: docs/TEAM_PLAN.md">
            <ul className="space-y-2 text-xs leading-relaxed text-ink-dim">
              <li>Harshit Bajpai — frontend integration, upload workflow, dashboard, frontend tests.</li>
              <li>Oisiky Sinha — Python backend, preprocessing, model integration, evaluation, backend tests.</li>
              <li>Both — architecture review, integration testing, README.</li>
              <li>Feature branches, small commits, pull requests with review before merge.</li>
              <li>Each teammate commits from their own GitHub account with their own verified email.</li>
            </ul>
          </Card>

          <Callout tone="ethics" title="Ethical use">
            VoxSentinel AI is a research and decision-support tool. It never claims definitive
            proof that audio is genuine or synthetic. For high-impact decisions (payments,
            access), it recommends independent human verification. Only process audio you are
            authorized to analyze.
          </Callout>
        </div>
      </div>
    </div>
  )
}
