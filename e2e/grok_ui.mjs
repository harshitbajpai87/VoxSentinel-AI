import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
const errs = [];
p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
await p.goto('http://127.0.0.1:5173/protection', { waitUntil: 'networkidle' });
await p.waitForTimeout(2000);

// Open the newest analysis in the history list
const firstRow = p.locator('[data-testid="analysis-row"], li button, .history-row').first();
if (await firstRow.count()) { await firstRow.click().catch(() => {}); }
await p.waitForTimeout(2500);

const body = await p.locator('body').innerText();
const grokLabel = /Grok \(xAI\) contextual analysis/.test(body);
const staleGemini = /Gemini AI contextual analysis/.test(body);
const fallbackText = /not configured|rule-based fallback/i.test(body);
console.log(JSON.stringify({ grokLabel, staleGemini, fallbackText, errors: errs.slice(0, 3) }, null, 1));
await p.screenshot({ path: 'grok-provider-label.png', fullPage: false });
await b.close();
