import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1280, height: 900 } });
const errs = [];
p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
await p.goto('http://127.0.0.1:5173/protection', { waitUntil: 'networkidle' });
await p.waitForTimeout(2500);
const row = p.locator('li button, .history-row').first();
if (await row.count()) { await row.click().catch(() => {}); }
await p.waitForTimeout(3000);
const body = await p.locator('body').innerText();
console.log(JSON.stringify({
  groqLabel: /Groq \(GPT-OSS\) contextual analysis/.test(body),
  scamShown: /LIKELY SCAM|KIDNAPPING/i.test(body),
  criticalShown: /CRITICAL/i.test(body),
  staleGemini: /Gemini AI contextual/.test(body),
  errors: errs.slice(0, 3),
}, null, 1));
const el = p.locator('text=/contextual analysis/i').first();
await el.scrollIntoViewIfNeeded();
await p.waitForTimeout(600);
const box = await el.boundingBox();
const vp = p.viewportSize();
const clip = { x: Math.max(0, Math.min(box.x - 30, 470)), y: Math.max(0, box.y - 90), width: 740, height: 430 };
clip.width = Math.min(clip.width, vp.width - clip.x);
clip.height = Math.min(clip.height, vp.height - clip.y);
await p.screenshot({ path: 'groq-live-label.png', clip });
await b.close();
