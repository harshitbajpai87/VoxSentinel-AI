import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
const errs = [];
p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
await p.goto('http://127.0.0.1:5173/protection', { waitUntil: 'networkidle' });
await p.waitForTimeout(1500);

// Upload the ransom clip through the real UI.
const fileInput = p.locator('input[type="file"]').first();
await fileInput.setInputFiles('/tmp/ransom.mp3');
await p.waitForTimeout(1200);

// Consent if present.
const consent = p.locator('input[type="checkbox"]').first();
if (await consent.count()) { await consent.check().catch(() => {}); }

const analyze = p.getByRole('button', { name: /analy/i }).first();
await analyze.click();
await p.waitForSelector('text=/analysis result|verdict/i', { timeout: 90000 });
await p.waitForTimeout(3000);

const t = await p.locator('body').innerText();
console.log(JSON.stringify({
  groqLabel: /Groq \(GPT-OSS\) contextual analysis/i.test(t),
  anyGroq: /groq/i.test(t),
  scam: /LIKELY SCAM/i.test(t),
  critical: /CRITICAL/.test(t),
  kidnap: /KIDNAPPING/i.test(t),
  staleGemini: /Gemini used|gemini-test-model/i.test(t),
  errors: errs.slice(0, 3),
}, null, 1));

const el = p.locator('text=/contextual analysis/i').last();
await el.scrollIntoViewIfNeeded();
await p.waitForTimeout(700);
const box = await el.boundingBox();
const vp = p.viewportSize();
const clip = { x: Math.max(0, box.x - 40), y: Math.max(0, box.y - 110), width: 760, height: 460 };
clip.width = Math.min(clip.width, vp.width - clip.x);
clip.height = Math.min(clip.height, vp.height - clip.y);
await p.screenshot({ path: 'groq-live-result.png', clip });
await b.close();
