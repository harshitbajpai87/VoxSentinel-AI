import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
const errs = [], failed = [];
p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
p.on('requestfailed', r => failed.push(`${r.method()} ${r.url()} :: ${r.failure()?.errorText}`));
await p.goto('http://127.0.0.1:5173/analyze', { waitUntil: 'networkidle' });
await p.waitForTimeout(1200);
const fileInput = p.locator('input[type="file"]').first();
await fileInput.setInputFiles('/tmp/ransom.mp3');
await p.waitForTimeout(1000);
const consent = p.locator('input[type="checkbox"]').first();
if (await consent.count()) await consent.check().catch(() => {});
await p.getByRole('button', { name: /analy/i }).first().click();
await p.waitForTimeout(9000);
const t = await p.locator('body').innerText();
console.log(JSON.stringify({
  corsErrorShown: /Could not reach the backend/i.test(t),
  transcriptShown: /kidnap|ransom|send 50/i.test(t),
  verdictShown: /likely scam|suspicious|legitimate/i.test(t),
  requestFailures: failed.slice(0, 3),
  consoleErrors: errs.slice(0, 2),
}, null, 1));
const el = p.locator('text=/analysis|transcript|verdict/i').first();
await el.scrollIntoViewIfNeeded().catch(() => {});
await p.waitForTimeout(400);
await p.screenshot({ path: 'analyze-fixed.png', fullPage: false });
await b.close();
