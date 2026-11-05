import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1280, height: 900 } });
await p.goto('http://127.0.0.1:5173/protection', { waitUntil: 'networkidle' });
await p.waitForTimeout(2500);
const row = p.locator('li button, .history-row').first();
if (await row.count()) { await row.click().catch(() => {}); }
await p.waitForTimeout(3000);
const text = await p.locator('body').innerText();
console.log(text.split('\n').filter(l => /groq|contextual|provider|model/i.test(l)).slice(0, 12).join('\n---\n'));
await b.close();
