import { chromium } from 'playwright';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1280, height: 900 } });
await p.goto('http://127.0.0.1:5173/protection', { waitUntil: 'networkidle' });
await p.waitForTimeout(2000);
const row = p.locator('li button, .history-row').first();
if (await row.count()) { await row.click().catch(() => {}); }
await p.waitForTimeout(2500);
const el = p.locator('text=/contextual analysis/i').first();
await el.scrollIntoViewIfNeeded();
await p.waitForTimeout(600);
const box = await el.boundingBox();
const clip = {
  x: Math.max(0, Math.min(box.x - 30, 520)),
  y: Math.max(0, box.y - 30),
  width: 680,
  height: 260,
};
const vp = p.viewportSize();
clip.width = Math.min(clip.width, vp.width - clip.x);
clip.height = Math.min(clip.height, vp.height - clip.y);
await p.screenshot({ path: 'grok-label-crop.png', clip });
console.log('cropped', JSON.stringify(clip));
await b.close();
