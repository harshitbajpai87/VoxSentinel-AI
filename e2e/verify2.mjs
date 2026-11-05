import { chromium } from 'playwright'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const out = []
page.on('console', (m) => { if (m.type() === 'error') out.push('console-error: ' + m.text()) })
await page.goto('http://localhost:5173/verify', { waitUntil: 'networkidle' })
const t = await page.locator('body').innerText()
console.log('verify page availability:', /available|ready/i.test(t) ? 'mentions available/ready' : 'no availability wording')
console.log('verify page model status:', (t.match(/ECAPA[^\n]*/) ?? ['not shown'])[0])
console.log('no console errors:', out.length === 0)
await browser.close()
