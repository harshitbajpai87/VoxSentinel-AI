import { chromium } from 'playwright'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const errs = []
page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()) })
await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
const t = await page.locator('body').innerText()
console.log('landing headline:', /Know What.s Real/.test(t) ? 'PRESENT' : 'MISSING')
console.log('gradient CTA Analyze Audio:', /Analyze Audio/.test(t) ? 'PRESENT' : 'MISSING')
console.log('illustrative demo label:', /Illustrative demo/i.test(t) ? 'LABELED' : 'NOT LABELED')
const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor)
console.log('body background (protected):', bg)
// dashboard reachable from landing CTA
await page.click('a[href="/analyze"]')
await page.waitForLoadState('networkidle')
console.log('CTA navigates to analyze:', page.url().endsWith('/analyze') ? 'YES' : 'NO')
// mobile landing
await page.setViewportSize({ width: 390, height: 844 })
await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)
console.log('mobile 390px landing overflow:', overflow ? 'FAIL' : 'none')
console.log('console errors:', errs.length === 0 ? 'none' : errs.slice(0, 3))
await browser.close()
