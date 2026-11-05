import { chromium } from 'playwright'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const errs = []
page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()) })
await page.goto('http://localhost:5173/protection', { waitUntil: 'networkidle' })
await page.waitForTimeout(1500)

// Open the newest saved analysis from history
const firstItem = page.locator('ul li button').first()
await firstItem.click()
await page.waitForSelector('text=Analysis result', { timeout: 30000 })
await page.waitForTimeout(500)
const raw = await page.locator('body').innerText()
const t = raw.toLowerCase()
console.log('risk category:', /(low|medium|high|critical|inconclusive) risk/.test(t) ? 'YES' : 'NO')
console.log('transcript blockquote text:', /Go! Do you hear|staircase|minutes/.test(raw) ? 'REAL TEXT PRESENT' : 'no text')
console.log('recommended actions:', t.includes('recommended actions') ? 'YES' : 'NO')
console.log('limitations:', t.includes('limitations') ? 'YES' : 'NO')
console.log('risk reasoning:', t.includes('how the risk was assessed') ? 'YES' : 'NO')
console.log('honesty footer:', t.includes('not a guarantee') ? 'YES' : 'NO')
console.log('console errors:', errs.length ? errs.slice(0, 2) : 'none')
await browser.close()
