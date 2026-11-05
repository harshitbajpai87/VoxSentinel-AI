import { chromium } from 'playwright'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const errs = []
page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()) })

await page.goto('http://localhost:5173/protection', { waitUntil: 'networkidle' })
let t = await page.locator('body').innerText()
console.log('nav item present:', /Family Emergency & Scam Protection/.test(t) ? 'YES' : 'NO')
console.log('dropzone present:', (await page.locator('[data-testid="protection-dropzone"]').count()) > 0 ? 'YES' : 'NO')

// upload real speech, consent, analyze
await page.locator('input[type="file"]').first().setInputFiles('/tmp/test_speech.wav')
await page.waitForTimeout(400)
console.log('file shown:', (await page.locator('body').innerText()).includes('test_speech.wav') ? 'YES' : 'NO')
const analyzeBtn = page.locator('button', { hasText: /Analyze call/i }).first()
console.log('analyze disabled without consent:', await analyzeBtn.isDisabled() ? 'YES (correct)' : 'NO (bug)')
await page.locator('input[type="checkbox"]').check()
await analyzeBtn.click()
await page.waitForSelector('text=Analysis result', { timeout: 120000 })
await page.waitForTimeout(600)
t = await page.locator('body').innerText()
console.log('risk category shown:', /(LOW|MEDIUM|HIGH|CRITICAL|INCONCLUSIVE) risk/.test(t) ? 'YES' : 'NO')
console.log('transcript shown:', /Transcript/.test(t) ? 'YES' : 'NO')
console.log('recommended actions:', /Recommended actions/.test(t) ? 'YES' : 'NO')
console.log('limitations shown:', /Limitations/.test(t) ? 'YES' : 'NO')

// history should have the record
await page.waitForTimeout(1200)
t = await page.locator('body').innerText()
console.log('history lists analysis:', /test_speech\.wav/.test(t.split('Analysis history')[1] ?? '') ? 'YES' : 'check manually')

// responsive + overflow
for (const [name, w, h] of [['tablet', 768, 1024], ['mobile', 390, 844]]) {
  await page.setViewportSize({ width: w, height: h })
  await page.waitForTimeout(300)
  const ov = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)
  console.log(`${name} ${w}px overflow:`, ov ? 'FAIL' : 'none')
}
console.log('console errors:', errs.length ? errs.slice(0, 2) : 'none')
await browser.close()
