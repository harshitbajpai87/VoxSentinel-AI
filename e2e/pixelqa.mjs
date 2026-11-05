import { chromium } from 'playwright'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })

// Effective luminance: walk up the DOM to the first OPAQUE background
await page.goto('http://localhost:5173/dashboard', { waitUntil: 'networkidle' })
const lum = await page.evaluate(() => {
  const opaque = (el) => {
    while (el) {
      const c = getComputedStyle(el).backgroundColor
      const m = c.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/)
      if (m && (m[4] === undefined || +m[4] > 0.5)) {
        return (0.2126 * +m[1] + 0.7152 * +m[2] + 0.0722 * +m[3]) / 255
      }
      el = el.parentElement
    }
    return 1
  }
  const n = 40
  let sum = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const el = document.elementFromPoint(((i + 0.5) * innerWidth) / n, ((j + 0.5) * innerHeight) / n)
      sum += opaque(el)
    }
  return sum / (n * n)
})
console.log('viewport effective surface luminance:', lum.toFixed(3), lum > 0.75 ? '(LIGHT OK)' : '(TOO DARK)')

// Primary CTA on the dashboard: the gradient Analyze audio button
const btn = page.locator('a', { hasText: /analyze audio/i }).first()
if (await btn.count()) {
  const bg = await btn.evaluate((el) => getComputedStyle(el).backgroundImage)
  console.log('primary CTA gradient:', /linear-gradient/.test(bg) ? 'PRESENT' : 'MISSING', '|', bg.slice(0, 90))
} else {
  console.log('primary CTA gradient: button not found')
}
await page.screenshot({ path: 'light-theme-final.png' })
await browser.close()
