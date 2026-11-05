import { chromium } from 'playwright'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
await page.goto('http://localhost:5173/dashboard', { waitUntil: 'networkidle' })
const found = await page.evaluate(() => {
  const els = [...document.querySelectorAll('a, button')]
  return els.filter((e) => /analyze audio/i.test(e.textContent || '')).map((e) => ({
    tag: e.tagName,
    classes: e.className.slice(0, 120),
    bg: getComputedStyle(e).backgroundImage.slice(0, 90),
  }))
})
console.log(JSON.stringify(found, null, 2))
await browser.close()
