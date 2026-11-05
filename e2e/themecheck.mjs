import { chromium } from 'playwright'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
await page.goto('http://localhost:5173/analyze', { waitUntil: 'networkidle' })
const checks = await page.evaluate(() => {
  const bg = (el) => el ? getComputedStyle(el).backgroundColor : 'missing'
  const sidebar = document.querySelector('nav') ?? document.querySelector('aside')
  const main = document.body
  const card = document.querySelector('[class*="bg-surface"]')
  const html = document.documentElement
  return {
    colorScheme: getComputedStyle(html).colorScheme,
    bodyBg: bg(main),
    sidebarBg: bg(sidebar),
    surfaceSample: bg(card),
    bodyText: getComputedStyle(main).color,
  }
})
console.log(JSON.stringify(checks, null, 2))
await page.screenshot({ path: 'light-analyze.png', fullPage: false })
await browser.close()
