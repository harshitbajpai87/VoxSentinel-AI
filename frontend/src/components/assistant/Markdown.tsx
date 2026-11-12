import type { ReactNode } from 'react'

/**
 * Renders the small Markdown subset Gemini actually uses in answers
 * (headings, bullet/numbered lists, bold, italic, inline code, code blocks).
 *
 * Deliberately not `dangerouslySetInnerHTML`: every node is built as React
 * elements, so model output can never inject markup into the page.
 */

type Block =
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'bullets'; items: string[] }
  | { kind: 'numbers'; items: string[] }
  | { kind: 'code'; text: string }

const BULLET = /^[-*•]\s+(.*)$/
const NUMBERED = /^(\d+)[.)]\s+(.*)$/

function parse(source: string): Block[] {
  const lines = source.replace(/\r\n/g, '\n').split('\n')
  const blocks: Block[] = []
  let paragraph: string[] = []
  let bullets: string[] = []
  let numbers: string[] = []
  let code: string[] | null = null

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ kind: 'paragraph', text: paragraph.join(' ').trim() })
      paragraph = []
    }
  }
  const flushLists = () => {
    if (bullets.length) {
      blocks.push({ kind: 'bullets', items: bullets })
      bullets = []
    }
    if (numbers.length) {
      blocks.push({ kind: 'numbers', items: numbers })
      numbers = []
    }
  }

  for (const raw of lines) {
    const line = raw.trimEnd()
    if (line.trimStart().startsWith('```')) {
      if (code) {
        blocks.push({ kind: 'code', text: code.join('\n') })
        code = null
      } else {
        flushParagraph()
        flushLists()
        code = []
      }
      continue
    }
    if (code) {
      code.push(raw)
      continue
    }
    if (!line.trim()) {
      flushParagraph()
      flushLists()
      continue
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line)
    if (heading) {
      flushParagraph()
      flushLists()
      blocks.push({ kind: 'heading', level: heading[1].length, text: heading[2].trim() })
      continue
    }
    const bullet = BULLET.exec(line.trim())
    if (bullet) {
      flushParagraph()
      if (numbers.length) {
        blocks.push({ kind: 'numbers', items: numbers })
        numbers = []
      }
      bullets.push(bullet[1].trim())
      continue
    }
    const numbered = NUMBERED.exec(line.trim())
    if (numbered) {
      flushParagraph()
      if (bullets.length) {
        blocks.push({ kind: 'bullets', items: bullets })
        bullets = []
      }
      numbers.push(numbered[2].trim())
      continue
    }
    flushLists()
    paragraph.push(line.trim())
  }
  if (code) blocks.push({ kind: 'code', text: code.join('\n') })
  flushParagraph()
  flushLists()
  return blocks
}

const INLINE = /(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\*[^*]+\*|_[^_]+_)/g

function inline(text: string, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = []
  let index = 0
  for (const piece of text.split(INLINE)) {
    if (!piece) continue
    const key = `${keyPrefix}-${index++}`
    if ((piece.startsWith('**') && piece.endsWith('**')) || (piece.startsWith('__') && piece.endsWith('__'))) {
      out.push(<strong key={key}>{piece.slice(2, -2)}</strong>)
    } else if (piece.startsWith('`') && piece.endsWith('`') && piece.length > 2) {
      out.push(<code key={key} className="assistant-code">{piece.slice(1, -1)}</code>)
    } else if ((piece.startsWith('*') && piece.endsWith('*')) || (piece.startsWith('_') && piece.endsWith('_'))) {
      out.push(<em key={key}>{piece.slice(1, -1)}</em>)
    } else {
      out.push(<span key={key}>{piece}</span>)
    }
  }
  return out
}

export function Markdown({ text }: { text: string }) {
  const blocks = parse(text)
  return (
    <>
      {blocks.map((block, i) => {
        if (block.kind === 'heading') {
          return (
            <p key={i} className={`assistant-h${Math.min(block.level, 4)}`}>
              {inline(block.text, `h${i}`)}
            </p>
          )
        }
        if (block.kind === 'bullets') {
          return (
            <ul key={i} className="assistant-list">
              {block.items.map((item, j) => (
                <li key={j}>{inline(item, `b${i}-${j}`)}</li>
              ))}
            </ul>
          )
        }
        if (block.kind === 'numbers') {
          return (
            <ol key={i} className="assistant-list">
              {block.items.map((item, j) => (
                <li key={j}>{inline(item, `n${i}-${j}`)}</li>
              ))}
            </ol>
          )
        }
        if (block.kind === 'code') {
          return (
            <pre key={i} className="assistant-pre">
              <code>{block.text}</code>
            </pre>
          )
        }
        return <p key={i}>{inline(block.text, `p${i}`)}</p>
      })}
    </>
  )
}
