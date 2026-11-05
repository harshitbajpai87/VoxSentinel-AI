// Decorative audio-waveform component used in processing indicators and
// headers. It is a visual indicator only — it is never presented as
// evidence about whether audio is synthetic.
//
// Heights come from a fixed deterministic pattern, so it renders identically
// everywhere (useful in UI tests and screenshots).

const BAR_COUNT = 36

function barHeight(i: number): number {
  // Deterministic pseudo-waveform. Range 0.15 .. 1.
  const v = Math.abs(Math.sin(i * 0.55) * Math.cos(i * 0.21)) + 0.12
  return Math.min(1, v)
}

export function Waveform({
  active = false,
  className = '',
  barClassName = 'bg-accent',
}: {
  /** When true the bars pulse as a "processing" indicator. */
  active?: boolean
  className?: string
  barClassName?: string
}) {
  return (
    <div
      className={`flex h-10 items-center gap-[3px] ${className}`}
      role={active ? 'status' : 'img'}
      aria-label={active ? 'Processing audio' : 'Audio waveform decoration'}
    >
      {Array.from({ length: BAR_COUNT }, (_, i) => {
        const h = Math.round(barHeight(i) * 100)
        return (
          <span
            key={i}
            className={`w-[3px] rounded-full ${barClassName} ${
              active ? 'animate-pulse-bar' : 'opacity-70'
            }`}
            style={{
              height: `${h}%`,
              animationDelay: active ? `${(i % 12) * 90}ms` : undefined,
              animationDuration: '1.1s',
            }}
          />
        )
      })}
    </div>
  )
}
