import { useCallback, useEffect, useRef, useState } from 'react'

export interface AudioPlayer {
  playing: boolean
  currentTimeS: number
  durationS: number
  toggle: () => void
  seekToFraction: (fraction: number) => void
}

/**
 * Playback for a local object URL via an HTMLAudioElement. All listeners are
 * removed and the element is released when the URL changes or the component
 * unmounts — no leaked audio elements between files.
 */
export function useAudioPlayer(url: string | null): AudioPlayer {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTimeS, setCurrentTimeS] = useState(0)
  const [durationS, setDurationS] = useState(0)

  useEffect(() => {
    if (!url) {
      audioRef.current = null
      setPlaying(false)
      setCurrentTimeS(0)
      setDurationS(0)
      return
    }

    const audio = new Audio()
    audio.preload = 'metadata'
    audio.src = url
    audioRef.current = audio

    const onTime = () => setCurrentTimeS(audio.currentTime)
    const onMeta = () =>
      setDurationS(Number.isFinite(audio.duration) ? audio.duration : 0)
    const onEnded = () => {
      setPlaying(false)
      setCurrentTimeS(0)
    }
    const onPlay = () => setPlaying(true)
    const onPause = () => setPlaying(false)

    audio.addEventListener('timeupdate', onTime)
    audio.addEventListener('loadedmetadata', onMeta)
    audio.addEventListener('durationchange', onMeta)
    audio.addEventListener('ended', onEnded)
    audio.addEventListener('play', onPlay)
    audio.addEventListener('pause', onPause)

    return () => {
      audio.pause()
      audio.removeEventListener('timeupdate', onTime)
      audio.removeEventListener('loadedmetadata', onMeta)
      audio.removeEventListener('durationchange', onMeta)
      audio.removeEventListener('ended', onEnded)
      audio.removeEventListener('play', onPlay)
      audio.removeEventListener('pause', onPause)
      audio.removeAttribute('src')
      audio.load() // releases the resource
      audioRef.current = null
    }
  }, [url])

  const toggle = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) {
      void audio.play().catch(() => setPlaying(false))
    } else {
      audio.pause()
    }
  }, [])

  const seekToFraction = useCallback((fraction: number) => {
    const audio = audioRef.current
    if (!audio || !Number.isFinite(audio.duration) || audio.duration <= 0) return
    const clamped = Math.min(1, Math.max(0, fraction))
    audio.currentTime = clamped * audio.duration
    setCurrentTimeS(audio.currentTime)
  }, [])

  return { playing, currentTimeS, durationS, toggle, seekToFraction }
}
