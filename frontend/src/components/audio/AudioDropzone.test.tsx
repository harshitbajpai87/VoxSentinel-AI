import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { AudioDropzone } from './AudioDropzone'

const wav = new File([new Uint8Array([1, 2, 3])], 'clip.wav', { type: 'audio/wav' })
const mp3 = new File([new Uint8Array([1, 2, 3])], 'clip.mp3', { type: 'audio/mpeg' })

function zone() {
  return screen.getByTestId('audio-dropzone')
}

afterEach(() => cleanup())

describe('AudioDropzone', () => {
  it('passes a dropped file to the parent handler', () => {
    const onFileSelected = vi.fn()
    render(<AudioDropzone onFileSelected={onFileSelected} />)
    fireEvent.drop(zone(), { dataTransfer: { files: [wav] } })
    expect(onFileSelected).toHaveBeenCalledTimes(1)
    expect((onFileSelected.mock.calls[0][0] as File).name).toBe('clip.wav')
  })

  it('uses only the first file when several are dropped', () => {
    const onFileSelected = vi.fn()
    render(<AudioDropzone onFileSelected={onFileSelected} />)
    fireEvent.drop(zone(), { dataTransfer: { files: [wav, mp3] } })
    expect(onFileSelected).toHaveBeenCalledTimes(1)
    expect((onFileSelected.mock.calls[0][0] as File).name).toBe('clip.wav')
  })

  it('ignores drops entirely while disabled', () => {
    const onFileSelected = vi.fn()
    render(<AudioDropzone onFileSelected={onFileSelected} disabled />)
    fireEvent.drop(zone(), { dataTransfer: { files: [wav] } })
    expect(onFileSelected).not.toHaveBeenCalled()
  })

  it('does not crash on a drop with no files', () => {
    const onFileSelected = vi.fn()
    render(<AudioDropzone onFileSelected={onFileSelected} />)
    fireEvent.drop(zone(), { dataTransfer: { files: [] } })
    fireEvent.drop(zone(), { dataTransfer: null })
    expect(onFileSelected).not.toHaveBeenCalled()
  })
})
