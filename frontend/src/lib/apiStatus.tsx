import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'
import { api } from './api'
import type { HealthResponse } from './types'

export type ConnectionStatus = 'checking' | 'online' | 'offline'

interface ApiStatusState {
  status: ConnectionStatus
  health: HealthResponse | null
  lastCheckedAt: Date | null
  refresh: () => Promise<void>
}

const ApiStatusContext = createContext<ApiStatusState | null>(null)

const POLL_INTERVAL_MS = 15000

export function ApiStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ConnectionStatus>('checking')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [lastCheckedAt, setLastCheckedAt] = useState<Date | null>(null)

  const refresh = useCallback(async () => {
    try {
      const h = await api.getHealth()
      setHealth(h)
      setStatus('online')
    } catch {
      setHealth(null)
      setStatus('offline')
    } finally {
      setLastCheckedAt(new Date())
    }
  }, [])

  useEffect(() => {
    void refresh()
    const id = window.setInterval(() => void refresh(), POLL_INTERVAL_MS)
    return () => window.clearInterval(id)
  }, [refresh])

  return (
    <ApiStatusContext.Provider value={{ status, health, lastCheckedAt, refresh }}>
      {children}
    </ApiStatusContext.Provider>
  )
}

export function useApiStatus(): ApiStatusState {
  const ctx = useContext(ApiStatusContext)
  if (!ctx) throw new Error('useApiStatus must be used inside ApiStatusProvider')
  return ctx
}
