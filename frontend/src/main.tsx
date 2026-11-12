import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ApiStatusProvider } from './lib/apiStatus'
import { AssistantProvider } from './components/assistant/AssistantProvider'
import { AssistantWidget } from './components/assistant/AssistantWidget'
import './index.css'
import './pages/landing.css'

createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>
    <BrowserRouter>
      <ApiStatusProvider>
        <AssistantProvider>
          <App />
          <AssistantWidget />
        </AssistantProvider>
      </ApiStatusProvider>
    </BrowserRouter>
  </StrictMode>,
)
