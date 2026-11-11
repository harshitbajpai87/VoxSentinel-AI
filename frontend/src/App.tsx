import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/layout/AppLayout'
import LandingPage from './pages/LandingPage'
import DashboardPage from './pages/DashboardPage'
import ProtectionPage from './pages/ProtectionPage'
import AnalyzeAudioPage from './pages/AnalyzeAudioPage'
import RecordAudioPage from './pages/RecordAudioPage'
import SpeakerVerificationPage from './pages/SpeakerVerificationPage'
import HistoryPage from './pages/HistoryPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route element={<AppLayout />}>
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/protection" element={<ProtectionPage />} />
        <Route path="/analyze" element={<AnalyzeAudioPage />} />
        <Route path="/record" element={<RecordAudioPage />} />
        <Route path="/verify" element={<SpeakerVerificationPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  )
}
