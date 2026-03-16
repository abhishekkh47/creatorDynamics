import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import NavBar from '@/components/NavBar'
import PrePost from '@/pages/PrePost'
import LiveTracker from '@/pages/LiveTracker'
import History from '@/pages/History'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen">
        <NavBar />
        <main>
          <Routes>
            <Route path="/"         element={<PrePost />} />
            <Route path="/tracker"  element={<LiveTracker />} />
            <Route path="/history"  element={<History />} />
            {/* Redirect any unknown path to home */}
            <Route path="*"         element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
