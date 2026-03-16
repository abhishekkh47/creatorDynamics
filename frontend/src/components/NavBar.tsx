import { NavLink } from 'react-router-dom'

const links = [
  { to: '/',        label: 'Pre-Post Predict' },
  { to: '/tracker', label: '1h Check-in' },
  { to: '/history', label: 'History' },
]

export default function NavBar() {
  return (
    <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-5xl mx-auto px-4 flex items-center justify-between h-14">
        <span className="font-bold text-lg tracking-tight text-white">
          Creator<span className="text-brand-500">Dynamix</span>
        </span>
        <nav className="flex gap-1">
          {links.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
