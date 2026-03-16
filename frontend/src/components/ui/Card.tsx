interface CardProps {
  children: React.ReactNode
  className?: string
}

export default function Card({ children, className = '' }: CardProps) {
  return (
    <div className={`rounded-2xl bg-gray-900 border border-gray-800 p-6 ${className}`}>
      {children}
    </div>
  )
}
