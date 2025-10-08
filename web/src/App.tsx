import React, { useCallback } from 'react'
import { ChatKit, useChatKit } from '@openai/chatkit-react'

const getBase = () => {
  // Prefer same-origin for production; allow override via global if needed
  const w = globalThis as any
  return typeof w.__BACKEND_URL__ === 'string' ? w.__BACKEND_URL__ : ''
}

const App: React.FC = () => {
  const getClientSecret = useCallback(async (current: string | null) => {
    const base = getBase()
    const res = await fetch(`${base}/api/chatkit/session`, { method: 'POST' })
    const raw = await res.text()
    if (!res.ok) {
      throw new Error(raw || `Session request failed (${res.status})`)
    }
    const data = raw ? JSON.parse(raw) : {}
    const secret = data.client_secret as string | undefined
    if (!secret) throw new Error('Missing client_secret from backend')
    return secret
  }, [])

  const chatkit = useChatKit({
    api: { getClientSecret },
    theme: { colorScheme: 'dark', radius: 'round' },
    startScreen: { greeting: 'How can I help you today?' },
    composer: { placeholder: 'Ask anything about markets, execution, or strategy…' },
    onError: ({ error }) => console.error('ChatKit error', error),
  })

  return (
    <main className="shell">
      <div className="chat">
        <ChatKit control={chatkit.control} className="chatkit" />
      </div>
      <aside className="dock">
        <h2>Insights &amp; Actions</h2>
        <p className="muted">Assistant widgets will appear here when relevant.</p>
      </aside>
    </main>
  )
}

export default App

