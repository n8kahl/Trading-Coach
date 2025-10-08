import React, { useEffect, useState } from 'react'

const getBase = () => {
  // Prefer same-origin for production; allow override via global if needed
  const w = globalThis as any
  return typeof w.__BACKEND_URL__ === 'string' ? w.__BACKEND_URL__ : ''
}

const App: React.FC = () => {
  const [clientSecret, setClientSecret] = useState<string | null>(null)
  const [workflowId, setWorkflowId] = useState<string | null>(null)
  const [userId, setUserId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const base = getBase()
        const res = await fetch(`${base}/api/chatkit/session`, { method: 'POST' })
        const raw = await res.text()
        if (!res.ok) throw new Error(raw || `Session request failed (${res.status})`)
        const data = raw ? JSON.parse(raw) : {}
        const secret = data.client_secret as string | undefined
        if (!secret) throw new Error('Missing client_secret from backend')
        setClientSecret(secret)
        setWorkflowId((data.workflow_id as string) || null)
        setUserId((data.user_id as string) || null)
      } catch (e: any) {
        console.error('Failed to init ChatKit', e)
        setError(String(e?.message || e))
      }
    }
    bootstrap()
  }, [])

  return (
    <main className="shell">
      <div className="chat">
        {/* The ChatKit web component is registered by a script tag in index.html (/assets/chatkit.js) */}
        {clientSecret ? (
          <openai-chatkit
            className="chatkit"
            client-secret={clientSecret}
            workflow-id={workflowId || undefined}
            user-id={userId || undefined}
          />
        ) : (
          <div className="muted" style={{ padding: '1rem' }}>
            {error ? `Error: ${error}` : 'Connecting…'}
          </div>
        )}
      </div>
      <aside className="dock">
        <h2>Insights &amp; Actions</h2>
        <p className="muted">Assistant widgets will appear here when relevant.</p>
      </aside>
    </main>
  )
}

export default App
