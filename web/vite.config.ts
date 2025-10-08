import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

// Build to ../frontend_dist so FastAPI can serve it directly
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, '../frontend_dist'),
    emptyOutDir: true,
  },
})

