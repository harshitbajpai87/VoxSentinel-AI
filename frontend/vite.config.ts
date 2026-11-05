import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite reads VITE_API_BASE_URL from .env.local / shell env at dev/build time.
export default defineConfig({
  plugins: [react()],
})
