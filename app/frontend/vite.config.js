import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    historyApiFallback: true, // 새로고침 시 404 방지
    proxy: {
      // /api/* → backend :8000
      '/api': {
        target:       'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // WebSocket (향후 파이프라인 실시간 알림)
      '/ws': {
        target:    'ws://localhost:8000',
        ws:        true,
        changeOrigin: true,
      },
    },
  },
})
