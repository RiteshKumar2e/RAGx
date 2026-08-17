import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_');
  // In development we proxy /api to the backend so the browser sees a
  // same-origin request and CORS never enters the picture.
  const backend = env.VITE_BACKEND_ORIGIN || 'http://localhost:8000';

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(process.cwd(), 'src'),
      },
    },
    server: {
      port: 5173,
      strictPort: false,
      proxy: {
        '/api': {
          target: backend,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: mode !== 'production',
      chunkSizeWarningLimit: 900,
      rollupOptions: {
        output: {
          // Split the heavy visualisation libraries so the first paint of the
          // Dashboard and Research Assistant is not blocked by React Flow.
          manualChunks: {
            react: ['react', 'react-dom', 'react-router-dom'],
            charts: ['recharts'],
            graph: ['reactflow'],
            markdown: ['react-markdown', 'remark-gfm'],
          },
        },
      },
    },
  };
});
