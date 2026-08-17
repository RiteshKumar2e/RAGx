import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_');

  // In development we proxy /api to the backend so the browser sees a
  // same-origin request and CORS never enters the picture.
  //
  // `localhost` is rewritten to `127.0.0.1` because Node 18+ resolves it to
  // IPv6 (::1) first, while uvicorn binds IPv4 only by default — the mismatch
  // shows up as a bare ECONNREFUSED that looks like "the backend is down".
  const backend = (env.VITE_BACKEND_ORIGIN || 'http://localhost:8000').replace(
    /^(https?:\/\/)localhost(?=[:/]|$)/i,
    '$1127.0.0.1',
  );

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
          configure: (proxy) => {
            // Without this, a wrong VITE_BACKEND_ORIGIN or an unrelated service
            // squatting on the port surfaces as an opaque 500 in the browser
            // with no indication of the cause. Fail loudly instead.
            proxy.on('error', (err, _req, res) => {
              const message =
                `[RAGX] Cannot reach the backend at ${backend} — ${err.code || err.message}.\n` +
                `Start it with:  cd backend && uvicorn app.main:app --reload --port ` +
                `${new URL(backend).port || 8000}\n` +
                `If another service already uses that port, pick a free one and set ` +
                `VITE_BACKEND_ORIGIN in frontend/.env to match.`;
              // eslint-disable-next-line no-console
              console.error(`\n${message}\n`);
              if (res && !res.headersSent && res.writeHead) {
                res.writeHead(502, { 'Content-Type': 'application/json' });
                res.end(
                  JSON.stringify({
                    error: { code: 'backend_unreachable', message, detail: null },
                  }),
                );
              }
            });

            // A response from something that is not RAGX (e.g. a different app
            // on the same port) is the other silent-failure mode.
            proxy.on('proxyRes', (proxyRes, req) => {
              if (proxyRes.statusCode === 404 && req.url?.startsWith('/api/v1/')) {
                // eslint-disable-next-line no-console
                console.warn(
                  `[RAGX] ${backend}${req.url} returned 404. Is a different ` +
                    `service running on that port?`,
                );
              }
            });
          },
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
