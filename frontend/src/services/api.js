import axios from 'axios';

/**
 * Centralised Axios client.
 *
 * Every network call in the application goes through this instance and the
 * service modules that wrap it — components never call axios directly. That
 * keeps base URLs, timeouts, error normalisation and request IDs in one place.
 *
 * Security note: this client never carries a provider credential. Gemini and
 * Groq keys live only in the backend environment; the browser talks to the
 * FastAPI backend, and the backend talks to the model providers.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: Number(import.meta.env.VITE_API_TIMEOUT_MS || 120000),
  headers: { 'Content-Type': 'application/json' },
});

// An optional shared secret for deployments that lock down mutating routes.
// It is not a model-provider key and grants no access to any LLM.
const RAGX_KEY = import.meta.env.VITE_RAGX_API_KEY;

api.interceptors.request.use((config) => {
  if (RAGX_KEY) config.headers['X-Ragx-Key'] = RAGX_KEY;
  config.metadata = { startedAt: performance.now() };
  return config;
});

/** A normalised, user-safe error. Raw stack traces never reach the UI. */
export class ApiError extends Error {
  constructor({ message, code, status, detail, retryable }) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.detail = detail;
    this.retryable = retryable;
  }
}

function normalizeError(error) {
  // 1. The request never left the browser / no response came back.
  if (error.code === 'ECONNABORTED') {
    return new ApiError({
      message:
        'The request timed out. Complex queries can take a while — try again, or reduce Top-K in Settings.',
      code: 'timeout',
      status: 0,
      retryable: true,
    });
  }

  if (!error.response) {
    return new ApiError({
      message:
        'Cannot reach the RAGX backend. Check that it is running on http://localhost:8000 and that VITE_API_BASE_URL is correct.',
      code: 'network_error',
      status: 0,
      retryable: true,
    });
  }

  const { status, data } = error.response;

  // 2. The backend returned its structured {error: {code, message, detail}} envelope.
  if (data && typeof data === 'object' && data.error) {
    return new ApiError({
      message: data.error.message || 'The request failed.',
      code: data.error.code || 'error',
      status,
      detail: data.error.detail,
      retryable: status >= 500 || status === 429,
    });
  }

  // 3. Anything else — map the status to a message a user can act on.
  const byStatus = {
    401: 'Not authorised. Check VITE_RAGX_API_KEY if the backend has an API key configured.',
    403: 'This action is not permitted.',
    404: 'Not found. It may have been deleted.',
    413: 'That file is too large for the configured upload limit.',
    415: 'That file type is not supported.',
    422: 'The request was rejected as invalid.',
    429: 'Too many requests — please wait a moment and retry.',
    502: 'The upstream LLM provider did not respond. Retrieval may still work.',
    503: 'A backend service is unavailable. Check the Settings page for component health.',
  };

  return new ApiError({
    message: byStatus[status] || `The request failed (HTTP ${status}).`,
    code: `http_${status}`,
    status,
    retryable: status >= 500 || status === 429,
  });
}

api.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(normalizeError(error)),
);

/** Unwraps `response.data` for the common case. */
export async function request(config) {
  const response = await api.request(config);
  return response.data;
}

export const get = (url, config = {}) => request({ method: 'GET', url, ...config });
export const post = (url, data, config = {}) => request({ method: 'POST', url, data, ...config });
export const patch = (url, data, config = {}) => request({ method: 'PATCH', url, data, ...config });
export const del = (url, config = {}) => request({ method: 'DELETE', url, ...config });

/** Absolute URL for a backend resource the browser loads directly (images, files). */
export function resourceUrl(path) {
  const base = BASE_URL.endsWith('/') ? BASE_URL.slice(0, -1) : BASE_URL;
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

export default api;
