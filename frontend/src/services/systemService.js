import { get, patch, post } from './api';

/** Health, analytics, settings and the strategy catalogue. */
export const systemService = {
  /** `probe=true` makes a live request to each provider (costs a few tokens). */
  health(probe = false) {
    return get('/health', { params: { probe } });
  },

  /**
   * Cloud LLM provider status. Returns model names and whether a key is
   * configured — never the key itself.
   */
  llmStatus(probe = false) {
    return get('/llm/status', { params: { probe } });
  },

  analytics(days = 30) {
    return get('/analytics', { params: { days } });
  },

  settings() {
    return get('/settings');
  },

  /** Retrieval/verification tuning only. Credentials cannot be set from the UI. */
  updateSettings(payload) {
    return patch('/settings', payload);
  },

  strategies() {
    return get('/strategies');
  },

  clearCache() {
    return post('/cache/clear');
  },

  reloadSettings() {
    return post('/settings/reload');
  },
};

export default systemService;
