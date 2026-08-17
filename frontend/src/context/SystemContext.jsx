import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { systemService } from '../services';

/**
 * Application-wide system state: LLM provider status, component health and
 * operator warnings.
 *
 * Loaded once at mount and refreshed on demand, so the provider indicator and
 * the configuration banners stay consistent across every page without each one
 * re-fetching.
 */
const SystemContext = createContext(null);

export function SystemProvider({ children }) {
  const [settings, setSettings] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [settingsPayload, healthPayload] = await Promise.all([
        systemService.settings(),
        systemService.health(false),
      ]);
      setSettings(settingsPayload);
      setHealth(healthPayload);
    } catch (caught) {
      setError(caught);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const value = useMemo(() => {
    const providers = settings?.llm?.providers || [];
    const configured = providers.filter((provider) => provider.configured);

    return {
      settings,
      health,
      loading,
      error,
      reload: load,
      setSettings,
      providers,
      configuredProviders: configured,
      // Generation requires at least one cloud provider; retrieval does not.
      hasLlm: Boolean(settings?.llm?.any_configured),
      // True when the offline development embedder is active — every page that
      // reports retrieval quality must say so.
      usingDevEmbedder: settings?.embeddings?.production_ready === false,
      warnings: settings?.warnings || health?.warnings || [],
      strategies: settings?.strategies || [],
      graphBackend: settings?.storage?.graph_store?.backend,
      vectorMode: settings?.storage?.vector_store?.mode,
    };
  }, [settings, health, loading, error, load]);

  return <SystemContext.Provider value={value}>{children}</SystemContext.Provider>;
}

export function useSystem() {
  const context = useContext(SystemContext);
  if (!context) throw new Error('useSystem must be used inside a SystemProvider.');
  return context;
}

export default SystemContext;
