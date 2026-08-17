import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Data-fetching hook covering the four states every API-driven page must handle:
 * loading, success, empty and error.
 *
 * Returns `{ data, error, loading, refetch, setData }`. Stale responses from a
 * superseded request are discarded, and state is never written after unmount.
 */
export function useApi(fetcher, deps = [], { immediate = true, initialData = null } = {}) {
  const [data, setData] = useState(initialData);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(immediate);

  const mounted = useRef(true);
  const requestId = useRef(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const execute = useCallback(async (...args) => {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const result = await fetcherRef.current(...args);
      if (mounted.current && id === requestId.current) {
        setData(result);
        setLoading(false);
      }
      return result;
    } catch (caught) {
      if (mounted.current && id === requestId.current) {
        setError(caught);
        setLoading(false);
      }
      return undefined;
    }
  }, []);

  useEffect(() => {
    if (immediate) execute();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading, refetch: execute, setData };
}

/**
 * Imperative mutation hook for actions the user triggers (upload, delete, run).
 * Tracks `pending` and surfaces the normalised error for inline display.
 */
export function useMutation(mutator, { onSuccess, onError } = {}) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const mutate = useCallback(
    async (...args) => {
      setPending(true);
      setError(null);
      try {
        const result = await mutator(...args);
        if (mounted.current) setPending(false);
        onSuccess?.(result);
        return result;
      } catch (caught) {
        if (mounted.current) {
          setError(caught);
          setPending(false);
        }
        onError?.(caught);
        return undefined;
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mutator],
  );

  return { mutate, pending, error, reset: () => setError(null) };
}

export default useApi;
