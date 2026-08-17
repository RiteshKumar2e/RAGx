import { useEffect, useRef } from 'react';

/**
 * Calls `callback` on an interval while `enabled` is true.
 *
 * Used for live document-processing status and in-flight evaluation runs.
 * Polling pauses while the tab is hidden so a backgrounded tab does not keep
 * hitting the API, and resumes (with an immediate tick) when it becomes visible.
 */
export function usePolling(callback, intervalMs = 3000, enabled = true) {
  const savedCallback = useRef(callback);
  savedCallback.current = callback;

  useEffect(() => {
    if (!enabled || !intervalMs) return undefined;

    let timer = null;

    const tick = () => {
      if (document.visibilityState === 'visible') savedCallback.current();
    };

    const start = () => {
      if (timer === null) timer = setInterval(tick, intervalMs);
    };
    const stop = () => {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        savedCallback.current();
        start();
      } else {
        stop();
      }
    };

    start();
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      stop();
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [intervalMs, enabled]);
}

export default usePolling;
