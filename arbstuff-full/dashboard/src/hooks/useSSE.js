import { useEffect, useRef, useState } from 'react';

/**
 * useSSE — subscribes to a Server-Sent Events endpoint,
 * auto-parses JSON messages, and auto-reconnects on error
 * with exponential backoff capped at 8s.
 */
export function useSSE(url) {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);
  const retryRef = useRef(0);
  const esRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      try {
        const es = new EventSource(url);
        esRef.current = es;

        es.onopen = () => {
          if (cancelled) return;
          setConnected(true);
          retryRef.current = 0;
        };

        es.onmessage = (ev) => {
          if (cancelled) return;
          try {
            const parsed = JSON.parse(ev.data);
            setData(parsed);
          } catch {
            /* ignore parse errors */
          }
        };

        es.onerror = () => {
          setConnected(false);
          es.close();
          if (cancelled) return;
          retryRef.current = Math.min(retryRef.current + 1, 5);
          const delay = Math.min(1000 * 2 ** retryRef.current, 8000);
          timerRef.current = setTimeout(connect, delay);
        };
      } catch {
        const delay = 2000;
        timerRef.current = setTimeout(connect, delay);
      }
    };

    connect();

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      if (esRef.current) esRef.current.close();
    };
  }, [url]);

  return { data, connected };
}
