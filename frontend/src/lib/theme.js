import { useSyncExternalStore } from 'react';

// The active theme lives on <html data-theme="…">; index.html sets it before
// first paint (saved choice, else the OS preference) so there is no flash.
const STORAGE_KEY = 'aimos-theme';
const listeners = new Set();

const current = () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');

export function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // storage unavailable (private mode): the choice just isn't remembered
  }
  listeners.forEach((fn) => fn());
}

function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function useTheme() {
  const theme = useSyncExternalStore(subscribe, current);
  return { theme, toggle: () => setTheme(theme === 'light' ? 'dark' : 'light') };
}

// Recharts takes colours as props, not classes, so charts read them from here.
const CHART_COLORS = {
  dark: {
    grid: 'rgba(255, 255, 255, 0.08)',
    axis: '#94a3b8',
    tick: '#cbd5e1',
    label: '#e2e8f0',
    tooltip: { backgroundColor: '#12121e', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', color: '#f1f5f9' },
    cursor: 'rgba(255, 255, 255, 0.04)',
    actual: '#3b82f6',
    predicted: '#10b981',
    waiting: '#06b6d4',
    turnaround: '#8b5cf6',
  },
  light: {
    grid: 'rgba(15, 23, 42, 0.08)',
    axis: '#94a3b8',
    tick: '#475569',
    label: '#1e293b',
    tooltip: { backgroundColor: '#ffffff', border: '1px solid rgba(15, 23, 42, 0.12)', borderRadius: '8px', color: '#0f172a', boxShadow: '0 8px 24px -8px rgba(15, 23, 42, 0.2)' },
    cursor: 'rgba(15, 23, 42, 0.04)',
    actual: '#2563eb',
    predicted: '#059669',
    waiting: '#0891b2',
    turnaround: '#7c3aed',
  },
};

export function useChartColors() {
  return CHART_COLORS[useTheme().theme];
}
