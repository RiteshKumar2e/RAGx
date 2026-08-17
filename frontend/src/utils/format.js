/** Display formatting helpers. All of them tolerate null/undefined. */

const EM_DASH = '—';

export function formatNumber(value, options = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return new Intl.NumberFormat('en-US', options).format(value);
}

export function formatCompact(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
}

/** A 0–1 ratio as a fixed-precision decimal (0.842). */
export function formatRatio(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return Number(value).toFixed(digits);
}

export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  const ms = Number(value);
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
}

export function formatUsd(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  const amount = Number(value);
  if (amount === 0) return '$0.00';
  // API costs per query are tiny; show enough precision to be meaningful.
  if (amount < 0.01) return `$${amount.toFixed(5)}`;
  return `$${amount.toFixed(4)}`;
}

export function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) return EM_DASH;
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatDecimal(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return Number(value).toFixed(digits);
}

/** Dispatch on the `format` key used in EVAL_METRICS. */
export function formatMetric(value, format) {
  switch (format) {
    case 'ratio':
      return formatRatio(value);
    case 'percent':
      return formatPercent(value);
    case 'ms':
      return formatMs(value);
    case 'usd':
      return formatUsd(value);
    case 'decimal':
      return formatDecimal(value);
    case 'number':
      return formatNumber(value, { maximumFractionDigits: 0 });
    default:
      return value ?? EM_DASH;
  }
}

export function formatDate(value, options = {}) {
  if (!value) return EM_DASH;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return EM_DASH;
  return new Intl.DateTimeFormat('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    ...options,
  }).format(date);
}

export function formatDateTime(value) {
  return formatDate(value, { hour: '2-digit', minute: '2-digit' });
}

export function formatRelativeTime(value) {
  if (!value) return EM_DASH;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return EM_DASH;

  const seconds = Math.round((Date.now() - date.getTime()) / 1000);
  if (seconds < 45) return 'just now';

  const units = [
    ['minute', 60],
    ['hour', 3600],
    ['day', 86400],
    ['week', 604800],
    ['month', 2592000],
    ['year', 31536000],
  ];
  const formatter = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
  for (let i = units.length - 1; i >= 0; i -= 1) {
    const [unit, size] = units[i];
    if (seconds >= size) return formatter.format(-Math.round(seconds / size), unit);
  }
  return formatter.format(-Math.round(seconds / 60), 'minute');
}

export function truncate(text, maxLength = 140) {
  if (!text) return '';
  return text.length <= maxLength ? text : `${text.slice(0, maxLength - 1).trimEnd()}…`;
}

export function titleCase(text) {
  if (!text) return '';
  return text
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

/** "Research_Paper.pdf · p.7 · Methodology" from a citation-shaped object. */
export function citationLocation(item) {
  if (!item) return '';
  const parts = [];
  if (item.page) {
    parts.push(item.page_end && item.page_end !== item.page ? `pp.${item.page}–${item.page_end}` : `p.${item.page}`);
  }
  if (item.section) parts.push(item.section);
  if (item.figure) parts.push(item.figure);
  if (item.table) parts.push(item.table);
  return parts.join(' · ');
}

export function fileExtension(filename) {
  if (!filename || !filename.includes('.')) return '';
  return `.${filename.split('.').pop().toLowerCase()}`;
}
