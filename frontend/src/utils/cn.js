/**
 * Minimal class-name joiner.
 *
 * Accepts strings, arrays and `{class: condition}` objects, skipping falsy
 * values. A dependency-free stand-in for `clsx`.
 */
export function cn(...inputs) {
  const out = [];

  const walk = (value) => {
    if (!value) return;
    if (typeof value === 'string' || typeof value === 'number') {
      out.push(String(value));
    } else if (Array.isArray(value)) {
      value.forEach(walk);
    } else if (typeof value === 'object') {
      Object.entries(value).forEach(([key, condition]) => {
        if (condition) out.push(key);
      });
    }
  };

  inputs.forEach(walk);
  return out.join(' ');
}

export default cn;
