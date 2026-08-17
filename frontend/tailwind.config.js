/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Deep navy for typography and chrome.
        ink: {
          50: '#f5f7fa',
          100: '#e9edf3',
          200: '#cfd8e3',
          300: '#a8b8cc',
          400: '#7791b0',
          500: '#547296',
          600: '#3f5a7c',
          700: '#344964',
          800: '#2d3e54',
          900: '#1b2739',
          950: '#111a27',
        },
        // Primary accent — analytical blue.
        brand: {
          50: '#eef4ff',
          100: '#dae6ff',
          200: '#bdd3ff',
          300: '#90b6ff',
          400: '#5b8efb',
          500: '#3567f0',
          600: '#2049d9',
          700: '#1c3bb0',
          800: '#1c348c',
          900: '#1c316f',
          950: '#141f44',
        },
        // Secondary accent — used for strategy chips and the graph.
        violet: {
          50: '#f5f3ff',
          100: '#ede9fe',
          200: '#ddd6fe',
          300: '#c4b5fd',
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
          700: '#6d28d9',
          800: '#5b21b6',
          900: '#4c1d95',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(17, 26, 39, 0.04), 0 1px 3px rgba(17, 26, 39, 0.06)',
        'card-hover': '0 4px 12px rgba(17, 26, 39, 0.08), 0 2px 4px rgba(17, 26, 39, 0.04)',
        panel: '0 8px 30px rgba(17, 26, 39, 0.08)',
      },
      borderRadius: {
        xl: '0.75rem',
        '2xl': '1rem',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-500px 0' },
          '100%': { backgroundPosition: '500px 0' },
        },
        'pulse-subtle': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.55' },
        },
      },
      animation: {
        'fade-in': 'fade-in 180ms ease-out',
        shimmer: 'shimmer 1.4s linear infinite',
        'pulse-subtle': 'pulse-subtle 1.8s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
