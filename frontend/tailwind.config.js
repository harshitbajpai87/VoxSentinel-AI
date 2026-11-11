/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#FCF9F1', surface: '#FFFEF9', 'surface-2': '#F3EDDF', 'surface-3': '#E8DFCF',
        line: '#DDD3C3', 'line-subtle': '#EAE3D6',
        ink: '#352D29', 'ink-dim': '#6C6056', 'ink-faint': '#7B6B5C',
        accent: '#AD432D', 'accent-bright': '#A23D2B', magenta: '#804365', coral: '#B74F35',
        lavender: '#EDDFD7', 'lavender-pale': '#FAF0E9', 'soft-coral': '#F5DFD0',
        success: '#526733', warning: '#886111', danger: '#AC383B', info: '#804365',
      },
      fontFamily: {
        sans: ['Segoe UI', 'Arial', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      boxShadow: {
        card: '0 2px 6px rgba(83, 57, 32, 0.025)',
        'card-hover': '0 8px 22px rgba(83, 57, 32, 0.07)',
      },
      keyframes: {
        'pulse-bar': { '0%, 100%': { transform: 'scaleY(0.25)' }, '50%': { transform: 'scaleY(1)' } },
        'fade-in': { from: { opacity: '0', transform: 'translateY(5px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
      },
      animation: { 'pulse-bar': 'pulse-bar 1.1s ease-in-out infinite', 'fade-in': 'fade-in 0.3s ease-out' },
    },
  },
  plugins: [],
}
