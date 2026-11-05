/** @type {import('tailwindcss').Config} */
// VoxSentinel design tokens — premium light theme.
// Pale lavender page background, white surfaces, elegant navy typography,
// purple→magenta→coral brand gradient. Dark navy appears ONLY as text/icon
// color, never as a surface. Semantic colors map to genuine app states.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Base surfaces
        canvas: '#F7F5FC',
        surface: '#FFFFFF',
        'surface-2': '#F4F0FA',
        'surface-3': '#EEE8F8',
        line: '#E5DFEF',
        'line-subtle': '#EEE9F5',
        // Typography
        ink: '#17213D',
        'ink-dim': '#536078',
        'ink-faint': '#7B8499',
        // Brand
        accent: '#8652F5',
        'accent-bright': '#A342F0',
        magenta: '#C54FE0',
        coral: '#FF796D',
        lavender: '#EEE5FF',
        'lavender-pale': '#F7F1FF',
        // Semantic (foreground-grade hex; paired with /10 tint backgrounds)
        success: '#21865B',
        warning: '#98600A',
        danger: '#C83F4B',
        info: '#356BC4',
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      // Soft, minimal shadows for light surfaces.
      boxShadow: {
        card: '0 1px 2px rgba(23, 33, 61, 0.04), 0 4px 14px rgba(23, 33, 61, 0.05)',
        'card-hover': '0 2px 4px rgba(23, 33, 61, 0.06), 0 8px 22px rgba(23, 33, 61, 0.08)',
      },
      keyframes: {
        // Processing indicator only (real work in progress), never decoration.
        'pulse-bar': {
          '0%, 100%': { transform: 'scaleY(0.25)' },
          '50%': { transform: 'scaleY(1)' },
        },
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'pulse-bar': 'pulse-bar 1.1s ease-in-out infinite',
        'fade-in': 'fade-in 0.25s ease-out',
      },
    },
  },
  plugins: [],
}
