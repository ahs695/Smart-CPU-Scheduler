/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0c',
        card: '#141417',
        primary: '#3b82f6',
        secondary: '#10b981',
        accent: '#8b5cf6',
        mtx1: 'var(--mtx1)',
        mtx2: 'var(--mtx2)',
        mtx3: 'var(--mtx3)',
        mtx4: 'var(--mtx4)',
        mtx5: 'var(--mtx5)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
