/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        cyan: '#00FFF0',
        bgCyan: '#00ced1',
      },
    },
  },
  plugins: [],
};