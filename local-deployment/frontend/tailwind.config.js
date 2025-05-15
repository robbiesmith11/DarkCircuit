/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: { 
        MonomaniacOne: ['"MonomaniacOne"', 'sans-serif'],
      },
      colors: {
        cyan: '#00FFF0',
        bgCyan: '#00ced1',
        form: "rgba(25, 57, 55, 40)",
        chat: "rgba(0, 206, 209, 0.2)",
      },
    },
  },
  plugins: [],
};