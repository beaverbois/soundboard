/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./static/**/*.html",
    "./templates/**/*.html",
    "./app/**/*.py"
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#222',
        'dark-text': '#eee',
        'dark-muted': '#888'
      }
    },
  },
  plugins: [],
}
