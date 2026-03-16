/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#f0f4ff',
          100: '#dde8ff',
          500: '#4f6ef7',
          600: '#3a57e8',
          700: '#2c44cc',
        },
      },
    },
  },
  plugins: [],
}
