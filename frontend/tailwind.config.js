/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        cyber: {
          dark: "#080C14",
          card: "#0F172A",
          cardBorder: "#1E293B",
          accent: "#38BDF8",
          neonGreen: "#10B981",
          neonRed: "#EF4444",
          neonYellow: "#F59E0B",
          neonPurple: "#8B5CF6",
          muted: "#64748B"
        }
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"]
      }
    },
  },
  plugins: [],
}
