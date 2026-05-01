export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        graph: {
          bg: "#1d1f23",
          panel: "#23262b",
          panelAlt: "#1a1c20",
          line: "#30343b",
          text: "#f3f5f7",
          muted: "#97a1af",
          cyan: "#6bdcff",
          gold: "#f4c06c",
          green: "#79f2a0",
          danger: "#ff7a90"
        }
      },
      boxShadow: {
        graph: "0 18px 50px rgba(0,0,0,0.35)",
        node: "0 0 0 1px rgba(255,255,255,0.06), 0 0 18px rgba(107,220,255,0.3)"
      },
      fontFamily: {
        display: ["'Avenir Next'", "'Segoe UI'", "'Helvetica Neue'", "Arial", "sans-serif"],
        mono: ["'SFMono-Regular'", "Menlo", "Monaco", "Consolas", "monospace"]
      }
    }
  },
  plugins: []
};
