import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],

  server: {
    allowedHosts: ["spherexium.earth"],
    proxy: {
      "/simbad": {
        target: "https://simbad.cfa.harvard.edu",
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(/^\/simbad/, "/simbad/sim-tap/sync"),
      },
    },
  },
});
