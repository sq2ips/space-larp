import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],

  server: {
    proxy: {
      "/irsa": {
        target: "https://irsa.ipac.caltech.edu",
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(/^\/irsa/, ""),
      },
    },
  },
});
