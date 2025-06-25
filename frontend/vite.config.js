import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { resolve } from "path"; // Import path resolution
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  base: "./", // Ensures relative paths in the output HTML
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"), // This sets up an alias for the src directory
    },
  },
});
