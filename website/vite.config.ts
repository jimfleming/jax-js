import tailwindcss from "@tailwindcss/vite";
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  plugins: [
    sveltekit(),
    tailwindcss(),
    process.env.BASIC_SSL ? basicSsl() : null,
  ],
  optimizeDeps: {
    // https://github.com/vitejs/vite/issues/14609
    exclude: ["@rollup/browser"],
  },
});
