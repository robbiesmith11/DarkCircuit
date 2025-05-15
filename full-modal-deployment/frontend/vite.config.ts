import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  css: {
    postcss: './postcss.config.js', // Ensure PostCSS uses the correct config file
  },
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
});