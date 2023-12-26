import { defineConfig } from "vite"
import { svelte } from "@sveltejs/vite-plugin-svelte"

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [svelte()],
    server: {
        proxy: {
            "/api/spotify": {
                target: `${import.meta.env.VITE_BACKEND_URL}/api/spotify`,
                changeOrigin: true,
            },
            "/api/spotify-account": {
                target: `${import.meta.env.VITE_BACKEND_URL}/api/spotify-account`,
                changeOrigin: true,
            },
        },
        cors: false
    }
})