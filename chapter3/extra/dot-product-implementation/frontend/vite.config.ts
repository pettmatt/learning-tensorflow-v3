import { defineConfig } from "vite"
import { svelte } from "@sveltejs/vite-plugin-svelte"

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [svelte()],
    server: {
        proxy: {
            "/api/spotify": {
                target: `http://localhost:3010/spotify`,
                changeOrigin: true,
            },
            // "/api/spotify-account": {
            //     target: `${import.meta.env.VITE_BACKEND_URL}/api/spotify-account`,
            //     changeOrigin: true,
            // },
        },
        cors: true
    }
})