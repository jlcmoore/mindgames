import { createApp } from "vue";
import App from "./AppVue.vue";
import router from "./router";
import "./assets/tailwind.css";

const app = createApp(App);
app.use(router); // Tell Vue to use the router
app.mount("#app");

window.vueApp = app; // Expose the app instance for debugging
