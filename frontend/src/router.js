import { createRouter, createWebHistory } from "vue-router";
import WelcomePage from "./components/WelcomeVue.vue";
import ConsentForm from "./components/ConsentForm.vue";
import Instructions from "./components/InstructionsVue.vue";
import Survey from "./components/SurveyVue.vue";
import Game from "./components/game/GameVue.vue";
import Lobby from "./components/LobbyVue.vue";
import Feedback from "./components/FeedbackVue.vue";
import Debrief from "./components/DebriefVue.vue";
import RoundSetup from "./components/RoundSetup.vue";
import NotFound from "./components/NotFound.vue";
import { api } from "@/api";

const routes = [
  { path: "/", component: WelcomePage },
  { path: "/consent", component: ConsentForm },
  { path: "/instructions", component: Instructions },
  {
    path: "/pre-game-survey",
    name: "PreGameSurvey",
    component: Survey,
  },
  {
    path: "/post-game-survey",
    name: "PostGameSurvey",
    component: Survey,
  },
  { path: "/lobby", component: Lobby },
  { path: "/game", component: Game },
  { path: "/feedback", component: Feedback },
  { path: "/debrief", component: Debrief },
  {
    path: "/round-setup",
    component: RoundSetup,
    beforeEnter: async (to, from, next) => {
      if (await api.isDevelopmentMode()) {
        next();
      } else {
        next("/404");
      }
      return;
    },
  },
  { path: "/404", component: NotFound }, // 404 route
  { path: "/:catchAll(.*)", redirect: "/404" }, // Redirect all undefined routes to 404
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
