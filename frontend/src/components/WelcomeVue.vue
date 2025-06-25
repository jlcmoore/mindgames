<template>
  <div class="welcome-container">
    <header class="header">
      <h1>Policy Game</h1>
    </header>

    <!-- Main Content: Centered text with white background -->
    <main class="main-content">
      <div class="content-wrapper">
        <h2>Welcome</h2>
        <p>Thank you for your interest in participating in our experiment!</p>
        <p>
          In this experiment, you will first answer a short survey and then have
          several short conversations with other players about different
          policies.
        </p>
        <p>
          Note!
          <strong
            >Using LLMs and Generative AI tools is strictly prohibited</strong
          >
          and will result in your exclusion from the study. (Don't try to police
          other players, though.)
        </p>
        <button @click="goToSurvey">Continue</button>
      </div>
    </main>
  </div>
</template>

<script>
import { api } from "@/api";

export default {
  async mounted() {
    // Wait for both mode and survey questions to be fetched.
    await Promise.all([this.checkDevelopmentMode()]);
  },
  methods: {
    async checkDevelopmentMode() {
      try {
        const response = await api.development_mode();
        this.developmentMode = response.development_mode;
        console.log("Development Mode:", this.developmentMode);
      } catch (error) {
        console.error(
          "Failed to fetch development mode status:",
          error.message,
        );
        // Default to false if there's an error
        this.developmentMode = false;
      }
    },
    goToSurvey() {
      localStorage.clear();
      let prolificId = new URLSearchParams(window.location.search).get(
        "PROLIFIC_PID",
      );
      // Generate an id for the participant if not from prolific
      prolificId =
        prolificId ||
        Math.random().toString(36).substring(2, 15) +
          Math.random().toString(36).substring(2, 15);
      console.log("Prolific ID:", prolificId);
      localStorage.setItem("prolificId", prolificId);

      // Only show the consent if not in dev mode
      if (this.developmentMode) {
        this.$router.push("/pre-game-survey");
      } else {
        this.$router.push("/consent");
      }
    },
  },
};
</script>

<style scoped></style>
