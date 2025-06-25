<template>
  <div class="instructions-container">
    <header class="header">
      <h1>Policy Game</h1>
    </header>

    <main class="main-content">
      <div class="content-wrapper">
        <h2>Instructions</h2>

        <!-- Display the dynamically fetched instructions -->
        <div v-if="instructions" v-html="instructions"></div>
        <div v-else>Loading instructions...</div>

        <button :disabled="!instructions || !canProceed" @click="goToGame">
          {{ buttonText }}
        </button>
      </div>
    </main>
  </div>
</template>

<script>
import { api } from "@/api";

export default {
  data() {
    return {
      participantId: localStorage.getItem("participantId"),
      instructions: null,
      timeRemaining: 30,
      canProceed: false,
    };
  },
  computed: {
    buttonText() {
      if (!this.instructions) {
        return "Loading...";
      }
      return this.canProceed
        ? "Continue to Game"
        : `Please wait ${this.timeRemaining} seconds`;
    },
  },
  mounted() {
    // Fetch instructions if participantId exists
    if (this.participantId) {
      this.fetchInstructions();
    } else {
      console.error("No participantId found in localStorage.");
      // Optionally redirect back to the survey or display an error message
    }
  },
  methods: {
    startTimer(dev_mode = false) {
      const timer = setInterval(() => {
        if (this.timeRemaining > 0 && !dev_mode) {
          this.timeRemaining--;
        } else {
          this.canProceed = true;
          clearInterval(timer);
        }
      }, 1000);
    },
    goToGame() {
      if (this.instructions) {
        this.$router.push("/game");
      } else {
        console.log("Instructions not loaded yet.");
      }
    },
    async fetchInstructions() {
      try {
        const instructionsResponse = await api.getParticipantInstructions(
          this.participantId,
        );
        // Assuming the instructions are returned as HTML string
        this.instructions = instructionsResponse;
        localStorage.setItem("instructions", instructionsResponse);
        console.log(instructionsResponse);
        // Start the timer when component is mounted
        const dev_mode = await api.isDevelopmentMode();
        this.startTimer(dev_mode);
      } catch (error) {
        console.error("Failed to fetch instructions:", error.message);
      }
    },
  },
};
</script>

<style scoped>
button:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}
</style>
