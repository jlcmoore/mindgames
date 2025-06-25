<template>
  <div class="instructions-container">
    <header class="header">
      <h1>Policy Game</h1>
    </header>

    <main class="main-content">
      <div class="content-wrapper">
        <h2>Lobby</h2>

        <p class="waiting">
          Please wait here while we connect you to another participant...
        </p>
        <p class="waiting">
          You have been waiting for {{ formattedWaitTime }}.
        </p>
        <p class="waiting">
          (Please don't close this page. The system should automatically move
          you on to the end of the study after ten minutes. If you have been
          waiting for <strong>more than ten</strong> minutes or if the counter
          is not changing there has likely been a glitch. Only then, message us
          with the number of rounds you have completed and we will compensate
          you for those rounds.)
        </p>
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
      participantIsReady: false,
      waitingTime: null,
    };
  },
  computed: {
    formattedWaitTime() {
      const total = Math.floor(this.waitingTime || 0);
      const mins = Math.floor(total / 60);
      const secs = total % 60;
      const minLabel = mins === 1 ? "minute" : "minutes";
      const secLabel = secs === 1 ? "second" : "seconds";
      return `${mins} ${minLabel} ${secs} ${secLabel}`;
    },
  },
  mounted() {
    this.participantReady();
  },
  methods: {
    goToInstructions() {
      if (this.participantIsReady) {
        const instructions = localStorage.getItem("instructions");
        if (instructions) {
          this.$router.push("/game");
        } else {
          this.$router.push("/instructions");
        }
      } else {
        console.log("Participant not yet ready");
      }
    },

    async participantReady() {
      console.log("Participant ready. Participant ID:", this.participantId);
      try {
        await api.participantReady(this.participantId);
      } catch (error) {
        if (!error.message.includes("is in a round")) {
          // TODO: it could be here that it is erroring out
          throw error;
        }
      }
      this.participantIsReady = true;
      await this.getCurrentRound();
    },

    async getCurrentRound() {
      let params = null;
      if (await api.isDevelopmentMode()) {
        params = JSON.parse(localStorage.getItem("current_round_params"));
      }
      console.log("Params:", params);
      let response;
      try {
        response = await api.getCurrentRound(this.participantId, params);
      } catch (error) {
        console.error("Failed to get round:", error.message);

        if (
          error.message.includes("has waited too long.") ||
          error.message.includes("has no more rounds to play")
        ) {
          // End the experiment early
          this.$router.push("/post-game-survey");
          return;
        } else if (params !== null) {
          // Don't retry on forced rounds
          localStorage.setItem("current_round_params", null);
          localStorage.setItem("participantId", null);
          this.$router.push("/round-setup");
          return;
        } else {
          console.log("Ignoring error");
          response = null;
        }
      }

      console.log("Round data:", response);

      if (response && response.waiting_time != null) {
        this.waitingTime = response.waiting_time;
        console.log(
          `We have been waiting ${response.waiting_time}s. retrying…`,
        );
        setTimeout(this.getCurrentRound, 2000);
        return;
      }

      if (!response) {
        console.log("No round data received. Retrying...");
        setTimeout(this.getCurrentRound, 2000); // Retry every 2 seconds
        return;
      }

      this.waitingTime = null;
      this.goToInstructions();
    },
  },
};
</script>

<style scoped>
/* Add styling here */
.waiting {
  text-align: center;
}
</style>
