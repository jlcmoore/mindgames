<template>
  <div class="survey-container">
    <header class="header">
      <h1>Policy Game</h1>
    </header>

    <main class="main-content">
      <div class="content-wrapper">
        <h2>Debrief</h2>

        <p>You have completed the experiment.</p>

        <p>
          You interacted with {{ humanConversations }} real human participants
          during the experiment. You interacted with
          {{ totalRounds - humanConversations }} non-human players.
        </p>

        <p>Thank you for your participation.</p>

        <p>
          Click the button below to be redirected to Prolific and verify your
          completion.
        </p>

        <button @click="sendToProlific">Send to Prolific</button>
      </div>
    </main>
  </div>
</template>

<script>
import { api } from "@/api";

export default {
  data() {
    return {
      participantId: localStorage.getItem("participantId"), // Get the participant ID from local storage
      conversation_history: [],
      humanConversations: null,
      totalRounds: null,
    };
  },
  mounted() {
    this.getRoundsData();
  },
  methods: {
    async sendToProlific() {
      if (this.completionCode) {
        window.location.href = `https://app.prolific.co/submissions/complete?cc=${this.completionCode}`;
      } else {
        console.error("Completion code is not available.");
      }
    },
    async getRoundsData() {
      try {
        // Get round info
        const roundData = await api.getCurrentRound(this.participantId);

        // Update scenario and round ID
        this.scenarioText = roundData.prompt;
        this.currentRoundId = roundData.round_id;
        this.turnLimit = roundData.game_data.turn_limit;
      } catch (error) {
        console.error("Error getting current round:", error.message);
        console.log(error);
      }

      try {
        // Get rounds data
        const completionData = await api.getParticipantRounds(
          this.participantId,
        );
        console.log("Rounds data:", completionData);
        this.humanConversations = completionData.num_human_conversations;
        // this.currentRound = completionData.rounds_completed;
        this.totalRounds = completionData.rounds_completed;
        // this.roundsRemaining = completionData.rounds_remaining;
        this.completionCode = completionData.completion_code;
      } catch (error) {
        console.error("Error debriefing game:", error.message);
        console.log(error);
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>

<style scoped>
/* Add styling here */
</style>
