<template>
  <div class="survey-container">
    <header class="header">
      <h1>Policy Game</h1>
    </header>

    <main class="main-content">
      <div class="content-wrapper">
        <h2>Feedback</h2>

        <p>
          Please use the box below to provide any feedback on the experiment.
          What strategy did you use? Did you experience any issues when
          completing the experiment?
        </p>

        <textarea
          v-model="feedback"
          placeholder="Enter your feedback here"
          class="feedback"
        >
        </textarea>

        <button @click="sendToDebrief">Submit</button>
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
    };
  },
  methods: {
    async sendToDebrief() {
      await this.sendFeedback();
      this.$router.push("/debrief");
    },
    async sendFeedback() {
      try {
        await api.sendFeedback(this.participantId, this.feedback);
        console.log("feedback sent: ", this.feedback);
      } catch (error) {
        console.error("Error storing feedback:", error.message);
      }
    },
  },
};
</script>

<style scoped>
.content-wrapper {
  background-color: #fff;
  padding: 40px;
  max-width: 800px;
  width: 90%;
  text-align: center;
  display: flex;
  flex-direction: column;
}
textarea.feedback {
  width: 80%;
  margin: auto;
  height: 150px;
  padding: 12px 20px;
  box-sizing: border-box;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 16px;
  resize: none;
  overflow-y: auto;
  font-family: Source Sans Pro;
  display: block;
  margin-bottom: 20px;
}

button {
  background-color: #c2e9ec;
  color: #12494d;
  border: none;
  padding: 12px 24px;
  font-size: 16px;
  border-radius: 4px;
  cursor: pointer;
  margin-left: auto;
}
</style>
