<template>
  <div v-if="show" class="overlay">
    <div class="overlay-content">
      <div v-if="status == 'play'">
        <h2>Decision</h2>
        <p>Given what you now know, which proposal would you choose?</p>

        <div class="choice-buttons">
          <div v-for="proposal in proposals" :key="proposal">
            <button
              :class="{ selected: targetChoice === proposal }"
              @click="targetChoice = proposal"
            >
              {{ proposal }}
            </button>
          </div>
        </div>

        <!-- Decide/Cancel buttons -->
        <div id="decide-button-container">
          <button v-if="turnsRemaining" id="close-button" @click="closeOverlay">
            Cancel
          </button>
          <button id="decide-button" @click="makeChoice">Decide</button>
        </div>
      </div>

      <div v-else-if="status == 'waiting'">
        <h2>Decision</h2>
        <p>Submitting decision...</p>
      </div>

      <div v-else-if="status == 'result'">
        <div v-if="targetPerfectChoice">
          <div v-if="targetPerfectChoice == targetChoice" class="success">
            <h2>Success!</h2>
            <p class="decide-subtitle">You selected the optimal policy.</p>
          </div>
          <div v-else class="failure">
            <h2>Failure!</h2>
            <p class="decide-subtitle">
              You did not select the optimal policy.
            </p>
          </div>
          <!-- <p>
            Your ideal policy was: <b>{{ targetPerfectChoice }}</b>
          </p> -->
        </div>
        <!-- <p>
          Your partner's preferred policy was: <b>{{ persuaderChoice }}</b>
        </p> -->
        <p>
          You selected: <b>{{ targetChoice }}</b>
        </p>
        <button class="continue-btn" @click="newGame">Continue</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    show: {
      type: Boolean,
      required: true,
    },
    status: {
      type: String,
      required: true,
    },
    targetPerfectChoice: {
      type: String,
      default: null,
      required: false,
    },
    persuaded: {
      type: Boolean,
      required: true,
    },
    persuaderChoice: {
      type: String,
      required: true,
    },
    proposals: {
      type: Array,
      required: true,
    },
    turnsRemaining: {
      type: Boolean,
      required: true,
    },
  },
  data() {
    return {
      targetChoice: null, // Tracks selected choice (A, B, or C)
    };
  },
  methods: {
    closeOverlay() {
      this.$emit("close");
    },
    makeChoice() {
      if (this.targetChoice == null) return;
      this.$emit("makeChoice", this.targetChoice);
    },
    newGame() {
      this.$emit("newGame");
    },
  },
};
</script>

<style scoped>
.overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.overlay-content {
  background: white;
  padding: 20px;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
}

#decide-button-container {
  display: flex;
  justify-content: space-between;
  margin-top: 20px;
}

#close-button,
#decide-button {
  padding: 12px 24px;
  font-size: 16px;
  border-radius: 4px;
  cursor: pointer;
  width: 100%;
  margin-left: 2px;
  margin-right: 2px;
}

#close-button {
  background-color: #f44336;
  color: white;
}

#decide-button {
  background-color: #4caf50;
  color: white;
}

#decide-button:disabled {
  background-color: grey;
  cursor: not-allowed;
}

.choice-buttons {
  display: flex;
  justify-content: space-between;
  margin-bottom: 20px;
}

.choice-buttons button {
  background-color: lightgrey;
  padding: 12px 24px;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
}

.choice-buttons button:hover {
  background-color: darkgrey;
}

.choice-buttons button.selected {
  background-color: rgb(0, 101, 216);
  color: white;
}

.choice-buttons button.selected:hover {
  background-color: rgb(17, 0, 255);
  color: white;
}

.continue-btn {
  margin-left: auto;
}

.success h2 {
  color: #4caf50;
}

.failure h2 {
  color: #f44336;
}

.decide-subtitle {
  font-size: 1.1em;
  margin-bottom: 20px;
  color: #444444;
  font-style: italic;
}
</style>
