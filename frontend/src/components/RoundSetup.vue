<!-- RoundSetup.vue -->
<template>
  <div class="round-setup-container">
    <header class="header">
      <h1>Policy Game</h1>
    </header>

    <main class="main-content">
      <div class="content-wrapper">
        <h2>Customize Your Game Round</h2>

        <form @submit.prevent="initialize">
          <div>
            <label>
              External (fake Prolific) Participant ID:
              <input v-model="prolificId" type="text" required />
            </label>
          </div>

          <div>
            <label>
              <input v-model="isTarget" type="radio" :value="true" />
              Play as Target
            </label>
            <label>
              <input v-model="isTarget" type="radio" :value="false" />
              Play as Persuader
            </label>
            <label>
              <input v-model="isTarget" type="radio" :value="null" />
              No Preference
            </label>
          </div>

          <div>
            <label for="gameModelType">Game Model Type:</label>
            <select id="gameModelType" v-model="gameModelType">
              <option :value="null">-- Select Game Model Type --</option>
              <option value="solution">Solution</option>
              <option value="can-win">Can-win</option>
              <option value="always-win">Always-win</option>
              <option value="never-win">Never-win</option>
              <!-- Add other game model types here -->
            </select>
          </div>

          <div>
            <label>
              LLM Target:
              <input v-model="llmTarget" type="text" />
            </label>
          </div>

          <div>
            <label>
              LLM Persuader:
              <input v-model="llmPersuader" type="text" />
            </label>
          </div>

          <div>
            <label>
              Scenario ID:
              <input v-model="scenarioID" type="text" />
            </label>
          </div>

          <div>
            <label>
              Game Model ID:
              <input v-model="gameModelID" type="text" />
            </label>
          </div>

          <!-- Replace checkboxes with select inputs to allow null, true, or false -->
          <div>
            <label>
              Use Target's Own Values:
              <select v-model="useTargetsOwnValues">
                <option :value="null">--</option>
                <option :value="true">Yes</option>
                <option :value="false">No</option>
              </select>
            </label>
          </div>

          <div>
            <label>
              Reveal Motivation:
              <select v-model="revealMotivation">
                <option :value="null">--</option>
                <option :value="true">Yes</option>
                <option :value="false">No</option>
              </select>
            </label>
          </div>

          <div>
            <label>
              Reveal Belief:
              <select v-model="revealBelief">
                <option :value="null">--</option>
                <option :value="true">Yes</option>
                <option :value="false">No</option>
              </select>
            </label>
          </div>

          <div>
            <label>
              Allow Lying:
              <select v-model="allowLying">
                <option :value="null">--</option>
                <option :value="true">Yes</option>
                <option :value="false">No</option>
              </select>
            </label>
          </div>

          <!-- Add more parameters as needed -->

          <button type="submit">Start Game</button>
        </form>

        <div v-if="errorMessage" class="error">
          {{ errorMessage }}
        </div>
      </div>
    </main>
  </div>
</template>

<script>
export default {
  data() {
    return {
      prolificId: null,
      isTarget: null,
      gameModelType: "solution",
      llmTarget: null,
      llmPersuader: null,
      scenarioID: null,
      gameModelID: null,
      useTargetsOwnValues: null,
      revealMotivation: null,
      revealBelief: null,
      allowLying: null,
      errorMessage: "",
      participantInitialized: false,
      participantIsReady: false,
    };
  },
  mounted() {
    // Extract query parameters.
    const query = this.$route.query;
    const queryProvided = Object.keys(query).length > 0;
    let autoProceed = false;

    if (query.prolificId) {
      this.prolificId = query.prolificId;
      autoProceed = true;
    }
    if (query.isTarget !== undefined) {
      if (query.isTarget === "true") this.isTarget = true;
      else if (query.isTarget === "false") this.isTarget = false;
      else this.isTarget = null;
      autoProceed = true;
    }
    if (query.gameModelType) {
      this.gameModelType = query.gameModelType;
      autoProceed = true;
    }
    if (query.llmTarget) {
      this.llmTarget = query.llmTarget;
      autoProceed = true;
    }
    if (query.llmPersuader) {
      this.llmPersuader = query.llmPersuader;
      autoProceed = true;
    }
    if (query.scenarioID) {
      this.scenarioID = query.scenarioID;
      autoProceed = true;
    }
    if (query.gameModelID) {
      this.gameModelID = query.gameModelID;
      autoProceed = true;
    }
    if (query.useTargetsOwnValues !== undefined) {
      if (query.useTargetsOwnValues === "true") this.useTargetsOwnValues = true;
      else if (query.useTargetsOwnValues === "false")
        this.useTargetsOwnValues = false;
      else this.useTargetsOwnValues = null;
      autoProceed = true;
    }
    if (query.revealMotivation !== undefined) {
      if (query.revealMotivation === "true") this.revealMotivation = true;
      else if (query.revealMotivation === "false")
        this.revealMotivation = false;
      else this.revealMotivation = null;
      autoProceed = true;
    }
    if (query.revealBelief !== undefined) {
      if (query.revealBelief === "true") this.revealBelief = true;
      else if (query.revealBelief === "false") this.revealBelief = false;
      else this.revealBelief = null;
      autoProceed = true;
    }
    if (query.allowLying !== undefined) {
      if (query.allowLying === "true") this.allowLying = true;
      else if (query.allowLying === "false") this.allowLying = false;
      else this.allowLying = null;
      autoProceed = true;
    }

    // If URL parameters were provided but no prolificId then auto generate one.
    if (queryProvided && (!this.prolificId || this.prolificId.trim() === "")) {
      this.prolificId =
        Math.random().toString(36).substring(2, 15) +
        Math.random().toString(36).substring(2, 15);
      autoProceed = true;
    }

    // If any relevant URL parameter was provided, use $nextTick to ensure all bindings are updated
    // before automatically calling initialize().
    if (autoProceed) {
      this.$nextTick(() => {
        this.initialize();
      });
    }
  },
  methods: {
    async initialize() {
      // Clear any previous session state
      localStorage.clear();

      // If somehow still missing, generate a new participant id before proceeding.
      if (!this.prolificId || this.prolificId.trim() === "") {
        this.prolificId =
          Math.random().toString(36).substring(2, 15) +
          Math.random().toString(36).substring(2, 15);
      }

      // Build the parameters object, ignoring empty values.
      const params = {};
      if (this.isTarget !== null) params.is_target = this.isTarget;
      if (this.gameModelType !== null)
        params.game_model_type = this.gameModelType;
      if (this.llmTarget !== null && this.llmTarget.trim() !== "")
        params.llm_target = this.llmTarget;
      if (this.llmPersuader !== null && this.llmPersuader.trim() !== "")
        params.llm_persuader = this.llmPersuader;
      if (this.scenarioID !== null && this.scenarioID.trim() !== "")
        params.scenario_id = this.scenarioID;
      if (this.gameModelID !== null && this.gameModelID.trim() !== "")
        params.game_model_id = this.gameModelID;
      if (this.useTargetsOwnValues !== null)
        params.targets_values = this.useTargetsOwnValues;
      if (this.revealMotivation !== null)
        params.reveal_motivation = this.revealMotivation;
      if (this.revealBelief !== null) params.reveal_belief = this.revealBelief;
      if (this.allowLying !== null) params.allow_lying = this.allowLying;

      // Store the participant ID and game parameters.
      localStorage.setItem("prolificId", this.prolificId);
      localStorage.setItem("current_round_params", JSON.stringify(params));

      // Route to the pre‑game survey.
      this.$router.push("/pre-game-survey");
    },
  },
};
</script>

<style scoped>
.round-setup-container {
  /* Add styling here */
}

.error {
  color: red;
  margin-top: 10px;
}
</style>
