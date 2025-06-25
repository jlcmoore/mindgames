<template>
  <div class="survey-container">
    <header class="header">
      <h1>Policy Game</h1>
    </header>

    <main class="main-content">
      <div v-if="!autoSubmit" class="content-wrapper">
        <h2>
          <em v-if="isPostGame">Post Game</em>
          Survey
        </h2>
        <p>
          Here you will read several hypothetical scenarios involving potential
          trade-offs.
          <strong
            >For each scenario, please indicate your preferences toward three
            different attributes</strong
          >
          by selecting "Increased a lot", "Increased a little", "Stayed the
          same", "Decreased a little", or "Decreased a lot". Please answer these
          questions as if they were independent. (E.g., you could answer
          "increased a lot" for all three of the attributes of a scenario.)
        </p>

        <div v-if="surveyQuestions.length > 0">
          <form @submit.prevent="submitSurvey">
            <!-- loop over each cover‐story group -->
            <div
              v-for="(group, gIndex) in groupedSurvey"
              :key="group.coverStory"
              class="cover-story-group"
            >
              <!-- show the cover story once -->
              <p class="cover-story">
                <strong>{{ gIndex + 1 }}: {{ group.coverStory }}</strong>
              </p>

              <!-- now loop over that group’s questions -->
              <div
                v-for="(question, qIndex) in group.questions"
                :key="question.id"
                class="survey-question"
              >
                <p>
                  <strong>{{ gIndex + 1 }}.{{ qIndex + 1 }}:</strong>
                  {{ question.statement }}
                </p>
                <div class="likert-scale">
                  <label
                    v-for="n in likertScaleValues"
                    :key="n"
                    class="likert-option"
                  >
                    <input
                      v-model.number="surveyResponses[question.id]"
                      type="radio"
                      :name="question.id"
                      :value="n"
                      :required="!developmentMode"
                    />
                    {{ likertLabels[n.toString()] }}
                  </label>
                </div>
              </div>
            </div>

            <button type="submit">
              {{ isPostGame ? "Continue to Feedback" : "Continue to Lobby" }}
            </button>
          </form>
        </div>
        <div v-else>Loading survey questions...</div>
      </div>
      <div v-else class="auto-submit-message">
        <p>Submitting survey...</p>
      </div>
    </main>
  </div>
</template>

<script>
import { api } from "@/api";

export default {
  data() {
    return {
      surveyQuestions: [],
      surveyResponses: {},
      prolificId: localStorage.getItem("prolificId"),
      participantInitialized: false,
      participantId: null,
      likertScaleValues: [2, 1, 0, -1, -2],
      likertLabels: {
        2: "Increased a lot",
        1: "Increased a little",
        0: "Stayed the same",
        "-1": "Decreased a little",
        "-2": "Decreased a lot",
      },
      developmentMode: false,
      isPostGame: false,
      autoSubmit: false, // flag to track if survey is being auto-submitted
    };
  },
  computed: {
    // group questions by cover_story
    groupedSurvey() {
      const groups = {};
      this.surveyQuestions.forEach((q) => {
        const story = q.cover_story || "General";
        if (!groups[story]) groups[story] = [];
        groups[story].push(q);
      });
      // turn into an array for easier v-for
      return Object.entries(groups).map(([coverStory, questions]) => ({
        coverStory,
        questions,
      }));
    },
  },
  created() {
    // Determine if this is post-game survey based on route
    this.isPostGame = this.$route.path === "/post-game-survey";
  },
  async mounted() {
    // Wait for both mode and survey questions to be fetched.
    await Promise.all([this.checkDevelopmentMode(), this.fetchSurvey()]);

    // Check if the conditions for automatic submission are met
    const paramsJson = localStorage.getItem("current_round_params");
    if (paramsJson) {
      const params = JSON.parse(paramsJson);
      if (this.developmentMode && params.targets_values === false) {
        console.log(
          "Auto-submitting blank survey because targets_values is false in development mode.",
        );
        // Set the flag to hide the survey content, if desired.
        this.autoSubmit = true;
        // Proceed to submission immediately.
        this.submitSurvey();
      }
    }
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
    async fetchSurvey() {
      try {
        const response = await api.survey();
        // Assuming the API returns an array of survey questions
        this.surveyQuestions = response;
        console.log("Survey questions:", this.surveyQuestions);
      } catch (error) {
        console.error("Failed to fetch survey questions:", error.message);
      }
    },
    async submitSurvey() {
      // In development mode, if we are auto-submitting, we can skip the "all answered" check.
      if (!this.developmentMode) {
        const allQuestionsAnswered = this.surveyQuestions.every(
          (question) => this.surveyResponses[question.id] !== undefined,
        );
        if (!allQuestionsAnswered) {
          alert("Please answer all the survey questions.");
          return;
        }
      }

      try {
        // Prepare survey responses in the required format
        // Include only the answered questions
        const survey_responses = this.surveyQuestions
          .filter((question) => this.surveyResponses[question.id] !== undefined)
          .map((question) => ({
            ...question,
            rating: this.surveyResponses[question.id],
          }));

        if (this.isPostGame) {
          // Call post-game survey API
          await api.submitPostGameSurvey(
            localStorage.getItem("participantId"),
            survey_responses,
          );
          this.$router.push("/feedback");
        } else {
          // Call pre-game survey API

          // Initialize participant with survey responses
          const response = await api.initializeParticipant(
            this.prolificId,
            survey_responses,
          );
          console.log("Participant initialized:", response);

          // Store participantId
          this.participantId = response.participant_id;
          localStorage.setItem("participantId", this.participantId);
          this.participantInitialized = true;

          // Navigate to Lobby
          this.$router.push("/lobby");
        }
      } catch (error) {
        console.error("Failed to send survey:", error.message);
        alert(
          "An error occurred while submitting the survey. Please try again.",
        );
      }
    },
  },
};
</script>

<style scoped>
/* Add any necessary styles */
.likert-scale {
  display: flex;
  justify-content: space-between;
  margin-bottom: 1em;
}
.likert-option {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.cover-story {
  margin-top: 3em;
  font-size: 1.1em;
}
.auto-submit-message {
  text-align: center;
  font-size: 1.2em;
  padding: 2em;
}
</style>
