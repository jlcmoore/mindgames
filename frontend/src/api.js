import axios from "axios";

// Create an instance of axios to centralize common configurations
const apiClient = axios.create({
  baseURL: "/", // Base URL for your API
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 60000, // Set a timeout of 60 seconds
});

// Add an interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response, // Let successful responses pass through
  (error) => {
    // Check if the error is from the API
    if (error.response) {
      console.error(
        `API Error: ${error.response.status} - ${error.response.data.detail || error.response.statusText}`,
      );
      return Promise.reject({
        message:
          error.response.data.detail ||
          "An error occurred while processing the request.",
        status: error.response.status,
      });
    } else if (error.request) {
      // No response received from server
      console.error("Network Error: No response received from the server.");
      return Promise.reject({
        message:
          "No response from server. Please check your network connection.",
      });
    } else {
      // Something else went wrong
      console.error(`Error: ${error.message}`);
      return Promise.reject({
        message: "An unexpected error occurred.",
      });
    }
  },
);

export const api = {
  // Initialize a participant
  async initializeParticipant(participantId, survey_responses) {
    try {
      const response = await apiClient.post("/participant_init/", {
        id: participantId,
        survey_responses: survey_responses,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to initialize participant:", error.message);
      throw error; // Re-throw the error so the component can handle it
    }
  },

  // Submit post game survey
  async submitPostGameSurvey(participantId, survey_responses) {
    try {
      const response = await apiClient.post("/post_game_survey/", {
        id: participantId,
        survey_responses: survey_responses,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to submit survey:", error.message);
      throw error; // Re-throw the error so the component can handle it
    }
  },

  // Mark participant as ready
  async participantReady(participantId) {
    console.log("participantReady(participantId=", participantId, ")");
    try {
      await apiClient.post("/participant_ready/", { id: participantId });
    } catch (error) {
      console.error("Failed to mark participant as ready:", error.message);
      throw error;
    }
  },

  // Get the current round information
  async getCurrentRound(participantId, params = {}) {
    try {
      const requestData = { participant_id: participantId, ...params };
      const response = await apiClient.post("/current_round/", requestData);
      return response.data;
    } catch (error) {
      console.error("Failed to get current round:", error.message);
      throw error;
    }
  },

  // Send a message in the chat
  async sendMessage(roundId, participantId, messageContent, thoughtContent) {
    try {
      await apiClient.post("/send_message/", {
        round_id: roundId,
        participant_id: participantId,
        message_content: messageContent,
        thought_content: thoughtContent,
      });
    } catch (error) {
      console.error("Failed to send message:", error.message);
      throw error;
    }
  },

  // Poll for a response from the other participant
  async retrieveResponse(roundId, participantId) {
    try {
      const response = await apiClient.post("/retrieve_response/", {
        round_id: roundId,
        participant_id: participantId,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to retrieve response:", error.message);
      throw error;
    }
  },

  // Request target decision
  async requestTargetDecision(roundId, participantId) {
    try {
      await apiClient.post("/request_target_decision/", {
        round_id: roundId,
        participant_id: participantId,
      });
    } catch (error) {
      console.error("Failed to make a choice:", error.message);
      throw error;
    }
  },

  // Submit target decision
  async makeChoice(roundId, participantId, choice, initialChoice = false) {
    let data = {
      round_id: roundId,
      participant_id: participantId,
      choice: choice,
      initial_choice: initialChoice,
    };
    console.log("makeChoice(data=", data, ")");
    try {
      await apiClient.post("/make_choice/", data);
    } catch (error) {
      console.error("Failed to submit target decision:", error.message);
      throw error;
    }
  },

  // Get the result of the round
  async getRoundResult(roundId, participantId) {
    try {
      const response = await apiClient.post("/round_result/", {
        round_id: roundId,
        participant_id: participantId,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to retrieve round result:", error.message);
      throw error;
    }
  },

  // get Participant rounds
  async getParticipantRounds(participantId) {
    try {
      const response = await apiClient.post("/participant_rounds/", {
        id: participantId,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to get participant rounds:", error.message);
      throw error;
    }
  },

  // send feedback
  async sendFeedback(participantId, feedback) {
    if (feedback === undefined) {
      feedback = "";
    }
    try {
      await apiClient.post("/send_feedback/", {
        participant_id: participantId,
        feedback: feedback,
      });
    } catch (error) {
      console.error("Failed to send feedback:", error.message);
      throw error;
    }
  },

  // get high-level instructions
  async getParticipantInstructions(participantId) {
    try {
      const response = await apiClient.post("/participant_instructions/", {
        id: participantId,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to get instructions:", error.message);
      throw error;
    }
  },

  // get whether or not we are in development mode
  async development_mode() {
    try {
      const response = await apiClient.get("/development_mode/");
      return response.data;
    } catch (error) {
      console.error("Failed to get development_mode:", error.message);
      throw error;
    }
  },

  async conversation_timeout() {
    try {
      const response = await apiClient.get("/conversation_timeout/");
      return response.data;
    } catch (error) {
      console.error("Failed to get conversation_timeout:", error.message);
      throw error;
    }
  },

  // get whether or not we are in chain_of_thought
  async chain_of_thought() {
    try {
      const response = await apiClient.get("/chain_of_thought/");
      return response.data;
    } catch (error) {
      console.error("Failed to get chain_of_thought:", error.message);
      throw error;
    }
  },

  // get the survey
  async survey() {
    try {
      const response = await apiClient.get("/survey/");
      return response.data;
    } catch (error) {
      console.error("Failed to get survey:", error.message);
      throw error;
    }
  },

  async isDevelopmentMode() {
    // Local storage caching isn't working here
    try {
      const data = await api.development_mode();
      const isDevelopmentMode = data.development_mode === true;
      return isDevelopmentMode;
    } catch (error) {
      console.error("Error checking development mode:", error);
      return false; // Default to false if there's an error
    }
  },

  async showChainOfThought() {
    // Local storage caching isn't working here
    try {
      const data = await api.chain_of_thought();
      const showChain = data.chain_of_thought === true;
      return showChain;
    } catch (error) {
      console.error("Error checking chain of thought: ", error);
      return false; // Default to false if there's an error
    }
  },
};
