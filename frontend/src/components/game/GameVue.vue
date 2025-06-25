<template>
  <div class="game-container">
    <header class="header">
      <h1>Policy Game</h1>
      <div class="round-info">
        <span>Round: {{ currentRound }} of {{ totalRounds }}</span>
        <span> Turns remaining: {{ turnsRemaining }}</span>
      </div>
    </header>

    <main class="main-content">
      <!-- Scenario Section -->
      <Scenario
        :loading="loading"
        :error="error"
        :scenario-text="scenarioText"
        :scenario-view="scenarioView"
        :proposals="proposals"
        :utilities="utilities"
        :persuader-coefficients="persuaderCoefficients"
        :attributes="attributes"
        :scratchpad="scratchpad"
        @updateScenarioView="updateScenarioView"
        @update:scratchpad="scratchpad = $event"
      />

      <div class="chat-with-decision">
        <!-- 1) show the decision‐panel while pending -->
        <div v-if="initialDecisionPending" class="decision-side-panel">
          <h3>Given what you now know, which proposal would you choose?</h3>
          <ul class="side-choices">
            <li v-for="p in proposals" :key="p">
              <button
                :class="{ selected: p === initialTargetChoice }"
                @click="initialTargetChoice = p"
              >
                {{ p }}
              </button>
            </li>
          </ul>
          <button
            class="confirm-btn"
            :disabled="!initialTargetChoice"
            @click="submitInitialChoice"
          >
            Confirm
          </button>
        </div>

        <!-- 2) or Chat, once they’ve confirmed -->
        <div v-else class="chat-panel">
          <Chat
            :new-message="newMessage"
            :new-thought="newThought"
            :chat-messages="chatMessages"
            :typing="typing && turnsRemaining > 0"
            :is-target="isTarget"
            @update:new-message="newMessage = $event"
            @update:new-thought="newThought = $event"
            @send-message="handleSendMessage"
          />
        </div>
      </div>

      <!-- Decision Overlay Component -->
      <Decide
        v-if="isTarget"
        :show="showDecisionOverlay"
        :status="decisionStatus"
        :proposals="proposals"
        :target-choice="targetChoice"
        :persuaded="persuaded"
        :turns-remaining="turnsRemaining > 0"
        :target-perfect-choice="targetPerfectChoice"
        :persuader-choice="persuaderChoice"
        @close="showDecisionOverlay = false"
        @makeChoice="makeChoice"
        @newGame="newGame"
      />
      <RequestDecision
        v-else
        :show="showDecisionOverlay"
        :status="decisionStatus"
        :persuader-choice="persuaderChoice"
        :target-choice="targetChoice"
        :persuaded="persuaded"
        @close="showDecisionOverlay = false"
        @decide="requestTargetDecision"
        @newGame="newGame"
      />

      <!-- Popover Component -->
      <Popover
        :show="popover !== null"
        :title="popover?.title"
        :subtitle="popover?.subtitle"
        :message="popover?.message"
        @close="popover = null"
      />
    </main>
  </div>
</template>

<script>
import Scenario from "./ScenarioVue.vue";
import Chat from "./ChatVue.vue";
import Decide from "./DecideVue.vue";
import RequestDecision from "./RequestDecision.vue";
import Popover from "./PopoverVue.vue";
import { api } from "@/api";

const MILLISECONDS_PER_POLL = 2000;

export default {
  components: { Scenario, Chat, Decide, Popover, RequestDecision },
  data() {
    return {
      // General
      participantId: localStorage.getItem("participantId"), // Get the participant ID from local storage
      currentRoundId: null,
      loading: false,
      error: null,
      popover: null,
      chatTimeout: null,
      pollingErrorTries: null,

      // Scenario
      scenarioText: "",
      scenarioView: localStorage.getItem("scenarioView") || "Text",
      proposals: [],
      utilities: {},
      persuaderCoefficients: [],
      attributes: [],
      scratchpad: "",

      // Round info
      isTarget: false,
      userRole: null,
      currentRound: 1,
      totalRounds: null,
      roundsRemaining: null,
      turnLimit: null,
      turnsRemaining: null,

      // If the target still needs to decide
      initialDecisionPending: false,
      initialTargetChoice: null,

      // Chat
      chatMessages: [],
      newMessage: "",
      newThought: "",
      typing: false,
      typingTimeout: null,

      // holds the ID from setTimeout
      chatPollTimeout: null,
      // optional: a flag to say “I want to continue polling until I get new content”
      waitingForResponse: false,

      // Decision
      showDecisionOverlay: false,
      decisionStatus: "play",
      targetChoice: null,
      persuaderChoice: null,
      persuaded: false,
      targetPerfectChoice: null,
    };
  },
  async created() {
    // Initialize values when component is created
    this.developmentMode = await api.isDevelopmentMode();
    // This is in seconds
    this.conversation_timeout = await api.conversation_timeout();
    // We retry longer so that the server can error on us.
    this.pollingErrorTriesInitial =
      (this.conversation_timeout * 1.5) / (MILLISECONDS_PER_POLL / 1000);
    this.pollingErrorTries = this.pollingErrorTriesInitial;
  },
  mounted() {
    this.initGame();
  },
  methods: {
    checkGameError(error) {
      return (
        error.message.includes("has waited too long. Staring a new game.") ||
        error.message.includes("The participant is not in this round") ||
        error.message.includes("Round not found")
      );
    },

    gameTimedOut(error) {
      console.log(error.message);
      console.log(error);
      this.popover = {
        title: "Error",
        message: "The game has timed out. Moving you to the next round.",
      };
      setTimeout(() => this.newGame(), 3000);
    },

    async initGame() {
      this.loading = true;
      this.error = null;

      try {
        // Get round info
        let params = null;
        if (this.developmentMode) {
          params = JSON.parse(localStorage.getItem("current_round_params"));
          console.log(params);
        }
        const roundData = await api.getCurrentRound(this.participantId, params);
        console.log("Round data:", roundData);

        // Update scenario and round ID
        this.scenarioText = roundData.prompt;
        this.isTarget = roundData.is_target;
        this.userRole = this.isTarget ? "target" : "persuader";
        this.currentRoundId = roundData.round_id;
        this.turnLimit = roundData.game_data.turn_limit;
        this.turnsRemaining = this.turnLimit;
        let model = roundData.game_data.model;
        this.proposals = model.proposals;
        this.proposals.sort();
        this.utilities = model.utilities;
        this.persuaderCoefficients = model.persuader_coefficients;
        this.attributes = model.attributes;

        // Get rounds data
        const roundsData = await api.getParticipantRounds(this.participantId);
        console.log("Rounds data:", roundsData);
        this.currentRound = roundsData.rounds_completed;
        this.totalRounds = roundsData.total_rounds;
        this.roundsRemaining = roundsData.rounds_remaining;

        console.log("this.isTarget: ", this.isTarget);
        console.log("proposals: ", this.proposals);

        // If this user is the target, we hold off on normal chat until they make their
        // *initial* decision:
        if (
          this.isTarget &&
          (roundData == null ||
            roundData.game_data.target_initial_choice == null)
        ) {
          this.initialDecisionPending = true;
        } else {
          // Persuader starts normal chat polling immediately
          this.pollForResponse(false);

          // Target polls for Persuader's messages
          if (this.isTarget) {
            this.waitingForResponse = true;
            this.setTypingTimeout();
            this.pollForResponse(/*retry=*/ true);
          }
        }
      } catch (error) {
        console.error("Error initializing game:", error.message);
        console.log(error);
        this.popover = {
          title: "Error",
          message: "Failed to initialize game. Returning to Lobby...",
        };
        setTimeout(() => this.$router.push("/lobby"), 3000);
      } finally {
        this.loading = false;
      }
    },

    async setTypingTimeout() {
      let readDelay = 40 * 40;

      this.typingTimeout = setTimeout(
        () => {
          if (this.waitingForResponse) {
            this.typing = true;
            this.$nextTick(this.updateChatUI);
          }
        },
        readDelay + 1500 + Math.random() * 2000,
      );
    },

    updateChatUI() {
      const chatWindow = document.getElementById("chat-window");
      chatWindow.scrollTop = chatWindow.scrollHeight;
    },

    updateScenarioView(view) {
      this.scenarioView = view;
      localStorage.setItem("scenarioView", view);
    },

    beforeDestroy() {
      clearTimeout(this.chatPollTimeout);
      clearTimeout(this.typingTimeout);
    },

    async handleSendMessage({ message, thought }) {
      if (!message.trim()) return;
      // Add the user's message to the chat
      this.chatMessages.push({ text: message, sender: "You" });
      this.$nextTick(this.updateChatUI);

      // 2) clear any old timers & flags
      clearTimeout(this.typingTimeout);
      this.typingTimeout = null;
      this.typing = false;

      // Clear the input field
      this.newMessage = "";
      this.newThought = "";
      // get thought or scratchpad
      thought = thought || "";
      thought = thought + this.scratchpad;

      try {
        // Send the message to the backend
        await api.sendMessage(
          this.currentRoundId,
          this.participantId,
          message,
          thought,
        );

        // Poll for the agent's response
        this.waitingForResponse = true;
        this.setTypingTimeout();

        this.pollForResponse(/*retry=*/ true);
      } catch (error) {
        if (this.checkGameError(error)) {
          this.gameTimedOut(error);
          return;
        }
        console.error("Error sending message:", error.message);
        console.log(error);

        if (error.message == "Round is completed") {
          return;
        }

        // Remove the last sent chat message if there was an error
        this.chatMessages.pop();
        this.popover = {
          title: "Error",
          message: "Failed to send message. Please try again.",
        };
      }
    },

    async pollForResponse(retry = true) {
      // 1) if there’s an existing timer, clear it
      if (this.chatPollTimeout) {
        clearTimeout(this.chatPollTimeout);
        this.chatPollTimeout = null;
      }

      try {
        const response = await api.retrieveResponse(
          this.currentRoundId,
          this.participantId,
        );

        console.log("PollForResponse: ", response);

        // reset your retry‐counter on a successful reply
        if (response) {
          this.pollingErrorTries = this.pollingErrorTriesInitial;
        }

        // Flagged message
        const lastMessage =
          this.chatMessages.length > 0
            ? this.chatMessages[this.chatMessages.length - 1].text
            : null;
        if (response?.flagged_response) {
          if (lastMessage && lastMessage === response.content) {
            this.chatMessages.pop();
            this.popover = {
              title: "Flagged Message",
              message:
                "Your message could not be sent because it contained abusive language or false information about one of the policies.",
            };
            clearTimeout(this.typingTimeout);
            this.typing = false;

            return;
          }
        }

        if (response && response.turns_left != null) {
          const newMessages = response?.all_messages ?? [];

          this.turnsRemaining = response?.turns_left ?? this.turnsRemaining;

          // First check to see if the user has any turns left
          if (this.turnsRemaining < 1) {
            clearTimeout(this.chatPollTimeout);
            this.chatPollTimeout = null;
            if (this.isTarget) {
              this.showDecisionOverlay = true;
            } else {
              this.pollForDecisionResult();
            }
            clearTimeout(this.typingTimeout);
            this.typing = false;
            return;
          }

          // If there _are_ new messages (i.e. new length > old length)
          if (newMessages.length > this.chatMessages.length) {
            // Update chat messages with the latest response
            this.chatMessages = newMessages.map((message) => ({
              text: message.content,
              sender: message.role == this.userRole ? "You" : "Agent",
            }));
            this.$nextTick(this.updateChatUI);

            // we got the reply, so stop polling
            this.waitingForResponse = false;
            clearTimeout(this.chatPollTimeout);
            this.chatPollTimeout = null;

            clearTimeout(this.typingTimeout);
            this.typing = false;
            return;
          }
        }

        // otherwise, we still haven’t seen new content
        if (retry && this.waitingForResponse) {
          // schedule the _one_ next check
          this.chatPollTimeout = setTimeout(
            () => this.pollForResponse(true),
            MILLISECONDS_PER_POLL,
          );
        }
      } catch (err) {
        console.log("Error:", err);
        clearTimeout(this.typingTimeout);
        this.typing = false;

        // your existing error handling
        if (this.checkGameError(err)) {
          this.gameTimedOut(err);
          return;
        }

        if (retry && this.pollingErrorTries > 0) {
          this.pollingErrorTries--;
          this.chatPollTimeout = setTimeout(
            () => this.pollForResponse(true),
            MILLISECONDS_PER_POLL,
          );
        } else {
          // give up, show an error popover
          clearTimeout(this.chatPollTimeout);
          this.chatPollTimeout = null;
          this.popover = {
            title: "Error",
            message: "Failed to retrieve response. Please try again.",
          };
        }
      }
    },

    async requestTargetDecision() {
      this.decisionStatus = "waiting";
      this.showDecisionOverlay = true;
      console.log("Requesting target decision...");

      try {
        // Send user choice to backend
        await api.requestTargetDecision(
          this.currentRoundId,
          this.participantId,
        );

        // Poll for the decision result
        this.pollForDecisionResult();
      } catch (error) {
        if (this.checkGameError(error)) {
          this.gameTimedOut(error);
          return;
        }
        console.error("Failed to send decision: " + error.message);
        console.log(error);

        this.popover = {
          title: "Error",
          message: "Failed to send decision. Please try again.",
        };
      }
    },

    async pollForDecisionResult() {
      this.decisionStatus = "waiting";
      this.showDecisionOverlay = true;
      try {
        const result = await api.getRoundResult(
          this.currentRoundId,
          this.participantId,
        );

        if (result) {
          this.targetChoice = result.target_choice;
          this.persuaderChoice = result.persuader_choice;
          this.persuaded = result.persuaded;
          this.decisionStatus = "result";
          this.targetPerfectChoice = result.perfect_target_choice;
        } else {
          setTimeout(this.pollForDecisionResult, MILLISECONDS_PER_POLL);
        }
      } catch (error) {
        if (this.checkGameError(error)) {
          this.gameTimedOut(error);
          return;
        }
        console.error("Failed to get round result:", error.message);
        console.log(error);

        setTimeout(this.pollForDecisionResult, MILLISECONDS_PER_POLL);
      }
    },
    // called when user clicks “Confirm”
    async submitInitialChoice() {
      if (!this.initialTargetChoice) return;
      try {
        // pass that “initial=true” flag so your server knows this is
        // the pre‐chat target choice:
        await api.makeChoice(
          this.currentRoundId,
          this.participantId,
          this.initialTargetChoice,
          /* initial= */ true,
        );

        // now hide the panel and start the conversation
        this.initialDecisionPending = false;

        // clear it so Chat starts rendering fresh, then kick off polling:
        this.waitingForResponse = true;
        this.setTypingTimeout();
        this.pollForResponse(/*retry=*/ true);
      } catch (err) {
        console.error("initial choice failed", err);
        this.popover = {
          title: "Error",
          message: "Could not submit initial choice, please retry.",
        };
      }
    },
    async makeChoice(targetChoice) {
      console.log("Making choice:", targetChoice);
      try {
        // Send user choice to backend
        await api.makeChoice(
          this.currentRoundId,
          this.participantId,
          targetChoice,
          /*initial*/ false,
        );
      } catch (error) {
        if (this.checkGameError(error)) {
          this.gameTimedOut(error);
          return;
        }
        console.error("Failed to send decision: " + error.message);
        console.log(error);

        this.popover = {
          title: "Error",
          message: "Failed to send decision. Please try again.",
        };
      }

      // Poll for result
      await this.pollForDecisionResult();
    },

    newGame() {
      // Navigate to the lobby or debrief depending on round
      console.log("New game");
      console.log("Rounds remaining:", this.roundsRemaining);
      const params = JSON.parse(localStorage.getItem("current_round_params"));
      console.log("new game params:", params);
      if (params !== null) {
        localStorage.setItem("current_round_params", null);
        this.$router.push("/round-setup");
      } else if (this.roundsRemaining > 0) {
        this.$router.push("/lobby");
      } else {
        this.$router.push("/post-game-survey");
      }
    },
  },
};
</script>

<style scoped>
.header {
  display: flex;
  justify-content: space-between;
}

.round-info {
  display: flex;
  flex-direction: column;
  align-items: end;
  padding-right: 20px;
  color: #12494d;
  font-weight: 300;
}

/* Main container styles */
.main-content {
  display: flex;
  /*  justify-content: space-between;*/
  padding: 0 20px;
  align-items: baseline;
  height: calc(100vh - 70px);
  overflow: hidden;
}

h2 {
  font-weight: 500;
  margin-bottom: 16px;
  font-size: 1.5em;
  color: #333333;
  font-weight: 500;
}

.chat-with-decision {
  display: flex;
  height: 100%;
}

/* wrap Chat in a .chat-panel, give it flex:2 */
.chat-panel {
  flex: 2;
  /* if your Chat component has its own overflow, you might add: */
  display: flex;
}

/* and your decision‐panel takes the remaining third */
.decision-side-panel {
  flex: 1;
  border-left: 1px solid #ddd;
  padding: 1rem;
  background: #fafafa;
  display: flex;
  flex-direction: column;
  max-width: 400px;
}

.side-choices {
  list-style: none;
  padding: 0;
  margin: 1em 0;
}
.side-choices li + li {
  margin-top: 0.5em;
}
.side-choices button {
  width: 100%;
  padding: 0.75em;
  background: #eee;
  border: none;
  cursor: pointer;
}
.side-choices button.selected {
  background: #0065d8;
  color: white;
}
.confirm-btn {
  margin-top: auto;
  padding: 0.75em;
  background: #4caf50;
  color: white;
  border: none;
  cursor: pointer;
}
.confirm-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

/* Other shared styles here */
</style>
