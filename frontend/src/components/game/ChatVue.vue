<template>
  <section class="chat">
    <div class="chat-header">
      <h2>Chat</h2>
      <!-- <button class="decide-button" v-if="isTarget" @click="$emit('show-decision-overlay')">Decide</button> -->
    </div>

    <div id="chat-window" class="chat-window">
      <div
        v-for="(message, index) in chatMessages"
        :key="index"
        :class="getMessageClass(message)"
      >
        {{ message.text }}
      </div>
      <div v-if="typing" class="message message-left">
        <div class="message-text typing">
          <div class="dot"></div>
          <div class="dot"></div>
          <div class="dot"></div>
        </div>
      </div>
    </div>

    <div class="chat-info">
      <p class="character-count" :class="{ warning: newMessage.length >= 300 }">
        {{ newMessage.length }} / 300
      </p>
    </div>
    <div class="chat-input">
      <!-- <button @click="$emit('show-decision-overlay')" id="gavel-button">
        <img src="@/assets/gavel.svg">
      </button> -->
      <!-- Binding the input's value and emitting the update event -->
      <div class="textareas-row">
        <textarea
          v-if="showChainOfThought"
          id="thought-input"
          ref="thoughtInput"
          :value="newThought"
          placeholder="Draft your thoughts here...
(Not shared with other player.)"
          @input="updateThought($event.target.value)"
        />
        <textarea
          id="chat-input"
          ref="chatInput"
          :value="newMessage"
          :class="{ 'single-input': !showChainOfThought }"
          placeholder="Type your message here..."
          @input="updateMessage($event.target.value)"
          @keyup.enter="handleEnter"
        />
      </div>
      <button id="send-button" :disabled="!canSendMessage" @click="sendMessage">
        <img src="@/assets/send.svg" />
      </button>
    </div>
    <p id="timeout">You'll move on after ~four mintues with no response.</p>
  </section>
</template>

<script>
import { api } from "@/api";

export default {
  props: {
    chatMessages: {
      type: Array,
      default: () => [],
    },
    newMessage: {
      type: String,
      default: "",
    },
    newThought: {
      type: String,
      default: "",
    },
    typing: Boolean,
    isTarget: Boolean,
  },
  data() {
    return {
      developmentMode: false,
      showChainOfThought: false, // Default value
    };
  },
  computed: {
    canSendMessage() {
      if (this.newMessage.trim().length < 1) {
        return false;
      }
      if (this.newMessage.length >= 300) {
        return false;
      }
      if (this.chatMessages.length < 1) {
        if (this.isTarget == false) {
          return true;
        }
        return false;
      }
      const lastMessage = this.chatMessages[this.chatMessages.length - 1];
      return lastMessage.sender !== "You";
    },
  },
  async created() {
    // Initialize values when component is created
    this.developmentMode = await api.isDevelopmentMode();
    this.showChainOfThought = await api.showChainOfThought();
  },
  methods: {
    getMessageClass(message) {
      let msgClass = "message";
      // let msgClass = message.type;
      if (message.sender === "You") {
        msgClass += " message-right";
      } else {
        msgClass += " message-left";
      }
      return msgClass;
    },
    updateMessage(value) {
      // const truncatedValue = value.length > 300 ? value.substring(0, 300) : value;
      this.$emit("update:newMessage", value);

      this.adjustTextareaHeight("chat-input");
    },
    updateThought(value) {
      this.$emit("update:newThought", value);

      this.adjustTextareaHeight("thought-input");
    },
    adjustTextareaHeight(textareaId = null) {
      // If no specific textarea ID provided, adjust both
      if (!textareaId) {
        this.adjustTextareaHeight("chat-input");
        if (this.showChainOfThought) {
          this.adjustTextareaHeight("thought-input");
        }
        return;
      }

      const textarea = this.$refs[textareaId.replace("-", "")];
      if (textarea) {
        textarea.style.height = "auto"; // Reset height to calculate scroll height
        textarea.style.height = `${Math.min(textarea.scrollHeight, 80)}px`; // Max height for 4 lines
      }
    },
    async handleEnter(event) {
      if (!event.shiftKey) {
        event.preventDefault();
        await this.sendMessage();
      }
    },
    async sendMessage() {
      if (this.developmentMode && this.newMessage.includes("\\decide")) {
        console.log("requesting decision overlay");
        this.$emit("show-decision-overlay");
        this.$emit("update:newMessage", ""); // Clear the message
        this.resetTextareaHeight(); // Reset the textarea height
      }
      if (this.canSendMessage) {
        // Emit both message and thought
        this.$emit("send-message", {
          message: this.newMessage,
          thought: this.newThought,
        });
        if (this.showChainOfThought) {
          this.$emit("update:newThought", ""); // Clear the thought
        }
        this.$emit("update:newMessage", ""); // Clear the message
        this.resetTextareaHeight(); // Reset the textarea height
      } else {
        console.error(
          "You cannot send an empty message or send two messages in a row",
        );
      }
    },
    resetTextareaHeight(textareaId = null) {
      // If no specific textarea ID provided, reset both
      if (!textareaId) {
        this.resetTextareaHeight("chat-input");
        if (this.showChainOfThought) {
          this.resetTextareaHeight("thought-input");
        }
        return;
      }

      const textarea = this.$refs[textareaId.replace("-", "")]; // remove hyphen to match ref name
      if (textarea) {
        textarea.style.height = "auto";
      }
    },
  },
};
</script>

<style scoped>
section.chat {
  flex: 1;
  padding: 20px;
  display: flex;
  flex-direction: column;
  max-width: 400px;
  height: 100%;
}

.chat-header {
  display: flex;
  justify-content: space-between;
}

button.decide-button {
  /* height: 20px; */
  padding: 19px 20px;
  height: 30px;
  line-height: 0px;
}

.chat-window {
  min-height: 400px;
  height: 100%;
  overflow-y: scroll;
  padding-right: 20px;
}

.chat-input {
  display: flex;
  align-items: end;
  justify-content: space-between;
  background: black;
  padding: 10px;
}

.chat-input textarea {
  flex: 1;
  height: 40px;
  /*  border-radius: 20px;*/
  background: #373636;
  box-shadow: none;
  border: none;
  color: white;
  resize: none;
  overflow: hidden;
  max-height: 80px;
  line-height: 20px;
  height: auto;
  width: 100%;
  padding: 10px;
  font-family: "Source Sans Pro", sans-serif;
}

.chat-input button {
  height: 40px;
  width: 40px;
  padding: 5px;
  border-radius: 20px;
  border: none;
  cursor: pointer;
}

#thought-input {
  border-top-left-radius: 20px;
  border-top-right-radius: 20px;
  border-bottom: 3px dashed #555; /* Add dashed separator */
}

/* Style for bottom textarea */
#chat-input:not(.single-input) {
  border-bottom-left-radius: 20px;
  border-bottom-right-radius: 20px;
}

/*  a container for the bottom row with send button */
.textareas-row {
  display: flex;
  align-items: center;
  flex-direction: column;
  width: 100%;
  margin: 0 13px;
}

/* Style for when there's only one input (showChainOfThought is false) */
.single-input {
  border-radius: 20px !important;
}

button img {
  width: 30px;
  height: 30px;
}

#send-button {
  background-color: #007bff;
  color: white;
  padding-left: 7px;
}

#send-button:hover {
  background-color: #0056b3;
}

#send-button:disabled {
  background-color: #5e7a98;
  cursor: not-allowed;
}

button#gavel-button {
  background-color: orange;
}

button#gavel-button:hover {
  background-color: rgb(229, 125, 5);
}

/* Chat Message Styles */

.message {
  max-width: 70%;
  width: fit-content;
  font-weight: 300;
  padding: 8px 16px;
  border-radius: 10px;
  margin-bottom: 20px;
  word-wrap: break-word;
  white-space: pre-wrap;
  line-height: 1.4;
  position: relative;
  font-size: 16px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.message-left {
  background-color: #333;
  color: white;
}

.message-right {
  background-color: #e0f7fa;
  margin-left: auto;
}

p.character-count {
  margin-bottom: 0;
  float: right;
  font-size: 16px;
}

p.character-count.warning {
  color: red;
}

/* Typing animation */

.typing {
  align-items: center;
  display: flex;
  height: 17px;
  margin-top: 5px;
}

.typing .dot {
  animation: mercuryTypingAnimation 1.8s infinite ease-in-out;
  background-color: #666;
  border-radius: 50%;
  height: 7px;
  margin-right: 4px;
  vertical-align: middle;
  width: 7px;
  display: inline-block;
}

.typing .dot:nth-child(1) {
  animation-delay: 200ms;
}

.typing .dot:nth-child(2) {
  animation-delay: 300ms;
}

.typing .dot:nth-child(3) {
  animation-delay: 400ms;
}

.typing .dot:last-child {
  margin-right: 0;
}

@keyframes mercuryTypingAnimation {
  0% {
    transform: translateY(0px);
    background-color: #666;
  }

  28% {
    transform: translateY(-7px);
    background-color: #888;
  }

  44% {
    transform: translateY(0px);
    background-color: #666;
  }
}

#timeout {
  font-style: italic;
  font-size: small;
}
</style>
