<template>
  <div v-if="show" class="overlay">
    <div class="overlay-content">
      <h2>{{ title }}</h2>

      <div v-if="subtitle">
        <p class="subtitle">{{ subtitle }}</p>
      </div>

      <div v-if="message">
        <p>{{ message }}</p>
      </div>

      <!--Close button -->
      <div id="button-container">
        <button id="close-button" @click="closeOverlay">Okay</button>
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
    title: {
      type: String,
      required: true,
    },
    subtitle: {
      type: String,
      default: null,
    },
    message: {
      type: String,
      required: true,
    },
  },
  data() {
    return {};
  },
  methods: {
    closeOverlay() {
      this.$emit("close");
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
  width: 80%;
  max-width: 600px;
}

#button-container {
  display: flex;
  justify-content: space-between;
  margin-top: 20px;
}

#close-button {
  padding: 12px 24px;
  font-size: 16px;
  border-radius: 4px;
  cursor: pointer;
  margin-left: auto;
}
</style>
