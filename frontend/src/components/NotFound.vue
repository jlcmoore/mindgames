<!-- NotFound.vue -->

<template>
  <div class="not-found">
    <h1>404 - Page Not Found</h1>
    <p>The page you are trying to access doesn't exist or is not available.</p>
    <router-link to="/">Go back to Home</router-link>
  </div>
</template>

<script>
export default {
  name: "NotFound",
};
</script>

<style scoped>
.not-found {
  text-align: center;
  margin-top: 50px;
}
</style>
