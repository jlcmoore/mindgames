<template>
  <div id="app">
    <router-view />
    <!-- This will be replaced by the routed components -->
  </div>
</template>

<script>
export default {
  name: "App",
};
</script>

<style>
body {
  font-family: "Source Sans Pro", sans-serif;
  margin: 0;
  padding: 0;
  font-weight: 300;
  font-size: 20px;
}

/* Basic reset for margin */
* {
  margin: 0;
  box-sizing: border-box;
}

/* Main container takes full screen height */
.container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header styles */
.header {
  width: 100%;
  background-color: #c2e9ec;
  /* Dark blue-gray */
  color: white;
  padding: 10px 0;
  text-align: left;
  padding-left: 25px;
  font-weight: bold;
}

/* Main content area */
.main-content {
  flex-grow: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  /* background-color: #f5f5f5; */
  /* Light gray background */
}

/* Content wrapper to center the text and limit max width */
.content-wrapper {
  background-color: white;
  padding: 40px;
  /* border-radius: 8px; */
  /* box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); */
  max-width: 800px;
  width: 90%;
  text-align: center;
}

/* Typography */

h1 {
  font-weight: 300;
  color: #13494d;
  font-size: 1.8em;
}

h2 {
  font-weight: 500;
  margin-bottom: 16px;
  font-size: 1.5em;
}

p {
  margin-top: 12px;
  margin-bottom: 12px;
  line-height: 1.6;
  font-size: 1em;
  text-align: justify;
}

/* Button styles */
button {
  background-color: #c2e9ec;
  color: #12494d;
  border: none;
  padding: 12px 24px;
  font-size: 16px;
  border-radius: 4px;
  cursor: pointer;
}

button:hover {
  background-color: #12494d;
  color: white;
}

blockquote {
  margin: 1.5em 0;
  padding: 1em 1.5em;
  border-left: 3px solid #666;
  background-color: #fafafa;
  font-size: 0.95em;
  color: #444;
  text-align: justify;
}

/* for proposals */

.proposal {
  padding: 10px;
  box-shadow: 1px 1px 3px 1px #00000033;
  margin: 2px;
  border-radius: 10px;
  height: fit-content;
}

.scenario-inner ul {
  /*  margin-left: 25px;*/
  margin-bottom: 10px;
  margin-top: 10px;
}

.scenario-inner li {
  padding: 4px;
}

.proposal h4 {
  margin-bottom: 10px;
  font-weight: 500;
}

.proposals-container {
  display: flex;
  flex-flow: wrap;
  /*  overflow-y: scroll;*/
  height: 90%;
  /* flex-wrap: wrap; */
}

.objective {
  text-align: center;
  font-size: larger;
}
</style>
