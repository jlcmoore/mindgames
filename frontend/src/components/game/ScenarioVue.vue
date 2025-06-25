<template>
  <section class="scenario">
    <div class="scenario-header">
      <h2>Scenario</h2>
      <div class="scenario-tabs-container">
        <span style="font-size: 1.1em; margin-right: 5px">View: </span>
        <div
          v-for="viewOption in ['Text', 'Instructions']"
          :key="viewOption"
          class="scenario-tab"
          :class="{ selected: scenarioView == viewOption }"
          @click="$emit('updateScenarioView', viewOption)"
        >
          <span>{{ viewOption }}</span>
        </div>
      </div>
    </div>

    <div v-if="loading"><p>Loading scenario...</p></div>
    <div v-else-if="error">
      <p>{{ error }}</p>
    </div>

    <div v-else class="scenario-outer">
      <!-- Text -->
      <div
        v-if="scenarioView == 'Text'"
        class="scenario-inner"
        v-html="scenarioText"
      ></div>

      <!-- Proposals -->
      <div v-else-if="scenarioView == 'Proposals'">
        <h4>Your priorities:</h4>
        <div class="values-container">
          <span
            v-for="(value, index) in persuaderCoefficients"
            :key="index"
            class="value"
          >
            • You {{ formatCoefficient(value) }} {{ index }}
          </span>
        </div>

        <h4>Proposals:</h4>
        <div class="proposals-container">
          <div
            v-for="(proposal, index) in proposals"
            :key="index"
            class="proposal"
          >
            <h3>Proposal {{ proposal }}</h3>
            <ul>
              <li v-for="(utility, key) in utilities[proposal]" :key="key">
                {{ formatUtility(utility) }} {{ key }}
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Table -->

      <!-- Payoff Matrix view -->
      <div v-else-if="scenarioView == 'Table'" class="matrix-view">
        <table>
          <thead>
            <tr>
              <th style="border-bottom: none"></th>
              <th v-for="proposal in proposals" :key="proposal">
                {{ proposal }}
              </th>
              <th class="you">You</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(label, index) in attributes" :key="index">
              <th>{{ label }}</th>
              <td v-for="proposal in proposals" :key="proposal">
                {{ utilities[proposal][label] }}
              </td>
              <td class="you">
                {{ persuaderCoefficients[label] }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Instructions -->
      <div v-else-if="scenarioView == 'Instructions'" class="instructions-view">
        <div v-if="instructions" v-html="instructions"></div>
        <div v-else>
          <p>No instructions available.</p>
        </div>
      </div>

      <!-- Scratchpad -->
      <div v-else-if="scenarioView == 'Scratchpad'" class="scratchpad-view">
        <p>Use this scratchpad to take notes.</p>
        <textarea
          :value="scratchpad"
          style="width: 100%; height: 100%"
          @input="updateScratchpad($event.target.value)"
        ></textarea>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  props: {
    loading: {
      type: Boolean,
      default: false,
    },
    error: {
      type: String,
      default: null,
    },
    scenarioText: {
      type: String,
      default: "",
    },
    scenarioView: {
      type: String,
      default: "Text",
    },
    proposals: {
      type: Array,
      default: () => [], // Array needs factory function
    },
    utilities: {
      type: Object,
      default: () => ({}), // Object needs factory function
    },
    attributes: {
      type: Array,
      default: () => [], // Array needs factory function
    },
    persuaderCoefficients: {
      type: Array,
      default: () => [], // Array needs factory function
    },
    scratchpad: {
      type: String,
      default: "",
    },
  },

  data() {
    return {
      instructions: localStorage.getItem("instructions") || "",
    };
  },
  methods: {
    formatUtility(utility) {
      if (utility == -1) {
        return "Decreases";
      } else if (utility == 0) {
        return "Doesn't change";
      } else if (utility == 1) {
        return "Increases";
      } else {
        return utility;
      }
    },

    formatCoefficient: function (value) {
      if (value == -1) {
        return "dislike";
      } else if (value == 0) {
        return "don't care about";
      } else if (value == 1) {
        return "like";
      } else {
        return value;
      }
    },

    updateScratchpad(value) {
      this.$emit("update:scratchpad", value);
    },
  },
};
</script>

<style scoped>
/* Scenario Section Styles */
.scenario {
  width: 40%;
  padding: 20px;
  border-radius: 8px;
  flex: 1;
  font-size: 17px;
  height: calc(100% - 25px);
  overflow: hidden;
  max-width: 1000px;
}

.scenario-outer {
  height: 100%;
  overflow-y: scroll;
  padding-bottom: 40px;
  padding-right: 20px;
}

.scenario-inner {
  padding-bottom: 20px;
}

/* tabs */

.scenario-header {
  display: flex;
  justify-content: space-between;
}

.scenario-tabs-container {
  display: flex;
  align-items: center;
}

.scenario-tab {
  background: #828282;
  color: white;
  height: fit-content;
  padding: 1px 10px;
  border-radius: 8px;
  margin: 5px 2px;
  cursor: pointer;
}

.scenario-tab.selected {
  background: #c2e9ec;
  color: #12494d;
}

.scenario-tab:hover {
  background: #3f3f3f;
}

.scenario-tab.selected:hover {
  background: #12494d;
  color: white;
}

/* Text */

/* Proposals */

h4 {
  font-weight: 500;
  margin-top: 20px;
}

.values-container {
  display: flex;
  flex-wrap: wrap;
}

span.value {
  padding: 10px;
}

/* Table */

td,
th {
  padding: 15px;
  text-align: center;
  font-size: 24px;
  margin: 0px;
}

tbody th {
  text-align: right;
  font-variant: titling-caps;
  font-variant-caps: all-small-caps;
}

td.you {
  border-left: 1px solid black;
  border-collapse: collapse;
}

table {
  border-collapse: collapse;
}

thead th {
  border-bottom: 1px solid black;
}

th {
  font-weight: 500;
}

th.you {
  border-left: 1px solid black;
}
</style>
