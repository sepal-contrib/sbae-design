<template>
  <v-card
    :elevation="elevation"
    :style="{
      height: height,
      textAlign: 'center',
      cursor: 'pointer',
      position: 'relative',
    }"
    class="step-card"
    @click="handleClick"
    :hover="true"
  >
    <!-- Step number badge in top-left corner -->
    <v-badge
      :content="number"
      color="white"
      class="step-badge"
      overlap
      offset-x="10"
      offset-y="15"
      style="position: absolute; top: 12px; left: 12px; z-index: 2"
    >
    </v-badge>

    <v-card-text
      class="d-flex flex-column justify-center align-center fill-height"
    >
      <!-- Large Material Design Icon -->
      <div
        style="
          margin: 20px 0;
          display: flex;
          justify-content: center;
          align-items: center;
        "
      >
        <v-icon
          color="secondary"
          class="step-icon"
          style="
            font-size: 4rem !important;
            width: auto !important;
            height: auto !important;
          "
        >
          {{ icon }}
        </v-icon>
      </div>

      <!-- Step title -->
      <div style="margin-top: 10px">
        <h3
          style="
            margin: 0;
            font-size: 1.1rem;
            line-height: 1.3;
            font-weight: normal;
          "
        >
          {{ title }}
        </h3>
      </div>
    </v-card-text>
  </v-card>
</template>

<script>
export default {
  name: "StepCard",
  methods: {
    handleClick() {
      // Call the click event (event_click from Python becomes "click" method)
      if (this.click) {
        this.click(parseInt(this.number));
      }
    },
  },
};
</script>

<style scoped>
.step-card {
  transition: all 0.3s ease;
  position: relative;
}

.step-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
}

.step-badge {
  position: absolute;
  top: 12px;
  left: 12px;
  z-index: 2;
}

.icon-container {
  margin: 20px 0;
  display: flex;
  justify-content: center;
  align-items: center;
}

.step-icon {
  transition: all 0.3s ease;
  font-size: 4rem !important;
  width: auto !important;
  height: auto !important;
}

.step-card:hover .step-icon {
  transform: scale(1.1);
}

.step-content {
  margin-top: 10px;
}

.step-title {
  margin: 0;
  font-size: 1.1rem;
  line-height: 1.3;
  color: rgba(0, 0, 0, 0.87);
  font-weight: normal;
}

/* Dark theme support */
.theme--dark .step-title {
  color: rgba(255, 255, 255, 0.87);
}
</style>
