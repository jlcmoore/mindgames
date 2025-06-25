<template>
  <div class="container">
    <header class="header">
      <h1>Policy Game</h1>
    </header>

    <main class="main-content">
      <div class="content-wrapper">
        <h1>Research Consent Form</h1>

        <section class="description">
          <h2>Description</h2>
          <p>
            You are invited to participate in a research study on human
            reasoning. We will ask you play a dialogue-based game with another
            participant in order to understand how people reason. You will be
            asked to craft messages about the game. We will release anonymized
            versions of these for research purposes. Please do not disclose any
            personal information like names, dates, or other identifiers.
            Participation in this research is voluntary, and you are free to
            withdraw your consent at any time.
          </p>

          <p>
            Research sometimes requires that information regarding its purpose
            not be shared with the research participants because its knowledge
            could impact the results of the research. Note that none of the
            aspects of the research being withheld are reasonably expected to
            affect your willingness to take part. While the tasks you will be
            asked to perform for this research have been explained, the full
            intent of the research will not be provided until the completion of
            the study. At that time, you will have the opportunity to ask
            questions, including about the purpose of the study and the
            procedures used, and withdraw your data if you so choose.
          </p>
        </section>

        <section class="time">
          <h2>Time Involvement</h2>
          <p>
            Your participation will take a certain number of minutes (check
            Prolific).
          </p>
        </section>

        <section class="risks">
          <h2>Risks and Benefits</h2>
          <p>
            The risks associated with this study are minimal. Study data will be
            stored securely, in compliance with Stanford University standards,
            minimizing the risk of confidentiality breach. Your individual
            privacy will be maintained during the research and in all published
            and written data resulting from the study. We cannot and do not
            guarantee or promise that you will receive any benefits from this
            study.
          </p>
        </section>

        <section class="payments">
          <h2>Payments</h2>
          <p>
            You will receive payment for your participation (check Prolific), as
            well as a bonus depending on your performance.
          </p>
        </section>

        <section class="rights">
          <h2>Participant's Rights</h2>
          <p>
            If you have read this form and have decided to participate in this
            project, please understand your participation is voluntary and you
            have the right to withdraw your consent or discontinue participation
            at any time without penalty or loss of benefits to which you are
            otherwise entitled. The alternative is not to participate. The
            results of this research study may be presented at scientific or
            professional meetings or published in scientific journals. Your
            individual privacy will be maintained in all published and written
            data resulting from the study.
          </p>

          <p>
            In accordance with scientific norms, the data from this study may be
            used or shared with other researchers for future research (after
            removing personally identifying information) without additional
            consent from you.
          </p>
        </section>

        <section class="contact">
          <h2>Contact Information</h2>
          <h3>Questions</h3>
          <p>
            If you have any questions, concerns or complaints about this
            research, its procedures, risks and benefits, contact the Protocol
            Director, Nick Haber (<a href="mailto:nhaber@stanford.edu"
              >nhaber@stanford.edu</a
            >).
          </p>

          <h3>Independent Contact</h3>
          <p>
            If you are not satisfied with how this study is being conducted, or
            if you have any concerns, complaints, or general questions about the
            research or your rights as a participant, please contact the
            Stanford Institutional Review Board (IRB) to speak to someone
            independent of the research team at 650-723-2480 or toll free at
            1-866-680-2906, or email at
            <a href="mailto:irbnonmed@stanford.edu">irbnonmed@stanford.edu</a>.
            You can also write to the Stanford IRB, Stanford University, 1705 El
            Camino Real, Palo Alto, CA 94306.
          </p>
        </section>

        <p class="notice">
          Please save or print a copy of this page for your records.
        </p>

        <div class="consent-agreement">
          <p>
            If you agree to participate in this research, please complete this
            form. By completing the form, you are indicating that you have
            understood the information contained here and would like to be a
            volunteer in this research study.
          </p>

          <button class="consent-button" @click="provideConsent">
            I Agree to Participate
          </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
export default {
  name: "ConsentForm",
  methods: {
    provideConsent() {
      // Navigate to Lobby
      this.$router.push("/pre-game-survey");
    },
  },
};
</script>
