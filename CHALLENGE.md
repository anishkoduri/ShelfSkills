# TAMU 2026 | H-E-B
## Don't Just Build an Agent. Teach It to Improve.

**Can your agent learn a better strategy and recognize when further improvement isn't worth the cost?**

### The Challenge

Turn messy product information into reliable, structured attributes.

Product descriptions contain incomplete information, inconsistent wording, and distracting
details. Your agent must identify what the evidence supports and improve its approach through
an automated learning process.

The real challenge is **designing how it learns**.

Improve your agent's prompts, instructions, skills, memory, workflows, or model parameters.
Discover which changes produce measurable gains at an acceptable cost.

### Choose Your Learning Strategy

Explore approaches such as:

- **Prompts and instructions:** Refine extraction guidance, decision rules, and when to abstain.
- **Reusable skills:** Develop procedures for interpreting ingredients, resolving ambiguity, or validating outputs.
- **Rubric-guided critique:** Assess training predictions for evidence, completeness, consistency, and validity, then improve.
- **Memory and retrieval:** Curate useful demonstrations and retrieve lessons from previous mistakes.
- **Workflow optimization:** Learn when to retrieve information, invoke a skill, ask the teacher, or stop.
- **Model adaptation:** Use distillation to learn from a stronger teacher, or fine-tune weights where the approved runtime supports it.

Combine ideas. Compare alternatives. Discover which improvements generalize beyond the examples
your agent studied. Confirm organizer support before relying on fine-tuning for an official run;
it is not yet a verified official execution path.

Your rubrics guide learning, but they do not replace the harness's official scoring.

### Build a Learning Loop

**Start. Predict. Learn. Revise. Evaluate. Repeat.**

Your system should improve without manual intervention during the learning run. Save all learned
prompts, instructions, skills, memory, and model state so the resulting agent can be reconstructed
and evaluated independently.

You receive labeled experiment data for development. Final evaluation uses protected products
your agent has never seen.

### Better Isn't Just More Accurate

The harness measures three things:

| Dimension | The Question |
|---|---|
| **Quality (Q)** | How accurately does your agent extract supported attribute values? |
| **Inference Efficiency (E)** | How costly is the improved agent to run? |
| **Learning Cost (L)** | How much did it spend getting better? |

The leaderboard combines Q, E, and L using organizer-published weights and reference values.
A powerful strategy improves quality **without spending more than the improvement is worth**.

### Experiment. Explain. Compete.

Use the harness to compare learning strategies and inspect their quality and cost trade-offs.
Submit a reproducible agent through your team repository, with up to three official submissions,
each at a different Git commit. Official runs use only the permitted model and teacher services.

**What will your agent learn, and how efficiently will it learn it?**

For setup, experimentation, scoring details, and submission instructions, see the
[participant README](README.md).