# Competition Guidelines

These guidelines describe the fallback, team-run evaluation format. Teams must use
the **organizer-provided starter code and agent interface**. The **full harness is
optional** and may be used for learning, logging, and resource tracking.

Existing rules for permitted models, tools, and data remain in effect. In this
format, teams run their own final evaluation and report their results.

## Judging Criteria

| Criterion | Weight | What We Assess |
| --- | ---: | --- |
| **Performance** | **65%** | Your quality score (Q) on the organizer's hold-out dataset. |
| **Learning and Experimental Evidence** | **35%** | Your automated learning process and credible evidence of its effectiveness. |

## Performance and Scoring

Evaluate your final solution on the **complete organizer hold-out dataset** using
the **organizer-provided scoring utility without modification**. Follow the
[quality (Q) definition in the Participant Scoring Reference](scoring.md#quality-q).

**Q is exact-match micro-F1 over annotated attribute values.** Matching values earn
credit; missing and extra values reduce the score. Unannotated product-attribute
pairs are excluded, and matching uses exact strings without case folding,
trimming, or synonym credit.

**Only hold-out Q determines the performance portion of judging.** Inference
efficiency (E), learning cost (L), and the leaderboard composite are not used in
this rubric. Training and experiment scores do not count toward final performance.

## Learning and Experimental Evidence

Explain what your solution learns, what feedback it uses, and what it automatically
updates, such as **prompts, memory, rules, or reusable skills**. **Model parameter
updates are not permitted.**

Show a fair before-and-after comparison on the same experiment data, using the same
metric and comparable settings. Clearly distinguish automated learning from manual
changes, including automatically learned or refined skills versus manually authored
ones. Acknowledge limitations and unsuccessful experiments.

Fixed agents and manually tuned prompts remain eligible for performance points,
but do not earn automated-learning points. Tool use or longer reasoning alone does
not count as learning. Judges assess both the learning process and the evidence
supporting its effectiveness.

## What to Submit

- Your final code, saved agent state, and basic run instructions.
- Your hold-out predictions, Q score, and scoring utility output.
- A short explanation of your approach.
- Your learning process and before-and-after experiment results, if claiming learning.

## Integrity and Fair Participation

Teams are expected to report results accurately, follow the evaluation rules, and
clearly disclose how their solutions were developed.

- Finalize your code and agent state before hold-out evaluation. Keep them unchanged
  and learning disabled during final inference.
- Use hold-out data only for final evaluation, never for training, tuning, or
  selecting your solution.
- Do not hard-code hold-out answers or allow your agent to access hold-out labels
  when generating predictions.
- Use the scoring utility unchanged and evaluate every hold-out example. Do not
  omit failures or cherry-pick results.
- Disclose manual improvements, external resources, evaluation errors, and retries.

**Judges may request supporting evidence or a verification rerun. Cheating or
knowingly misrepresenting results may lead to disqualification.**
