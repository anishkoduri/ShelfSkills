# Participant Scoring Reference

This page summarizes official scoring. The active organizer release and the published
leaderboard show the policy used for an event; checked-in values below are provisional.

## Notation

| Symbol | Meaning |
| --- | --- |
| $Q$ | Quality: exact-match micro-F1 over observed attribute values |
| $E$ | Inference-efficiency measurements; its composite component uses cost per product $c_E$ |
| $L$ | Metered learning cost at the submitted checkpoint |
| $N$ | Number of products in an evaluation pass |
| $c_E$ | Measured base-model inference credits per product |
| $TP$, $FP$, $FN$ | Matching, extra, and missed observed values |
| $R_E$, $R_L$ | Cost reference values where the corresponding composite component reaches zero |
| $w_Q$, $w_E$, $w_L$ | Weights assigned to Q, normalized E, and normalized L |
| $S_E$, $S_L$ | Normalized E and L components of the composite |
| $S$ | Published 0-100 leaderboard composite; not an official scorecard field |

Formula-specific symbols are defined where they are introduced below.

## Quality (Q)

Q is micro-F1 over exact observed attribute values on the protected evaluation set. Only
annotated product-attribute pairs count. Unannotated pairs are excluded; they are not negative
labels. Matching is exact string equality, with no case folding, trimming, synonym, or semantic
credit.

For each annotated product-attribute pair $j$, let $Y_j$ be its observed value set and
$\widehat{Y}_j$ its validated prediction set. A true positive (TP) is a matching value, a
false positive (FP) is an extra predicted value, and a false negative (FN) is a missed observed
value. The counts are summed over the annotated pairs $\mathcal{O}$:

$$
TP=\sum_{j\in\mathcal{O}}|\widehat{Y}_j\cap Y_j|,\qquad
FP=\sum_{j\in\mathcal{O}}|\widehat{Y}_j\setminus Y_j|,\qquad
FN=\sum_{j\in\mathcal{O}}|Y_j\setminus\widehat{Y}_j|.
$$

$$
Q=\begin{cases}
\dfrac{2TP}{2TP+FP+FN},&2TP+FP+FN>0,\\
0,&\text{otherwise}.
\end{cases}
$$

An omitted value on an observed attribute is a false negative. Extra values on an observed
attribute are false positives. A malformed product response is scored as having no predictions
for that product. Predictions for unannotated attributes do not affect Q if they are otherwise
well-formed.

## Inference Efficiency (E)

E records the frozen agent's official prediction pass: model calls, tokens, requests, tools,
batch sizes, and timing. Here $N$ is the number of evaluated products; $T_{in}$ and $T_{out}$
are total base-model input and output tokens; $p_{in}$ and $p_{out}$ are that model's released
credit rates. These give total inference cost $C_E$ and per-product cost $c_E$:

$$
C_E=T_{in}p_{in}+T_{out}p_{out},\qquad c_E=C_E/N,
$$

where $N$ is the number of products, not calls or batches. The composite uses cost per product;
latency and call counts are reported diagnostics, not separate composite terms. Calls through the
injected model, including retries, contribute measured usage.

## Learning Cost (L)

L is cumulative metered cost at the checkpoint the team submits, not the total of all local
experiments. In the formula, $T^t_{in/out}$ are teacher tokens; $p^t_{in/out}$ are teacher
credit rates; $T^{b,L}_{in/out}$ are base-model tokens used inside learning; $p^b_{in/out}$
are base-model rates; $t_{evolve}$ is chargeable time inside `evolve`; and $p_{compute}$ is
the released compute rate:

$$
L=T^t_{in}p^t_{in}+T^t_{out}p^t_{out}
+T^{b,L}_{in}p^b_{in}+T^{b,L}_{out}p^b_{out}
+t_{evolve}p_{compute}.
$$

Credits are accounting units, not dollars. L continues to be measured even when there is no
learning-credit cap. A later learning step does not change the L recorded on an earlier checkpoint.

## Leaderboard Composite

Only the published leaderboard combines Q, E, and L. The score is calculated from one scored
submission as follows:

1. **Start with the three measured inputs.** $Q$ is observed-value micro-F1 from the protected
  evaluation; $L$ is the metered cost of the submitted checkpoint; $E$ uses base-model token
  usage from the prediction pass.
2. **Convert E to credits per product.** Let $T_{in}$ and $T_{out}$ be base-model input and output
  tokens, $p_{in}$ and $p_{out}$ their released credit-per-token rates, and $N$ the number of
  evaluated products:

  $$
  c_E=\frac{T_{in}p_{in}+T_{out}p_{out}}{N}.
  $$

3. **Normalize E and L.** $R_E$ and $R_L$ are the reference values from the active policy:
  reaching either reference gives that component zero; zero cost gives it one.
4. **Apply the weights and scale to 100.** $w_Q$, $w_E$, and $w_L$ are the policy weights; they
  are nonnegative and sum to 1.

The checked-in `composite-v1` values are provisional, not a final event policy:

| Parameter | Checked-in value | Meaning |
| --- | ---: | --- |
| $w_Q$ | 0.60 | Weight for raw Q |
| $w_E$ | 0.20 | Weight for normalized inference efficiency |
| $w_L$ | 0.20 | Weight for normalized learning cost |
| $R_E$ | 4 credits/product | Inference cost per product where the E component reaches zero |
| $R_L$ | 10 credits | Submitted-checkpoint learning cost where the L component reaches zero |

Below its reference, each cost component decreases linearly from 1 at zero cost to 0 at the
reference. At or above its reference it stays at 0. These references are score-normalization
points, **not usage caps or spending limits**. Actual values for an event come from the active
organizer policy, not necessarily this table.

$$
S_E=\max(0,1-c_E/R_E),\qquad S_L=\max(0,1-L/R_L),
$$

$$
S=100(w_QQ+w_ES_E+w_LS_L).
$$

The raw inference cost $c_E$ and learning cost $L$ are not capped: usage can exceed either
reference and continues to be measured. Only their normalized score components are clipped at
zero; costs at or above $R_E$ or $R_L$ earn zero for that component. These references are not
run limits; separately configured organizer step and runtime limits may still stop a run.

**Worked example.** Suppose an evaluation has $T_{in}=2{,}000$ tokens at
$p_{in}=0.001$ credits/token and $T_{out}=1{,}000$ tokens at $p_{out}=0.002$ credits/token,
over $N=2$ products. Then total inference cost is $2+2=4$ credits and
$c_E=4/2=2$ credits/product. If $Q=0.80$ and the submitted checkpoint has $L=3$ credits, the
checked-in references give:

$$
S_E=1-2/4=0.5,\qquad S_L=1-3/10=0.7,
$$

$$
S=100(0.60\times0.80+0.20\times0.5+0.20\times0.7)=72.
$$

These are illustrative numbers, not event model prices or approved policy values.

The publisher selects each team's highest-composite scored submission and reports Q, E, and L
from that same attempt. Q, E, and L remain separate in the official result. A local `dev-run`
does not calculate Q, E, or a composite; local quality checks are participant-owned and are not
official results.

## Local Development Evaluation

`dev-run` runs `evolve` on every record in the supplied package, records learning usage, and
does not run prediction evaluation. If you want to compare candidate prompts or guidance, your
agent can make a deterministic, product-disjoint split of `context.training_examples` inside
`evolve`. Use only the training portion to create a candidate. Score the reserved portion with
participant code and the observed-value Q formula above. Do not send validation labels to the
teacher. An agent may implement this same micro-F1 calculation itself to choose whether to keep
a change. That local score is not calculated or certified by the harness and is never the official
Q. Any base-model calls made for this comparison inside `evolve` are learning calls and contribute
to L.

To estimate E, run a separate local prediction pass on reserved participant data and measure the
base-model tokens and cost per product using the policy rates. This participant estimate is not
the official inference pass. The checked-in `model-policy.yaml` composite settings are provisional;
the activated organizer policy controls official scoring. The Python helper
`datathon.publish.composite` can calculate a composite from supplied `ScoreMetrics` and
`ScorePolicy`, but it does not run evaluation or collect metrics. The `datathon publish` CLI
publishes scored result records; it is not a development scoring command.

## Official Evaluation

The organizer runs the pushed source, freezes an eligible checkpoint, then evaluates that frozen
agent once on protected hold-out data. Participants do not receive hold-out rows, labels, or
iterative feedback. Existing submitted commits are not rescored when policy changes; follow the
organizers' instructions for a new attempt.