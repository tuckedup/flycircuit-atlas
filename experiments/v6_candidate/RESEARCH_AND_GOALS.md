# Next goals and relevant research — 2026-10-22

## Decision

Aim for **resource-constrained robust continual control**, with fair baselines
and uncertainty. A selected native cue-task run cannot establish SOTA on
another paper's environment. There is currently no demonstrated published
benchmark record for this implementation. Implementing these candidates is not
evidence of an improvement; training and held-out evaluation remain pending.

Keep the original acquisition>=0.82, signed forgetting advantage>=0.07 and
body success>=0.68 endpoints in historical reports. Add prospective endpoints:

1. Learning: final average task accuracy, with learned-task accuracy and signed
   forgetting reported jointly at equal replay/data/update budgets. Compare
   observed-stream replay against both standard and step-matched controls.
2. Embodied robustness: paired-model-seed success AUC over the frozen dropout
   grid [0,.1,.25,.5,1], plus clean success, success at full dropout and worst
   success over a finite declared mask/rate set. Full channel dropout in the
   native task is not the same as FLYNN's camera dropout with other senses kept.
3. Efficiency: actual model/critic/buffer storage, optimizer steps, training
   wall time and peak memory. Report the accuracy/robustness-resource Pareto
   tradeoff rather than asserting that fewer weights imply faster execution.

No numeric positive gain is promised or selected after looking at outcomes.
For any external SOTA comparison, first bind the same environment version,
observations/actions, success rule/horizon/control period, train/eval data,
initialization/pretraining, tuning/compute/memory budgets and inference access.
If these cannot be matched, the result is an internal engineering comparison.

## Primary-source findings (focused search, not an exhaustive leaderboard survey)

### Robustness and modality-aware embodied control

[Wang and Chen, FLYNN,2026](https://arxiv.org/abs/2607.00025)
studies a connectome-derived recurrent controller for a MuJoCo differential-drive
robot. The retrieved methods describe camera, angle and collision inputs,
DAgger-style data aggregation, approximately 5M-parameter image-model controls,
and 300 paired evaluation episodes with a 600-step horizon. Metrics include
success and SPL. Its small-world control matches more structural and routing
properties than global node/edge counts. Its OOD evidence is specifically a
visual-appearance shift, not general environmental robustness. The paper calls
its mechanism analysis correlational. our native cue controller and 180-tick
FlyGym body protocol are not the same task, model scale or observation budget.

[Whole-Brain Connectomic Graph Model,2026](https://arxiv.org/abs/2602.17997)
reports connectome-structured whole-body locomotion via deep RL and sample
efficiency comparisons. In this task only its abstract was inspected; no exact
performance or reproducibility claim is inferred from it. It indicates a
relevant direction, not a matched baseline already implemented here.

### Stronger topology controls

[Topological Sensitivity in Connectome-Constrained Neural Networks,2026](https://arxiv.org/abs/2604.04033)
reports that apparent advantages shrink/disappear after shared from-scratch
initialization and degree-preserving controls replace weak comparisons. Its
small ensembles and setup do not prove that every topology advantage is absent.
For FlyCircuit this motivates preserving negative controls, checking activity
scale and initialization, and avoiding anatomy claims from engineering gains.

### Replay selection and consolidation/plasticity balance

[Improvements of Dark Experience Replay and Reservoir Sampling,2025](https://arxiv.org/abs/2504.20932)
studies adaptive weighting, blocking erroneous replay, and reservoir strategies.
Its retrieved RL experiments use SAC and different environments, so neither
its algorithm nor gains transfer automatically to our reward-only REINFORCE
task. Its reported budgets distinguish memory and replay updates. The v6
implementation supplies simple uniform/context-balanced observed-stream
baselines; it does not claim to reproduce A2ER or DER++.

These papers were returned by date-filtered searches through 2026-10-22, and
methods passages were read for FLYNN, topology controls and replay. They include
preprints. Publication prestige and a retrieved high number are not a verified
leaderboard standing. Any next public claim needs an explicit benchmark and
reproducible matched results, including unsuccessful runs.
