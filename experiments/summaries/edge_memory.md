# Persistent edge memory v1

## Method
Retained learned particle-pair and particle-wall embeddings across
predictions using an additive update followed by LayerNorm.

Trained on seed 42 with two consecutive predictions per example.
Memory initialized to zero for each training sequence and evaluation window.
This was learned latent memory, not explicit DEM friction history.

## Results
No screened checkpoint passed all five 50-prediction validation windows.

At epoch 20, retaining memory completed:
- Frame 2000: 50/50
- Frame 6000: 9/50
- Frame 10000: 21/50
- Frame 14000: 22/50
- Frame 18000: 20/50

Resetting memory every prediction, with identical weights:
- Passed all five windows.
- Overall normalized rollout score: 11.3269.
- Memoryless control's selected checkpoint score: 7.8988.
- Lower scores are better.

## Interpretation
Retaining embeddings contributed to overlap-screening failures.
The cause is not established. Two-step training may inadequately
prepare the recurrence for longer rollouts.

This does not rule out gated contact memory or physical friction history.

## Decision
Do not adopt this version.
Preserve code, checkpoint, and raw logs.
Next experiment: longer autoregressive fine-tuning of the memoryless model.