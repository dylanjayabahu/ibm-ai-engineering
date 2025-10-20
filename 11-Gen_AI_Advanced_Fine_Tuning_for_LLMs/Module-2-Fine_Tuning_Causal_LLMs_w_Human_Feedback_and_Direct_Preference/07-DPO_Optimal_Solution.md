# DPO Optimal Solution Notes

## Objective Functions in ML
- Guide models through learning to make accurate predictions.
- Measure difference between predicted outcomes and actual targets.
- Provide a clear optimization target.

## Closed-Form RL Objective
- Desired policy: `Pi*`, arbitrary policy: `Pi_ref`.
- KL divergence measures difference between policies:
  - `KL(Pi* || Pi_ref)` minimized when policies are identical.
- Visualization: Two Gaussians show divergence reduction as overlap increases.
- Tricks for optimization:
  - Convert maximization → minimization by negating function.
  - Scaling by a constant does not change the optimum location.

## Reformulating the RL Objective
- Multiply by -1 and 1/β for easier optimization; location of optimum unchanged.
- Express as an expectation value for simplification.
- Objective becomes: `log(Pi / Pi_ref) - (reward / β)`.

## DPO Objective
- Reformulate using logarithms: combine terms → `log(Pi / exp(reward) * Pi_ref)`.
- Normalize using partition function `z(x)` to ensure valid distribution.
- Denominator contains reward-weighted policy `Pi_r`.
- Subtracting constants does not affect the optimum.

## Optimal Solution
- Minimize KL divergence between target policy and reward policy.
- Optimal policy = **reward policy**, scaling reference model to reward function.
- β controls weighting of reference vs reward.
- Example:
  - Input token: `"this is a"`.
  - Outputs: `"cats"` (ref 0.8), `"reward function"` (ref 0.1).
  - After scaling with reward, new policy assigns high probability to `"reward function"`.

## Partition Function Complexity
- For sequence length 1: sum over all words in vocabulary V.
- Sequence length 2: sum over all pairs → V².
- Sequence length T: sum over sequences → V^T (exponential growth).
- Reward policy allows handling this complexity efficiently.

## Recap
- Objective functions reveal patterns and optimize predictions.
- KL divergence aligns desired policy with reference policy.
- DPO scales reference model with reward policy for optimal solution.
- β parameter adjusts influence of reward function vs reference model.
