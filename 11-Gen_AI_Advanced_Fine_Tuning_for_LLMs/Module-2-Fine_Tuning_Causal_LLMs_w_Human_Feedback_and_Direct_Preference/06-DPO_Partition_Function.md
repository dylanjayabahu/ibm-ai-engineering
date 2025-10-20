# Direct Preference Optimization (DPO) and Partition Function

## Overview of DPO
- DPO is a **reinforcement learning (RL) technique** that fine-tunes models based on **human preferences**.
- It is more direct and efficient than traditional RL methods.
- **Human preference data** is collected by showing multiple outputs to users and asking them to choose the better one.
- Unlike traditional RL, DPO **directly optimizes model parameters** rather than relying on an indirect reward signal.

## Models in DPO
1. **Reward Function**
   - Uses an encoder model to evaluate relevance.
   - Example: Input = "this is a", Response = "cat" → low score (0.1), Response = "reward function" → high score (0.99).
2. **Target Decoder**
   - Model being fine-tuned, with parameters θ.
3. **Reference Model**
   - Used to regularize the optimization and measure divergence.

## Optimization
- DPO converts a **complex RL objective** into a **simpler objective** that is easier to optimize.
- Regularization term (β) measures divergence from the reference model.
- PPO (Proximal Policy Optimization) is often used for solving the RL objective in traditional setups, but DPO can bypass reward modeling.

## Partition Function
- Ensures probabilities sum to 1, **normalizing custom probability functions**.
- Used to convert simple distributions into more complex ones.
- Example with logistic function σ(x):
  - `p(y=0|x) = 1 - σ(x)` (decreases as x increases)
  - `p(y=1|x) = σ(x)` (increases as x increases)
- Scaling probabilities (e.g., exponential for y=0, Gaussian for y=1) produces non-normalized distributions.
- Partition function `Z(x)` normalizes these scaled probabilities:
  - `p'(y=0|x) = scaled(y=0)/Z(x)`
  - `p'(y=1|x) = scaled(y=1)/Z(x)`
- Ensures **valid probability distributions** after scaling.


Basically, instead of using rewards, we just make the model assign a higher likelihood to the preferred output.