# From Optimal Policy to DPO

## Motivation
- RLHF (Reinforcement Learning from Human Feedback) optimizes LLMs but has challenges:
  - Computationally heavy
  - Non-differentiable
  - Unstable
- DPO reformulates the problem using a **closed-form optimal policy** as a function of the reward.

## Data and Scoring
- Human evaluators provide scores for responses (challenging to assign exact numbers).
- Easier approach: **pairwise ranking**:
  - WIN = higher-scoring response
  - LOSS = lower-scoring response
- Notation:
  - Dataset: `D`
  - Sampled values: `~`
  - `X` = query, `Yw` = winning response, `Yl` = losing response

## Bradley-Terry Model
- Original loss: `log(sigmoid(score_win - score_loss))`
- Transform sum over dataset → expected value over `D`
- Focus on difference between winning and losing scores

## DPO Objective Derivation
1. Represent **optimal reward policy**: `Pi_r`
   - Depends on query `X`, response `Y`, partition function `Z`, reference model `Pi_ref`, reward `R`, regularization `β`
2. Partition function `Z` is **intractable**
   - DPO reformulates to **eliminate the need for Z**
   - Allows direct training using Bradley-Terry pairwise loss
3. Steps:
   - Isolate exponential term
   - Multiply by partition function
   - Take natural log → linearize
   - Solve for reward function
4. Plug positive and negative samples:
   - Eliminates separate reward function
   - Loss becomes function of LLM and reference model
5. Simplifications:
   - Set `β = 1`
   - Replace reference model with constant `C` (uniform probability)
   - Combine terms in log → single variable `U` = ratio of winning to losing probabilities

## Loss Behavior
- `U < 1` (winning < losing probability): increasing `U` improves model
- `U > 1` (winning > losing probability): further increasing `U` decreases loss
- Loss decreases monotonically as preferred response probability increases

## Converting Loss to Cost
- Loss: negative sigmoid of log ratios of winning/losing policies, scaled by `β`
- Reformulated into **log-likelihood cost**
- Can implement in PyTorch directly or use Hugging Face’s DPO Trainer

## Recap
- DPO leverages **closed-form optimal policy** → avoids PPO complications
- Pairwise comparison eliminates need for exact reward scores
- Loss expressed as negative sigmoid of log ratio of winning and losing probabilities
- Maximizing DPO objective trains the LLM efficiently with human preference data
