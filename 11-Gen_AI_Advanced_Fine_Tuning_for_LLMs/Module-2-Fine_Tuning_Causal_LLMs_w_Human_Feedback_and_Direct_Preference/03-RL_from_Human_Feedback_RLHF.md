Reinforcement Learning from Human Feedback (RLHF) 
- 
- Goal: Fine tune a pretrained LLM with human feedback to produce higher quality, more accurate responses
- We will do this with a reward function to score outputs based on how well they align with human preferences 

Reward Calculation 
- 
- Use a reward function: $r(X,Y)$, where $X$ is the input query and $Y$ is the model response
- Example Query: "Which country owns Antarctica?"
    - Response: "u8904q" is irrelvant so the reward = 0
    - Response: "No country owns antarctica" is mostly correct so reward=0.9
    - Response: "Antartctica is governed by an international treaty" is ideal so reward = 1

Rollouts
- 
- recall from prev lesson
- We can explore multiple rollouts and see which is better to adjust our policiy 

Expected reward
- 
- We want to approzimate the expected reward by averaging the rewards earned over queries and responses
- Consider $N$ = total queries, $n$ = invidiual query, $K$ = responses per query, and $k$ = individual response
- The actual expected value is: $\mathbb{E}[r(X,Y)] = \sum_Yp(Y |X)r(X,Y)$

Incorperating Human Feedback
- 
- We produce a distribution of responses for a query from the pretrained LLM
- The reward model scores each response using human feedback 
- We train with an input wuery X, where the agent makes multiply responses Y (multiple rollouts)
    - we compute $r(X,Y)$
    - we then update the policy parameters, $\theta$ (the model weights) to increase probability of high-rewards
