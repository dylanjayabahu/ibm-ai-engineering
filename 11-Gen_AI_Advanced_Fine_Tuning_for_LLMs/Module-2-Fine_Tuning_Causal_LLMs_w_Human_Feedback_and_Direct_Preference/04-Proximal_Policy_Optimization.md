Proximal Policy Optimization (PPO)
- 
- Goal is to fine tune a policy (aka a language model) to maximize expected reward while maintaining training stabiility 

- We set hup the pretrained agent with learnable parameters $\theta$
    - We have many rolout pairs; with $X$ as the input and $Y$ as the output
    - The reward function evaluates the rollout, and updates $\theta$ to maximize $\mathbb{E}[r]$, the expected reward

- Policy Gradient Methods
    - The Objective function is the qunatity to maximize (the expected reward)
    - We estimate the reward for a sample response $Y$ given query $X$ over the entire dataset
    - Then we use the refernece model and a KL Penality, $\beta$, 
        - Basically we are keeping tue updated policy close the original $\theta$ so that we dont veertoo far off and we make sure we have stable training
- PPO
    - method to stably maximize the plicy gradient objective    
    - we clip the surrogate (subseqently learned) objectives to prevent large unstable updates
        - odne with the KL penalty coefficient $\beta$

- Log-Derivative trick
    - we can simplify the objective for individual queries by convertin expressions to an analytical distribution
    $$
    ∇_θ​E[r(X,Y)]=E[∇θ​logπ_θ​(Y∣X)r(X,Y)]
    $$
    - this lets us rearrange and factor the gradient; we can use gradient ascent with samples now 

- Practical Training
    - eval model regularly with human feedback
    - start with moderate $\beta$ value for KL penalty
    - increase temp to explore more
    - Focus on maximizing reward while keeping updates stable
    - We do this in code with PPO trainer and HuggingFace
