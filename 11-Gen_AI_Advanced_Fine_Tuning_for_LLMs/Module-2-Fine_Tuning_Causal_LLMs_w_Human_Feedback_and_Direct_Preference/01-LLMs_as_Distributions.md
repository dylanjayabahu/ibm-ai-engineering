LLMs as probability distributions  
-
Lanuage 
    Models can be thought of as defining a probability distribution over possible outputs given a query 
    For a query X like "Which is the largest ocean?",   
        the LLM can generate multiply responses Y (e.g "Pacific Ocean, Pacific Ocean is 155 M square kilometers, Atlatnic ocean")
        each with a probability 
    The policy interpretation of this:
        We can represent this distribution as $Y \sim \pi(Y|X)$ where $\pi$ is the probability policy given the query 
    
Token-level probabilties
    LLMS dont generate whole sentence at once; they generate token by token 
    at each timestep t the transformer gives a prob distribution over al possible tokens (probability over vocab) using softmax
    e.g. at time t, the word after "the largest ocean is" could be:
        pacific with prob 0.6
        atlantic with prob 0.3
        indian with prob 0.1
    The model samples from this distribution to pick the next word 

Sequnetial dependence 
    The prob of next token depends on all prev tokens
    If model ahs already generated Pacific, next token changes
    This is why sequences vary even for same promt; diff tokens lead to diff continuoations 

Sampling Techniques 
-
The
    default is to pick simply the most likely (with argmax). Instead, we can introduce randomness
    $\\$ 
1. Temperature, $\tau$  
- Controlled randomness
- Low temperature ($\tau$ <1) means model is more confident and outputs are less random 
- High temp ($\tau$ >1) means probabilities flatten and outputs are more random 
- A temp of 0 corresponds to simply taking argmax
- Softmax with temperature changes the probabilities: 
$$
p_i = {e^{z_i  \tau} \over \sum_je^{z_j/\tau}}
$$
2. Top-K Sampling 
- restrict the selection to the top k most probable tokens, then choose proportionally amonst them
3. Top-p (nucleus) sampling 
- choose from the smallest set of tokens whose cumulative probability $\ge$ p, p is chosen
- ensure sampling focuse on likely tokens without a fixed K unlike top K sampling
4. Beam search 
- keep track of multiple candidate sequences and expands the top sequences at each step
- useful for more coherent outputs 
5. repetition penalty
- discourage repeated tokens to improve output diversity
5. min/max tokens
- limits length of generated sequences


Generating Text (with improved sampling)
- 
- Convert input text to token embeddings
- Pass embedddings through transformer layers
- Apply softmmaxoutput to logits to get probability over vocab
- Sample next token (using temperature, top-k, top-p, etc), instead of argmax
- Feed chosen token back in and repeat until EOS or until max output length reached