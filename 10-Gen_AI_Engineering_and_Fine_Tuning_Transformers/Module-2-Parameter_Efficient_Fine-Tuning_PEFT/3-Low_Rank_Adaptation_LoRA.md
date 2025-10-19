Low Rank Adapataion (LoRA)
    simplifies largea nd complex ml model 
    adds lightweightplugins 
    lets u reduce trainable paramsuses pre trained model with high dim matrices
    decreases trainign time, resource usage, memory footprint 

E.g. 
    if u have layers of 10 and then 8 neurons, u have 10x8 = 80 weights
    if u have 10 then 3 then 8, you have 10x3+3x8 = 30+24= only 54 neurons

Recall that we compute the outputs from one layer to the next is just a matrix multiplication 
$$
\vec{h} = W\vec{x} 
$$

To do our fine tuning, without LoRA we would have to retrain the whole matrix $W \in M_{m \times n} (\R)$, which would take a lot of computation.
With LoRA, we instead compute the subsequent layer $\vec{h}$ as follows:
$$
\vec{h} = W\vec{x}  +\Delta W\vec{x} = (W+\Delta W) \vec{x}
$$
with $\Delta W \in M_{m \times n}(\R)$, the same shapw as $W$. 

To find $\Delta W$, we use $\Delta W = BA$, two different matrices with times $m \times r$ and $ r \times n$ (to preserve the shape of $\Delta W$). 
$\\r$
is the lower dimensional rank.
With this new setup we only have to train $mr + rn = r(m+n)$ parameters, which will be less than the number of parameters $mn$ we would have had to train without LoRA. (assuming $r<\min(m,n)$). 

Now when we train, the original weight matrix $W$ remains frozen and we only have to train te values in $A$ and $B$. 

Optimizing Lora:
$\\$
We instead consider:
$$
\vec{h} = W\vec{x}  +{\alpha \over r}\Delta W\vec{x} = (W+{\alpha \over r}\Delta W) \vec{x}
$$
where $r$ is the lower dimensional rank and $\alpha$ is a hyperparameter. 

We can make our loss function rely only on the matrices $A$ and $B$, meaning only $\Delta W$
is updated through gradient descent. 

We can apply this technique anwyere in the transformer architecture, including in the feedforward layers, andthe $Q,K$ and $V$ matrices as well. 