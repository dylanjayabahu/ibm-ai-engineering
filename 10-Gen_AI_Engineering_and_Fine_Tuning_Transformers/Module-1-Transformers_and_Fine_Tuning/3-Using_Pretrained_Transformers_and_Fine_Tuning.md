Transformer MOdels 
    BERT
    Llama
    GPT

Need large amounts of data for training to learn rich representations of language 
Billions of parameters, complex optimization, requires lots of hardware and computational power 

Fine tuning lets us use existing models for our purposes by taking the existing models an training them a bit more for our desired task 

Benefits of Fine Tuning
    Uses transfer learning
    Saves time and resource efficiency 
    tailored model responses 
    task-specific adaptatins 
    Allows better generalization for specificdomains

Pitfalls of Fine Tuning 
    Overfitting (aviod using small dataset or training too much on the new data)
    Underfitting (make sure there is still sufficient training to learn the required task; appropriate learning rate)
    Catastrophic forgetting (make sure model doesnt lose its initial broad knowledge from the pretraining)
    Data leaking (make sure u keep train/val/test data separate)

Challenges in Evaluating LLms pip install accelerate -
    humans can easily compare response to tell which one is better; but we cant rlly assign a numerical score
    FIne tuning can include reward modelling to align outputs with human preferences
        we assign values to the goodness of each response to align with what we would consider better or worse


Approaches to Fine Tuning 
    Self Supervised fine tuning 
        model predicts missing words or next words in large unlabeled datasets 
    
    SUpervised fine tuning 
        Uses labelled data for specific tasks like sentiment analysis
        
        Full fine tuning:
            all model parameters are updated 
        
        Parameter-Efficient FIne Tuning 
            only a subset of the parameters are updated 
    
    Reinforcement learning from human feedback (RLHF)
        MOdel adjusts outputs based on human feedback to align with human preferences 

Direct Preference Optimzation (DPO)
    EMerging method that optimizes models directly based on human preferences 
    Simpler than reinforcement learning; no reward model traning needed, has faster converges
    Focuses on human aligned output 

Hybrid fine tuning 
    combine self-supervised, suprvise,d andRLHF methods


