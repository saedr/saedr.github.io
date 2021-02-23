# GNN for Reasoning
**QUESTION:**  Can we use Graph Neural Networks for reasoning?
Short answer: Yes
Long answer: If we define *reasoning* as “finding new facts over the knowledge graph” then the question is how can GNN be used in graph inference?

- Since we are dealing with GNN and inference, two areas are involved: Neural Networks and Probabilistic Graphical Models. 
- In fact there should be an integration between this two concepts.
# Inference
- Computing the likelihood of observed data in models with latent variables.
- For instance in the following graphical model the goal is to find $$P(Z|X)$$.
![](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571713473507_Screenshot+from+2019-10-21+23-04-14.png)

- Approaches:
    - Exact inference: usually not tractable
    - Sampling: Does not scale
    - Approximate inference: Variational inference
## Variational Inference
- Given a model, the goal is to find distributions for the unobserved variables (posterior inference). 


- Using Bayes theorem: $$P(Z|X)=\frac{P(X|Z)P(Z)}{P(X)}=\frac{P(X,Z)}{P(X)}$$. Computing this probability usually involves hard-to-solve integrals with no analytical solution.


- One approach to solve this problem is called **Variational Inferences.** Variational Inference solves this problem by finding a distribution $$Q$$ that approximates the true distribution $$P$$.


- The idea behind variational inference is this: let's just perform inference on an easy, parametric distribution $$Q_{\phi}(Z|X)$$ (like a Gaussian) for which we know how to do posterior inference, but adjust the parameters $$\phi$$ so that $$Q_{\phi}$$ is as **close** to $$P$$ as possible.
![](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571706706043_Screenshot+from+2019-10-21+21-11-35.png)

- We can uses $$\text{KL}$$-divergence as a measure of how well our approximation fits the true posterior.
----------
## Kullback-Leibler divergence
- $$\text{KL}$$-divergence is a non-symmetric measure of the difference between two probability distributions $$P$$ and $$Q$$


                                    $$\text{KL}(Q||P)=-\sum\limits_z Q(Z)\log\frac{P(Z|X)}{Q(Z)}$$
                                
- The goal is to minimize this function so we have:
                                    $$\text{KL}(Q||P)=-\sum Q(Z)\log \frac{\frac{P(X,Z)}{P(X)}}{\frac{Q(Z)}{1}}$$
                                
                                                        $$=-\sum Q(Z)\log\frac{P(X,Z)}{Q(Z)}.\frac{1}{P(X)}$$
                                
                                                        $$=-\sum Q(Z)\left[\log\frac{P(X,Z)}{Q(Z)}-\log P(X)\right]$$
                                
                                                        $$=-\sum Q(Z)\log\frac{P(X,Z)}{Q(Z)}+\sum Q(Z)\log P(X)$$
    
    since the summation is over $$Z$$ we can take out $$P(X)$$ from the second sum and we know that $$\sum_z Q(Z)=1$$, thus we can write:
                                
                                  $$\log P(X)=\text{KL}(Q||P)+\sum Q(Z)\log\frac{P(X,Z)}{Q(Z)}$$
                                
    Note that minimizing the $$\text{KL}$$-divergence is equivalent to maximizing the second term.
    
- The second term in the above equation is called **Variational Lower Bound**
                                
                                                  $$\mathcal{L}=\sum Q(Z)\log\frac{P(X,Z)}{Q(Z)}$$
    
    since $$\text{KL}$$-divergence is non-negative we can write $$\mathcal{L}\leq\log P(X)$$ and hence the *lower bound* name*.*
                          $$\mathcal{L}=\sum Q(Z)\log\frac{P(X,Z)}{P(Z)}=\sum Q(Z)\log\frac{P(X|Z)P(Z)}{Q(Z)}$$


                                                                    $$=\sum Q(Z)\left[ \log P(X|Z) + \log\frac{P(Z)}{Q(Z)} \right]$$


                                                                    $$=\sum Q(Z)\log P(X|Z)+\sum Q(Z)\log\frac{P(Z)}{Q(Z)}$$
                                
                                                                    $$=\mathbb{E}_{Q}\left[\log P(X|Z)\right]-\text{KL}(Q(Z|X)||P(Z))$$


- Now we can model this with an encoder-decoder (autoencoder) architecture as follows where the encoder is $$Q(Z|X)$$ and the decoder is modeled by $$P(X|Z)$$. 
![](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571720037783_Screenshot+from+2019-10-22+00-53-40.png)

- We can think of the first part of the objective function as reconstruction error and the second part as a constraint on $$P(Z)$$ (i.e., $$p(Z)$$ should be similar to $$Q(Z)$$


- This idea was used in `Kingma & Welling, ICLR``'``14` [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)


----------

**RECAP**: We found a relation between Neural Networks and graphical models, how is this related to reasoning?

- There is a third piece which is **logic** and it is used to obtain new facts. 
- First let’s see what is the relation between **logic** and **graphical models**.
----------
# Markov Logic Networks
- `Richardson & Domingos, JML'06` [Markov logic networks](https://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf)
- Generating MLN based on knowledge base and additional hidden facts using logic rules.


## Logic Rules
- Example of logic rules:
![From MLN paper By Richardson & Domingos in JML’06](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571787070439_Screenshot+from+2019-10-22+19-30-26.png)



- Examples of KB and MLN:


![From ICLR’20 paper](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571790161547_Screenshot+from+2019-10-22+20-21-53.png)



----------

Now the relation between GNN, PGM, and Logic

# ICLR’20 paper

`Zhang et al., ICLR``'``20` [Efficient Probabilistic Logic Reasoning With Graph Neural Networks](https://openreview.net/pdf?id=rJg76kStwH)

## Model
- MLN can be defined as a joint distribution over all observed facts $$\mathcal{O}$$ and unobserved facts $$\mathcal{H}$$ as:
                                $$P(\mathcal{O},\mathcal{H})=\frac{1}{Z}\exp\left(\sum\limits_f w_f\sum\limits_{a_f}\phi_f(a_f)\right)$$
    
    where $$Z$$ is a normalization constant, $$w_f$$ is a formula weight, $$a_f$$ is called an assignment, e.g., $$a_r=(\text{Obama}, \text{U.S.})$$,  and $$\phi(.)$$ is a logic formula on a pair of entities.


- The exact inference of above distribution is computationally intractable, but we can optimize the following objective function as explained in [variational inference](https://paper.dropbox.com/doc/GNN-for-Reasoning--AnJLPGz_nVCYI3CshldJ8VQBAg-xLVoRB9NXfZDR6vdeVl5k#:uid=297909922505565975691373&h2=Variational-Inference).


                        $$\log P(\mathcal{O})\geq\mathbb{E}_Q\left[\log P_w(\mathcal{O},\mathcal{H})\right]-\mathbb{E}_Q\left[\log Q_{\theta}(\mathcal{H}|\mathcal{O})\right]$$
    
    where $$Q(\mathcal{H}|\mathcal{O})=\prod\limits_{r(a_r)\in\mathcal{H}}Q(r(a_r)|\mathcal{O})$$


- To optimize this function $$\text{EM}$$ algorithm is employed. In the E-step $$P$$ is fixed and $$Q$$ is parameterized by deep learning models. (GNN+MLP) and in the M-step they fix $$Q$$ and learn the  $$w$$ weights.


- How is it different from pLogicNet? structurally similar entities get similar embeddings while they are not similar in the MLN graph. To avoid this  they train separate GNN for different clusters (entities belong to different categories)
## Experiment
- Graph completion task on FreeBase
![](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571847391918_Screenshot+from+2019-10-23+12-16-19.png)

## Code
- We can use [Neural Logic Programming](https://github.com/fanyangxyz/Neural-LP) (`Yang et al., NIPS'17`) tool to generate candidate rules.
- [ExpressGNN](https://github.com/expressGNN/ExpressGNN)
- [Cora dataset](https://sites.google.com/site/semanticbasedregularization/home/software/experiments_on_cora)


# Future Plan
- Generate rules for MetaQA dataset.
- Apply ExpressGNN on the result. 

