---
layout: post
title: "GNN for Reasoning - part 1"
author: "Saed Rezayi"
categories: NN
tags: [GNN]
image: reason.webp
---

**QUESTION:**  Can we use Graph Neural Networks for reasoning?

Short answer: Yes

Long answer: If we define *reasoning* as "finding new facts over the knowledge graph" then the question is how can GNN be used in graph inference?

Since we are dealing with GNN and inference, two areas are involved: Neural Networks and Probabilistic Graphical Models. In fact there should be an integration between this two concepts.

## Inference
Computing the likelihood of observed data in models with latent variables. For instance in the following graphical model the goal is to find $P(Z|X)$.
![](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571713473507_Screenshot+from+2019-10-21+23-04-14.png)

- Approaches:
    - Exact inference: usually not tractable
    - Sampling: Does not scale
    - Approximate inference: Variational inference


## Variational Inference
Given a model, the goal is to find distributions for the unobserved variables (posterior inference). Using Bayes theorem: $P(Z|X)=\frac{P(X|Z)P(Z)}{P(X)}=\frac{P(X,Z)}{P(X)}$. Computing this probability usually involves hard-to-solve integrals with no analytical solution.

One approach to solve this problem is called **Variational Inferences.** Variational Inference solves this problem by finding a distribution $Q$ that approximates the true distribution $P$. The idea behind variational inference is this: let's just perform inference on an easy, parametric distribution $Q_{\phi}(Z|X)$ (like a Gaussian) for which we know how to do posterior inference, but adjust the parameters $\phi$ so that $Q_{\phi}$ is as **close** to $P$ as possible.
![](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571706706043_Screenshot+from+2019-10-21+21-11-35.png)

We can uses KL-divergence as a measure of how well our approximation fits the true posterior.
