---
layout: post
title: "GNN for Reasoning - part 2"
author: "Saed Rezayi"
categories: NN
tags: [GNN]
image: kld.png
---

KL-divergence is a non-symmetric measure of the difference between two probability distributions $P$ and $Q$

$$\text{KL}(Q||P)=-\sum\limits_z Q(Z)\log\frac{P(Z|X)}{Q(Z)}$$

The goal is to minimize this function so we have:

$$\text{KL}(Q||P)=-\sum Q(Z)\log \frac{\frac{P(X,Z)}{P(X)}}{\frac{Q(Z)}{1}}$$

$$=-\sum Q(Z)\log\frac{P(X,Z)}{Q(Z)}.\frac{1}{P(X)}$$

$$=-\sum Q(Z)\left[\log\frac{P(X,Z)}{Q(Z)}-\log P(X)\right]$$

$$=-\sum Q(Z)\log\frac{P(X,Z)}{Q(Z)}+\sum Q(Z)\log P(X)$$

since the summation is over $Z$ we can take out $P(X)$ from the second sum and we know that $\sum_z Q(Z)=1$, thus we can write:

$$\log P(X)=\text{KL}(Q||P)+\sum Q(Z)\log\frac{P(X,Z)}{Q(Z)}$$

Note that minimizing the KL-divergence is equivalent to maximizing the second term. The second term in the above equation is called **Variational Lower Bound**

$$\mathcal{L}=\sum Q(Z)\log\frac{P(X,Z)}{Q(Z)}$$

since KL-divergence is non-negative we can write $\mathcal{L}\leq\log P(X)$ and hence the *lower bound* name.

$$\mathcal{L}=\sum Q(Z)\log\frac{P(X,Z)}{P(Z)}=\sum Q(Z)\log\frac{P(X|Z)P(Z)}{Q(Z)}$$

$$=\sum Q(Z)\left[ \log P(X|Z) + \log\frac{P(Z)}{Q(Z)} \right]$$

$$=\sum Q(Z)\log P(X|Z)+\sum Q(Z)\log\frac{P(Z)}{Q(Z)}$$

$$=\mathbb{E}_{Q}\left[\log P(X|Z)\right]-\text{KL}(Q(Z|X)||P(Z))$$

Now we can model this with an encoder-decoder (autoencoder) architecture as follows where the encoder is $Q(Z|X)$ and the decoder is modeled by $P(X|Z)$. 
![](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571720037783_Screenshot+from+2019-10-22+00-53-40.png)

We can think of the first part of the objective function as reconstruction error and the second part as a constraint on $P(Z)$ (i.e., $p(Z)$ should be similar to $Q(Z)$). This idea was used in `Kingma & Welling, ICLR, 2014` [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)

