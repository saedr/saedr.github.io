---
layout: post
title: "GNN for Reasoning"
author: "Saed Rezayi"
categories: NN
tags: [GNN]
image: reason.png
---

**QUESTION:**  Can we use Graph Neural Networks for reasoning?
Short answer: Yes
Long answer: If we define *reasoning* as “finding new facts over the knowledge graph” then the question is how can GNN be used in graph inference?

- Since we are dealing with GNN and inference, two areas are involved: Neural Networks and Probabilistic Graphical Models. 
- In fact there should be an integration between this two concepts.
# Inference
- Computing the likelihood of observed data in models with latent variables.
- For instance in the following graphical model the goal is to find \(P(Z|X)\).
![](https://paper-attachments.dropbox.com/s_ACD8843D58A10AD799AE266302228E63B7692ABF84D737D63C8DB752F21F149B_1571713473507_Screenshot+from+2019-10-21+23-04-14.png)

