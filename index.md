---
layout: misc
title: About Me
---

I am an NLP research scientist working on **trustworthy evaluation of language models**.

My research asks one question: does an evaluation measure what it claims to measure? That is the validity question, and it is not a side concern in machine learning — it is the thing that determines whether any reported number means anything. Fairness, explainability, and robustness are not separate topics on my list. They are instances of the validity question, and I treat them that way.

The problem has become urgent because we increasingly evaluate models with other models. LLM judges, automated scorers, and faithfulness metrics all introduce a second system between us and the thing we wanted to know. When that happens, a reported failure can belong to the model under test or to the evaluator, and the standard metrics do not distinguish them. My recent work shows this confusion occurring in practice, quantifies how much of a typical result it accounts for, and develops methods to separate the two.

I care about evaluations that survive their own audit. In several cases that has meant withdrawing my own findings once a stronger design showed they were artifacts of the measurement rather than properties of the model.

This research happens at the [National Board of Medical Examiners](https://www.nbme.org/research/meet-our-experts), where I work on automated scoring of clinical communication. The setting matters to the work. Scores here are used in the assessment of physicians, which means the measurement is subject to expert human validation, adversarial pressure, and the kind of scrutiny that most evaluation research never encounters. Methods that hold up under those conditions tend to hold up elsewhere.

I bring one thing to this that is uncommon in NLP: psychometrics, the discipline that has spent a century on the question of when a measurement can be trusted. Much of what evaluation research is currently rediscovering — reliability across raters, construct-irrelevant variance, generalizability of a score beyond the conditions it was observed under — has established theory behind it. I use it.

Before NBME I completed my PhD in Computer Science at the University of Georgia under [Sheng Li](https://sheng-li.org/). During my doctoral work and internships at Adobe Research and Reuters, my research centered on improving machine comprehension by integrating auxiliary knowledge into language models.

Outside of work I play the setar and shoot street photography.
