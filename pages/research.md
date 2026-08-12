---
layout: misc
title: Research
---

My work sits at the intersection of NLP and measurement. Below are the threads I am currently pursuing. A full list of papers is on the [publications]({{ site.github.url }}/pages/publications.html) page.

## Evaluating the evaluators

When we judge a model's output with another model, the resulting number reflects two things at once: the property we wanted to measure, and the judge's own fit to the input it was shown. These are routinely conflated.

I develop methods that separate them. One line of work uses distillation to build a judge adapted to the exact input it will be scored on, then compares it against the conventional same-model judge. The difference between the two is an estimate of how much of the reported "failure" was the judge's mismatch rather than a genuine property of the output. In the settings I have studied, that share is large enough to change what the evaluation concludes.

A related result: I ran a panel of judges over identical predictions and measured how much the verdict depended on which judge was used. Judges disagreed with each other on the same items substantially more than the items differed from one another, and a variance decomposition attributed more variance to the judge and the judge-by-condition interaction than to the item itself. This has a direct consequence for practice — a single-judge evaluation result is not reproducible in the way it is usually reported, and papers should report the judge's training regime alongside the score.

## Faithfulness and rationale evaluation

Automated scoring systems increasingly emit evidence alongside their scores: the span of text that supposedly justifies the decision. Whether that evidence actually supports the label is a separate empirical question from whether the label is correct.

I audit this with perturbation methods — reading the evidence in isolation, masking it, replacing it with matched controls, paraphrasing it counterfactually, and checking agreement across independently trained readers. The recurring finding is that sufficiency is not a property of the evidence alone. It is a joint property of the evidence and the reader, which means the standard sufficiency and comprehensiveness metrics are measuring something less intrinsic than they are usually taken to measure.

## Fairness and validity in automated scoring

Fairness evidence in automated scoring is often reported as a table of group-wise metrics with no account of what would count as a problem. I argue that fairness claims only mean something when they cohere with a validity argument and an explicit account of risk — including which error direction is costly and to whom.

This work includes cost-sensitive audits of a deployed scoring system, a position paper proposing an alignment standard among validity claims, risk accounts, and detection contrasts, and a reporting checklist and detection typology intended to be usable rather than aspirational.

## Adversarial detection and test security

Assessment systems face a genuine adversary: people who obtain content in advance, or copy from one another. I build simulation frameworks and detection models for these problems, with an emphasis on identifying where detection breaks down rather than only where it works.

Recent results include mapping the identifiability boundary for joint detection of compromised content and affected respondents, and a temporal change-point method that localizes when a leak began — validated against a bank of adversarial null conditions designed to produce false alarms.

## Knowledge-infused language models

My doctoral research addressed a different problem: pretrained encoders underperform in specialized domains where the relevant knowledge is structured and external to the text. I built methods for injecting that knowledge into language models, producing domain-adapted models for agriculture (AgriBERT) and clinical radiation oncology (ClinicalRadioBERT), a framework for enriching knowledge graphs from external text (EDGE), and a cross-domain short-text clustering method (XDC).
