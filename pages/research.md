---
layout: misc
title: Research
---

Everything below is one question in different settings: **does this evaluation measure what it claims to measure?**

That is the validity question. Assessment has studied it for a century; machine learning is now rediscovering it under other names. Fairness, faithfulness, and robustness are not three separate research areas here — they are three ways an evaluation can fail to be valid, and I approach them with the same tools.

A full list of papers is on the [publications]({{ site.github.url }}/pages/publications.html) page.

---

## When the evaluator is part of the measurement

Using a model to judge another model's output produces a number that reflects two things at once: the property we wanted to measure, and the judge's own fit to the input it was shown. These are routinely reported as if they were only the first.

<img src="{{ site.github.url }}/assets/img/research/judge-decomposition.svg" alt="A reported evaluation score splits into evaluator mismatch and genuine model behavior" />


I develop methods that separate them. One approach uses distillation to build a judge adapted to the exact input it will score, then compares it against the conventional same-model judge. The gap between the two estimates how much of the reported failure was evaluator mismatch rather than a genuine property of the output. In the settings I have studied, that share is large enough to change what the evaluation concludes.

A related result concerns reproducibility. I ran a panel of independently trained judges over identical predictions and measured how much the verdict depended on which judge was used. Judges disagreed with one another on the same items substantially more than the items differed among themselves, and a variance decomposition attributed more variance to the judge and to the judge-by-condition interaction than to the item itself. The practical consequence: a single-judge evaluation result is not reproducible in the way it is usually reported, and the judge's training regime belongs in the results section alongside the score.

## Explainability: does the evidence support the claim?

Scoring systems increasingly emit evidence alongside their scores — the span of text that supposedly justifies the decision. Whether that evidence actually supports the label is a separate empirical question from whether the label is correct, and a system can be right for reasons it cannot defend.

<img src="{{ site.github.url }}/assets/img/research/evidence-sufficiency.svg" alt="The same text span receives different verdicts when read in context versus read alone" />


I audit this with perturbation methods: reading the evidence in isolation, masking it, replacing it with matched controls, paraphrasing it counterfactually, and checking agreement across independently trained readers. The recurring finding is that sufficiency is not a property of the evidence alone. It is a joint property of the evidence and the reader, which means the standard sufficiency and comprehensiveness metrics measure something considerably less intrinsic than they are usually taken to measure.

I also lead an organization-wide effort to make explainability an accountable property of deployed scoring systems rather than an aspiration, including a metric suite for evidence defensibility and a validation study against expert judgment.

## Fairness as a validity problem

Fairness evidence is often reported as a table of group-wise metrics with no account of what would count as a problem. This is not a rigor failure so much as a framing failure: a fairness claim means something only when it coheres with a validity argument and an explicit account of risk, including which error direction is costly and to whom.

<img src="{{ site.github.url }}/assets/img/research/fairness-validity.svg" alt="Validity claim, risk account, and detection contrast must align for a fairness claim to hold" />


This work includes cost-sensitive audits of a deployed scoring system, a position paper proposing an alignment standard among validity claims, risk accounts, and detection contrasts, and a reporting checklist and detection typology intended to be usable rather than aspirational.

## Robustness under adversarial pressure

Assessment systems face a real adversary: people who obtain content in advance, or who copy from one another. I build simulation frameworks and detection models for these problems, with emphasis on locating where detection breaks down rather than only demonstrating where it works.

<img src="{{ site.github.url }}/assets/img/research/leak-detection.svg" alt="A signal shifting at an estimated leak onset point on a timeline" />


Recent results include mapping the identifiability boundary for joint detection of compromised content and affected respondents, and a temporal change-point method that localizes when a leak began — validated against a bank of adversarial null conditions constructed specifically to produce false alarms.

## Evaluating generated content at scale

Generating assessment content with language models moves the bottleneck from writing to verification. I designed a layered evaluation pipeline for model-generated clinical cases, running from deterministic constraint checks through model-based quality gates to bank-level near-duplicate detection, and specified it to double as a verifiable reward function — so the same checks that gate content can also train the generator.

<img src="{{ site.github.url }}/assets/img/research/content-gates.svg" alt="Generated cases passing through deterministic, model-based, and deduplication gates" />


---

## Earlier work: knowledge-infused language models

My doctoral research addressed a different problem. Pretrained encoders underperform in specialized domains where the relevant knowledge is structured and external to the text. I built methods for injecting that knowledge into language models, producing domain-adapted models for agriculture (AgriBERT) and clinical radiation oncology (ClinicalRadioBERT), a framework for enriching knowledge graphs from external text (EDGE), and a cross-domain short-text clustering method (XDC).
