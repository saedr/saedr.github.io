---
layout: misc
title: Research
---

My work asks a general question across fairness, faithfulness, robustness, and model evaluation: does the evidence we measure actually support the conclusion we want to draw?

<div class="toc" markdown="1">
**In short.** Evaluations that use models to evaluate models are partly measuring the evaluator — I build methods that separate the two ([evaluators](#evaluators)). The evidence a scorer emits often does not survive being read on its own ([explainability](#explainability)). Fairness claims mean little without an explicit account of risk ([fairness](#fairness)). Detection methods should be tested on where they break, not only where they work ([robustness](#robustness)). And when models draft new test items, expert raters disagree about which drafts are usable ([item generation](#generation)).
</div>

Papers are listed on the [publications]({{ site.github.url }}/pages/publications.html) page.

<style>
.toc{border-left:3px solid #2E4A62;background:#f7f8f9;padding:.85rem 1.05rem;
  margin:1.6rem 0;font-size:.88rem;line-height:1.62}
.toc p{margin:0}
.yr{display:block;margin:-.35rem 0 .9rem;font-size:.68rem;letter-spacing:.16em;
  text-transform:uppercase;color:#8a8f98}
</style>

---

## When the evaluator is part of the measurement {#evaluators}

<span class="yr">2025–present</span>

Using a model to evaluate another model's output produces a number that reflects two things at once: the property we wanted to measure, and the evaluator's own fit to the input it was shown. These are routinely reported as if they were only the first.

<img src="{{ site.github.url }}/assets/img/research/evaluator-decomposition.svg" alt="A reported evaluation score splits into evaluator mismatch and genuine model behavior" />


I develop methods that separate them. One approach uses distillation to build an evaluator adapted to the exact input it will score, then compares it against the conventional same-model evaluator. The gap between the two estimates how much of the reported failure was evaluator mismatch rather than a genuine property of the output. In the settings I have studied, that share is large enough to change what the evaluation concludes.

A related result concerns reproducibility. I ran a panel of independently trained evaluators over identical predictions and measured how much the verdict depended on which evaluator was used. Evaluators disagreed with one another on the same items substantially more than the items differed among themselves, and a variance decomposition attributed more variance to the evaluator and to the evaluator-by-condition interaction than to the item itself. The practical consequence: a single-evaluator evaluation result is not reproducible in the way it is usually reported, and the evaluator's training regime belongs in the results section alongside the score.

---

## Explainability: does the evidence support the claim? {#explainability}

<span class="yr">2025–present</span>

Scoring systems increasingly emit evidence alongside their scores — the span of text that supposedly justifies the decision. Whether that evidence actually supports the label is a separate empirical question from whether the label is correct, and a system can be right for reasons it cannot defend.

<img src="{{ site.github.url }}/assets/img/research/evidence-sufficiency.svg" alt="The same text span receives different verdicts when read in context versus read alone" />


I audit this with perturbation methods: reading the evidence in isolation, masking it, replacing it with matched controls, paraphrasing it counterfactually, and checking agreement across independently trained readers. The recurring finding is that sufficiency is not a property of the evidence alone. It is a joint property of the evidence and the reader, which means the standard sufficiency and comprehensiveness metrics measure something considerably less intrinsic than they are usually taken to measure.

I also lead an organization-wide effort to make explainability an accountable property of deployed scoring systems rather than an aspiration, including a metric suite for evidence defensibility and a validation study against expert judgment.

---

## Fairness as a validity problem {#fairness}

<span class="yr">2024–2026</span>

Fairness evidence is often reported as a table of group-wise metrics with no account of what would count as a problem. This is not a rigor failure so much as a framing failure: a fairness claim means something only when it coheres with a validity argument and an explicit account of risk, including which error direction is costly and to whom.

<img src="{{ site.github.url }}/assets/img/research/fairness-validity.svg" alt="Validity claim, risk account, and detection contrast must align for a fairness claim to hold" />


This work includes cost-sensitive audits of a deployed scoring system, a position paper proposing an alignment standard among validity claims, risk accounts, and detection contrasts, and a reporting checklist and detection typology intended to be usable rather than aspirational.

---

## Robustness under adversarial pressure {#robustness}

<span class="yr">2025–present</span>

Assessment systems face a real adversary: people who obtain content in advance, or who copy from one another. I build simulation frameworks and detection models for these problems, with emphasis on locating where detection breaks down rather than only demonstrating where it works.

<img src="{{ site.github.url }}/assets/img/research/leak-detection.svg" alt="A signal shifting at an estimated leak onset point on a timeline" />


Recent results include mapping the identifiability boundary for joint detection of compromised content and affected respondents, and a temporal change-point method that localizes when a leak began — validated against a bank of adversarial null conditions constructed specifically to produce false alarms.

---

## Generating new items, and judging whether they are any good {#generation}

<span class="yr">2024–2026</span>

Assessment organizations constantly need new test items, and writing them is slow expert work. I work on whether language models can draft them, and on the harder question of how you would know whether a draft is any good.

<img src="{{ site.github.url }}/assets/img/research/counterfactual-items.svg" alt="A source vignette transformed through a counterfactual step into a new chart item, with expert ratings below" />

One approach tells the model that some findings in a source case were recorded incorrectly and asks it to rebuild a clinically coherent record, which lands on a different diagnosis. Items produced this way sit measurably further from their source material than straightforward paraphrasing, so the model is inventing rather than rewording. Roughly a quarter came back free of major flaws, and about half were considered useful starting points.

The evaluation is where this connects to the rest of the page. Expert raters disagreed with each other substantially about which drafts were acceptable, and in a related study rater differences accounted for about a third of the variance in quality ratings. As with model evaluators, the measured quality of a generated item depends heavily on who is measuring it.

---

## Earlier work: knowledge-infused language models {#earlier}

<span class="yr">2020–2023</span>

My doctoral research addressed a different problem. Pretrained encoders underperform in specialized domains where the relevant knowledge is structured and external to the text. I built methods for injecting that knowledge into language models, producing domain-adapted models for agriculture (AgriBERT) and clinical radiation oncology (ClinicalRadioBERT), a framework for enriching knowledge graphs from external text (EDGE), and a cross-domain short-text clustering method (XDC).
