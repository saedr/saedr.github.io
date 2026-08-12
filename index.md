---
layout: default
title: Home
---

<div class="hero">
  <img src="{{ site.github.url }}/assets/img/photos/hero.jpg"
       srcset="{{ site.github.url }}/assets/img/photos/hero-sm.jpg 900w, {{ site.github.url }}/assets/img/photos/hero.jpg 2000w"
       sizes="100vw" alt="" fetchpriority="high" decoding="async">
  <div class="hero-scrim"></div>
  <div class="hero-text">
    <h1>Saed Rezayi</h1>
    <p>NLP research scientist &middot; trustworthy evaluation of language models</p>
  </div>
</div>

<div class="lede" markdown="1">
I'm an NLP research scientist working on language models in high-stakes settings, currently at [NBME](https://www.nbme.org/research/meet-our-experts).

My research starts from a simple idea: almost everything we believe about what AI systems can do rests on a measurement. I study whether those measurements are telling us the truth, and how to rebuild them when they are not. This matters especially when AI systems evaluate other AI systems, because a failure may come from the system being tested or from the way we measured it.
</div>

<div class="cards">
  <a class="card" href="{{ site.github.url }}/pages/research.html">
    <span class="card-k">Research</span>
    <span class="card-t">Judge validity, explainability, fairness, robustness, item generation</span>
  </a>
  <a class="card" href="{{ site.github.url }}/pages/publications.html">
    <span class="card-k">Publications</span>
    <span class="card-t">20+ papers spanning model evaluation, fairness, explainability, and knowledge-infused NLP</span>
  </a>
  <a class="card" href="{{ site.github.url }}/pages/about.html">
    <span class="card-k">About</span>
    <span class="card-t">Background, and how I ended up working on this</span>
  </a>
  <a class="card" href="{{ site.github.url }}/pages/photography.html">
    <span class="card-k">Photography</span>
    <span class="card-t">Street work, mostly at the wrong shutter speed</span>
  </a>
</div>

<style>
.hero{position:relative;width:100vw;margin-left:calc(50% - 50vw);margin-bottom:2.4rem;
  line-height:0;background:#111;overflow:hidden}
.hero img{width:100%;height:auto;display:block;max-height:48vh;object-fit:cover;
  filter:saturate(.85) contrast(1.04) brightness(.94)}
.hero-scrim{position:absolute;inset:0;
  background:linear-gradient(to top,rgba(0,0,0,.72) 0%,rgba(0,0,0,.25) 42%,rgba(0,0,0,0) 72%)}
.hero-text{position:absolute;left:0;right:0;bottom:0;
  padding:0 clamp(18px,5vw,60px) clamp(16px,3.4vw,38px);line-height:1.15}
.hero-text h1{margin:0 0 .38rem;color:#fff;font-weight:600;
  font-size:clamp(1.75rem,4.6vw,3.1rem);letter-spacing:.012em;
  text-shadow:0 2px 22px rgba(0,0,0,.45)}
.hero-text p{margin:0;color:rgba(255,255,255,.82);font-size:clamp(.66rem,1.5vw,.82rem);
  letter-spacing:.2em;text-transform:uppercase;text-shadow:0 1px 14px rgba(0,0,0,.5)}
.lede{max-width:600px;margin:0 auto 2rem;padding:0 10px}
.lede p{margin-bottom:1rem}
.lede p:first-child{font-size:1.1rem}
.cards{max-width:600px;margin:0 auto 2.5rem;padding:0 10px;display:grid;gap:.7rem;
  grid-template-columns:repeat(2,1fr)}
@media(max-width:560px){.cards{grid-template-columns:1fr}.hero img{max-height:38vh}}
.card{display:block;padding:.85rem 1rem;border:1px solid #e4e4e4;border-radius:3px;
  text-decoration:none;transition:border-color .2s ease,transform .2s ease,box-shadow .2s ease}
.card:hover{border-color:#2E4A62;transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.06)}
.card-k{display:block;font-weight:600;font-size:.95rem;color:#1a1a1a;margin-bottom:.15rem}
.card-t{display:block;font-size:.8rem;line-height:1.45;color:#6b7280}
@media(prefers-reduced-motion:reduce){.card{transition:none}.card:hover{transform:none}}
</style>
