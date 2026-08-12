---
layout: misc
title: Photography
---

<div class="ph-intro" markdown="1">
I shoot mostly on the street, mostly at slow shutter speeds. I like the moment where a scene is still legible but no longer still.
</div>

<div class="ph-wrap">
  <div class="ph-grid" id="phGrid">
    {% for p in site.data.photos %}
    <figure class="ph-item" data-index="{{ forloop.index0 }}" data-full="{{ site.github.url }}/assets/img/photos/{{ p.slug }}.jpg" data-title="{{ p.title }}" tabindex="0" role="button" aria-label="Open {{ p.title }}">
      <img src="{{ site.github.url }}/assets/img/photos/{{ p.slug }}-thumb.jpg"
           alt="{{ p.title }}" width="{{ p.w }}" height="{{ p.h }}" loading="lazy" decoding="async">
      <figcaption><span>{{ p.title }}</span></figcaption>
    </figure>
    {% endfor %}
  </div>
</div>

<div class="setar" markdown="1">

## Setar

I also play the [setar](https://en.wikipedia.org/wiki/Setar) and [shurangiz](https://en.wikipedia.org/wiki/Shurangiz). Below is an adaptation of *Gereyli* — roughly, Leyli's cry — played on setar.

<div class="video"><iframe src="https://www.youtube-nocookie.com/embed/jngB5C5YUdw" title="Gereyli, played on setar" loading="lazy" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>

</div>

<div class="ph-lb" id="phLb" role="dialog" aria-modal="true" aria-label="Photograph viewer" hidden>
  <button class="ph-close" id="phClose" aria-label="Close">&times;</button>
  <button class="ph-nav ph-prev" id="phPrev" aria-label="Previous">&#8249;</button>
  <button class="ph-nav ph-next" id="phNext" aria-label="Next">&#8250;</button>
  <div class="ph-stage"><img id="phImg" src="" alt=""></div>
  <div class="ph-meta"><span id="phTitle"></span><span id="phCount"></span></div>
</div>

<style>
.setar{max-width:600px;margin:3.4rem auto 0;padding-top:2.2rem;border-top:1px solid #e6e6e6}
.setar h2{margin-top:0}
.video{position:relative;padding-bottom:56.25%;height:0;margin-top:1.1rem}
.video iframe{position:absolute;inset:0;width:100%;height:100%;border:0}
.ph-intro{max-width:34em;margin:0 0 2.2rem;opacity:.72;font-size:.98rem;line-height:1.65}

/* full-bleed breakout from the theme container.
   --sbw is the scrollbar width, set from JS: 100vw includes the scrollbar,
   so subtracting it keeps the page from scrolling sideways. */
.ph-wrap{width:calc(100vw - var(--sbw,0px));margin-left:calc(50% - (100vw - var(--sbw,0px))/2);
  padding:0 clamp(10px,3vw,34px);box-sizing:border-box}

.ph-grid{column-count:3;column-gap:clamp(10px,1.5vw,20px)}
@media(max-width:1000px){.ph-grid{column-count:2}}
@media(max-width:620px){.ph-grid{column-count:1;column-gap:0}}

.ph-item{
  break-inside:avoid;margin:0 0 clamp(10px,1.5vw,20px);position:relative;
  cursor:pointer;overflow:hidden;background:#0d0d0d;border-radius:2px;
  outline:none;-webkit-tap-highlight-color:transparent
}
/* reveal-on-scroll only when JS is running, so photos are never hidden without it */
.ph-grid.js .ph-item{opacity:0;transform:translateY(26px);
  transition:opacity .8s cubic-bezier(.2,.7,.3,1),transform .8s cubic-bezier(.2,.7,.3,1)}
.ph-grid.js .ph-item.in{opacity:1;transform:none}
.ph-item img{display:block;width:100%;height:auto;
  transition:transform .9s cubic-bezier(.2,.7,.3,1),filter .6s ease;
  filter:saturate(.92) contrast(1.02)}
.ph-item:hover img,.ph-item:focus img{transform:scale(1.045);filter:saturate(1) contrast(1.05)}
.ph-item:focus{box-shadow:0 0 0 2px #fff,0 0 0 4px #111}

.ph-item figcaption{
  position:absolute;left:0;right:0;bottom:0;padding:2.6rem .95rem .8rem;
  background:linear-gradient(to top,rgba(0,0,0,.78),rgba(0,0,0,0));
  opacity:0;transform:translateY(8px);transition:opacity .45s ease,transform .45s ease;
  pointer-events:none}
.ph-item:hover figcaption,.ph-item:focus figcaption{opacity:1;transform:none}
.ph-item figcaption span{
  color:#fff;font-size:.68rem;letter-spacing:.22em;text-transform:uppercase;font-weight:600}
@media(hover:none){.ph-item figcaption{opacity:1;transform:none}}

/* lightbox */
.ph-lb{position:fixed;inset:0;z-index:9999;background:rgba(8,8,8,.97);
  display:flex;align-items:center;justify-content:center;
  opacity:0;transition:opacity .32s ease;backdrop-filter:blur(6px)}
.ph-lb.on{opacity:1}
.ph-lb[hidden]{display:none}  /* the hidden attribute must beat display:flex */
.ph-stage{width:100%;height:100%;display:flex;align-items:center;justify-content:center;
  padding:clamp(16px,4vw,56px) clamp(16px,7vw,90px);box-sizing:border-box}
.ph-stage img{max-width:100%;max-height:100%;object-fit:contain;
  box-shadow:0 30px 90px rgba(0,0,0,.6);
  opacity:0;transform:scale(.985);transition:opacity .4s ease,transform .4s ease}
.ph-stage img.ready{opacity:1;transform:none}
.ph-meta{position:absolute;left:0;right:0;bottom:clamp(10px,2.5vh,26px);
  display:flex;justify-content:center;gap:1.1rem;color:rgba(255,255,255,.62);
  font-size:.66rem;letter-spacing:.22em;text-transform:uppercase}
.ph-close,.ph-nav{position:absolute;background:none;border:0;color:rgba(255,255,255,.62);
  cursor:pointer;line-height:1;padding:.35em .5em;transition:color .2s ease,transform .2s ease}
.ph-close:hover,.ph-nav:hover{color:#fff}
.ph-close{top:clamp(6px,2vh,20px);right:clamp(8px,2vw,26px);font-size:2.1rem}
.ph-nav{top:50%;transform:translateY(-50%);font-size:3rem}
.ph-nav:hover{transform:translateY(-50%) scale(1.12)}
.ph-prev{left:clamp(2px,1.4vw,24px)}
.ph-next{right:clamp(2px,1.4vw,24px)}
@media(max-width:620px){.ph-nav{font-size:2.1rem}.ph-stage{padding:12px 8px 56px}}
body.ph-open{overflow:hidden}
@media(prefers-reduced-motion:reduce){
  .ph-grid.js .ph-item,.ph-item img,.ph-stage img{transition:none;opacity:1;transform:none}
}
</style>

<script>
(function(){
  function sbw(){
    document.documentElement.style.setProperty('--sbw',
      (window.innerWidth - document.documentElement.clientWidth) + 'px');
  }
  sbw();
  window.addEventListener('resize', sbw);

  var items=[].slice.call(document.querySelectorAll('.ph-item'));
  if(!items.length) return;
  var lb=document.getElementById('phLb'), img=document.getElementById('phImg'),
      tEl=document.getElementById('phTitle'), cEl=document.getElementById('phCount'), cur=0;

  // staggered reveal on scroll
  var grid=document.getElementById('phGrid');
  if('IntersectionObserver' in window){
    grid.classList.add('js');
    var io=new IntersectionObserver(function(es){
      es.forEach(function(e,i){
        if(e.isIntersecting){
          var el=e.target;
          setTimeout(function(){el.classList.add('in');}, (i%3)*90);
          io.unobserve(el);
        }
      });
    },{rootMargin:'0px 0px -8% 0px',threshold:.08});
    items.forEach(function(el){io.observe(el);});
  } else { items.forEach(function(el){el.classList.add('in');}); }

  function preload(i){ if(items[i]){var p=new Image();p.src=items[i].dataset.full;} }

  function show(i){
    cur=(i+items.length)%items.length;
    var el=items[cur];
    img.classList.remove('ready');
    var n=new Image();
    n.onload=function(){ img.src=n.src; img.alt=el.dataset.title; img.classList.add('ready'); };
    n.src=el.dataset.full;
    tEl.textContent=el.dataset.title;
    cEl.textContent=(cur+1)+' / '+items.length;
    preload(cur+1); preload(cur-1);
  }
  function open(i){ lb.hidden=false; document.body.classList.add('ph-open');
    requestAnimationFrame(function(){lb.classList.add('on');}); show(i); }
  function close(){ lb.classList.remove('on'); document.body.classList.remove('ph-open');
    setTimeout(function(){lb.hidden=true; img.src='';},320); }

  items.forEach(function(el,i){
    el.addEventListener('click',function(){open(i);});
    el.addEventListener('keydown',function(e){
      if(e.key==='Enter'||e.key===' '){e.preventDefault();open(i);} });
  });
  document.getElementById('phClose').addEventListener('click',close);
  document.getElementById('phPrev').addEventListener('click',function(e){e.stopPropagation();show(cur-1);});
  document.getElementById('phNext').addEventListener('click',function(e){e.stopPropagation();show(cur+1);});
  lb.addEventListener('click',function(e){ if(e.target===lb||e.target.className==='ph-stage') close(); });
  document.addEventListener('keydown',function(e){
    if(lb.hidden) return;
    if(e.key==='Escape') close();
    else if(e.key==='ArrowRight') show(cur+1);
    else if(e.key==='ArrowLeft') show(cur-1);
  });
  // swipe
  var x0=null,y0=null;
  lb.addEventListener('touchstart',function(e){x0=e.touches[0].clientX;y0=e.touches[0].clientY;},{passive:true});
  lb.addEventListener('touchend',function(e){
    if(x0===null) return;
    var dx=e.changedTouches[0].clientX-x0, dy=e.changedTouches[0].clientY-y0;
    if(Math.abs(dx)>45&&Math.abs(dx)>Math.abs(dy)) show(cur+(dx<0?1:-1));
    else if(dy>90) close();
    x0=null;y0=null;
  },{passive:true});
})();
</script>
