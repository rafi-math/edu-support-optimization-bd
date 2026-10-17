(function () {
  var d = document.documentElement;
  if (d.className.indexOf('js-reveal') === -1) { return; }
  if (window.__revealGuard) { clearTimeout(window.__revealGuard); }
  var mainEl = document.querySelector('main');
  var kids = mainEl ? mainEl.children : [];
  var els = [];
  for (var i = 0; i < kids.length; i++) { els.push(kids[i]); }
  if (!('IntersectionObserver' in window)) {
    for (var j = 0; j < els.length; j++) { els[j].className += ' is-visible'; }
    return;
  }
  var io = new IntersectionObserver(function (entries) {
    for (var k = 0; k < entries.length; k++) {
      if (entries[k].isIntersecting) {
        entries[k].target.className += ' is-visible';
        io.unobserve(entries[k].target);
      }
    }
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
  for (var m = 0; m < els.length; m++) { io.observe(els[m]); }
})();
(function () {
  var KEY = 'readScale', MIN = 0.9, MAX = 1.3, STEP = 0.1;
  function store(op, a, b) { try { return op === 'get' ? localStorage.getItem(a) : localStorage.setItem(a, b); } catch (e) { return null; } }
  function get() { var v = parseFloat(store('get', KEY)); return (v >= MIN && v <= MAX) ? v : 1; }
  function apply(v) { document.documentElement.style.setProperty('--rs', String(v)); }
  function clamp(v) { return Math.min(MAX, Math.max(MIN, Math.round(v * 10) / 10)); }
  apply(get());
  document.addEventListener('click', function (e) {
    var b = e.target && e.target.closest ? e.target.closest('.zoombar button') : null;
    if (!b) { return; }
    var cur = get(), nv = cur, z = b.getAttribute('data-zoom');
    if (z === 'in') { nv = clamp(cur + STEP); }
    else if (z === 'out') { nv = clamp(cur - STEP); }
    else if (z === 'reset') { nv = 1; }
    if (nv !== cur) { store('set', KEY, String(nv)); apply(nv); }
  });
})();
