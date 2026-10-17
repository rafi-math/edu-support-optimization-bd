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
  var KEY = 'zoomVal', MIN = 0.5, MAX = 1.5;
  function store(op, a, b) { try { return op === 'get' ? localStorage.getItem(a) : localStorage.setItem(a, b); } catch (e) { return null; } }
  function scaleFrom(v) { return MIN + (v / 100) * (MAX - MIN); }
  function apply(v) {
    var s = scaleFrom(v);
    document.documentElement.style.setProperty('--rs', String(s));
    var lab = document.getElementById('zoomLabel');
    if (lab) { lab.textContent = Math.round(s * 100) + '%'; }
  }
  var r = document.getElementById('zoomRange');
  var saved = parseFloat(store('get', KEY));
  if (!isFinite(saved) || saved < 0 || saved > 100) { saved = 50; }
  if (r) { r.value = String(saved); }
  apply(saved);
  if (r) { r.addEventListener('input', function () { var v = parseFloat(r.value); store('set', KEY, String(v)); apply(v); }); }
})();
