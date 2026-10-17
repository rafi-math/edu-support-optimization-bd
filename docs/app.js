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
