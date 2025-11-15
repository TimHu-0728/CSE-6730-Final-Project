// ui/app.js
(() => {
  const pv = document.getElementById('pvVideo');
  if (!pv) return;

  const setRate = () => { pv.playbackRate = 0.5; };

  pv.addEventListener('loadedmetadata', setRate);
  pv.addEventListener('play', setRate);

  // Re-apply the rate when the tab regains focus (some browsers reset it)
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) setRate();
  });
})();

// app.js (append)
(() => {
  // Ensure all videos try to play muted if autoplay is blocked
  document.querySelectorAll('video').forEach(v => {
    const tryPlay = () => v.play().catch(() => { /* ignore */ });
    v.muted = true;               // required for autoplay
    v.addEventListener('loadedmetadata', tryPlay);
  });

  // Reduce motion for users who prefer it
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReduced) {
    document.querySelectorAll('video').forEach(v => { v.pause(); v.controls = true; });
  }
})();
