(() => {
  const assert = (value, message) => { if (!value) throw new Error(message); };
  let comparisons = 0;
  for (const row of window.IDM.data.online) {
    for (const delay of [0, 5, 20]) {
      for (let clock = 1; clock <= 24 + delay; clock++) {
        const visible = window.IDM.visibleVerification(row, delay, clock);
        assert(Boolean(visible) === (row.batch + delay <= clock), 'release boundary mismatch');
        if (visible) {
          assert(visible.cohort === row.cohort && visible.delay === delay, 'cohort/delay mismatch');
          assert(visible.available_at <= clock, 'future truth exposed');
        }
        comparisons++;
      }
    }
  }
  const set = (id, value) => {
    const el = document.getElementById(id);
    el.value = value;
    el.dispatchEvent(new Event('input', {bubbles: true}));
  };
  set('seed', 100); set('condition', 'C3'); set('delay', 20);
  // Automation/native select implementations may dispatch change without input.
  document.getElementById('delay').value='5';
  document.getElementById('delay').dispatchEvent(new Event('change',{bubbles:true}));
  assert(document.getElementById('clockValue').textContent.includes('/ 29'), 'change-only select event ignored');
  set('delay',20);
  set('clock', 24); set('batch', 9);
  assert(document.getElementById('verifiedValue').textContent === '—', 'unreleased accuracy displayed');
  assert(!document.getElementById('detail').textContent.includes('Macro-F1:'), 'unreleased F1 displayed');
  set('clock', 29);
  assert(document.getElementById('verifiedValue').textContent !== '—', 'due labels not displayed');
  set('clock', 1); set('batch', 9);
  assert(document.getElementById('detail').textContent.includes('Input has not arrived'), 'future input displayed');
  set('delay', 5); set('clock', 16); set('batch', 9);
  const target = [...document.querySelectorAll('#chart [data-batch="11"]')].at(-1);
  target.dispatchEvent(new MouseEvent('click', {bubbles: true}));
  assert(document.getElementById('batch').value === '11', 'shared selection failed');
  assert(document.getElementById('cohort').textContent === '100:C3:11', 'detail cohort diverged');
  set('seed', 129); set('condition', 'C6'); set('batch', 24); set('clock', 29);
  assert(document.getElementById('cohort').textContent === '129:C6:24', 'selector mismatch');
  assert(document.getElementById('verifiedValue').textContent !== '—', 'drained label absent');
  set('seed',100);set('condition','C3');set('delay',5);set('clock',16);set('batch',9);
  return {status:'passed',cohorts:window.IDM.data.online.length,release_boundary_comparisons:comparisons,
    checks:['all delay/release boundaries','change-only select handling','unreleased accuracy hidden','unreleased F1 hidden',
      'future inputs hidden','shared SVG cohort selection','seed/scenario switching','queue-drained truth visible'],
    user_agent:navigator.userAgent};
})()
