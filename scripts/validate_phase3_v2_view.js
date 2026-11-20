const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

const fs = require('node:fs');
const vm = require('node:vm');
const path = process.argv[2];
if (!path) throw new Error('usage: node validate_phase3_v2_view.js <evidence.html>');
const html = fs.readFileSync(path, 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
assert(scripts.length >= 2, 'standalone replay scripts missing');
const embedded = html.match(/<script id="evidence-data" type="application\/json">([\s\S]*?)<\/script>/);
assert(embedded, 'embedded replay records missing');
const initial = { task: '', seed: '', condition: '', model: '', delay: '0', clock: '24', batch: '1' };

class Element {
  constructor(id) {
    this.id = id;
    this.value = initial[id] || '';
    this.textContent = '';
    this.innerHTML = '';
    this.children = [];
    this.listeners = {};
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatchEvent(event) { for (const fn of this.listeners[event.type] || []) fn(event); }
  replaceChildren(...children) {
    this.children = children;
    if (!children.some(child => child.value === this.value)) this.value = children[0]?.value || '';
  }
}

const ids = ['task', 'seed', 'condition', 'model', 'delay', 'clock', 'batch', 'estimate',
  'verification', 'calibration', 'input', 'history', 'spark', 'evidence-data'];
const elements = Object.fromEntries(ids.map(id => [id, new Element(id)]));
elements['evidence-data'].textContent = embedded[1];
class EventStub { constructor(type) { this.type = type; } }
const context = {
  document: {
    getElementById: id => elements[id],
    createElement: tag => ({ tagName: tag, value: '', textContent: '' }),
  },
  Event: EventStub,
  Map, JSON, Math, String, Number, Set, Array,
  console,
};
vm.runInNewContext(scripts.at(-1)[1], context, { filename: path });
const set = (id, value, event = 'input') => {
  elements[id].value = String(value);
  elements[id].dispatchEvent(new EventStub(event));
};

set('task', 'v1_synthetic');
set('seed', 100);
set('condition', 'C3');
set('model', 'logistic');
set('batch', 9);
set('delay', 20);
set('clock', 24);
assert(elements.verification.innerHTML.includes('unresolved'), 'delay-20 truth was exposed before release');
assert(elements.history.innerHTML.includes('unresolved'), 'history exposed an unreleased cohort');
set('delay', 5, 'change');
assert(elements.verification.innerHTML.includes('available at clock 14'), 'change-only delay selection did not refresh');
set('delay', 20, 'change');
assert(elements.verification.innerHTML.includes('unresolved'), 'delay change did not hide future truth again');
set('clock', 29, 'change');
assert(elements.verification.innerHTML.includes('available at clock 29'), 'scheduled label absent at release');
set('clock', 1);
assert(elements.estimate.innerHTML.includes('not been observed'), 'future input was visible');
set('clock', 44);
set('batch', 24);
assert(elements.verification.innerHTML.includes('available at clock 44'), 'drained queue did not expose due label');
assert(elements.verification.innerHTML.includes('20 batches'), 'drained evidence age is incorrect');
set('task', 'xor_synthetic');
assert(elements.seed.value === '200', 'task change did not select an XOR seed');
assert(elements.condition.value === 'X0', 'task change did not select a valid condition');

process.stdout.write(JSON.stringify({ status: 'passed', checks: [
  'same-cohort panel refresh', 'input/change event handling', 'future inputs hidden',
  'delay changes update every panel', 'unreleased accuracy and F1 hidden',
  'release-clock reveal', 'queue drainage through clock 44', 'evidence age at release',
  'task/seed/condition switching'
]}));
