"""Build a standalone, label-safe v2 evidence replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FIELDS = ("task", "seed", "condition", "model", "batch", "cohort", "reference_accuracy",
          "observed_at", "input_alarm", "original_alarm", "latent_rejection", "min_adjusted_p",
          "cal_estimate", "cal_lower", "cal_upper", "cal_concern", "cal_concern_flag",
          "raw_estimate", "raw_lower", "raw_upper", "raw_concern", "atc_estimate")


def build(run: Path, output: Path):
    run, output = run.resolve(), output.resolve()
    online = pd.read_csv(run / "online_batches.csv")
    releases = pd.read_csv(run / "released_verification.csv")
    records = online[list(FIELDS)].to_dict(orient="records")
    labels = releases[["task", "seed", "condition", "model", "batch", "delay", "available_at",
                       "accuracy", "macro_f1", "harm", "class0_n", "class1_n"]].to_dict(orient="records")
    data = json.dumps({"online": records, "releases": labels}, separators=(",", ":"), allow_nan=False)
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 3 Evidence Replay</title>
<style>
:root{color-scheme:light;--ink:#1c2a31;--muted:#64747b;--line:#d8e1e2;--paper:#f3f6f4;--card:#fff;--blue:#176b87;--teal:#438b78;--amber:#d98c5f;--violet:#7868a6}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Segoe UI,Arial,sans-serif}header{padding:28px clamp(20px,5vw,64px);background:#173b49;color:#f6fbfa}h1{font-size:clamp(24px,3vw,36px);line-height:1.12;margin:0 0 8px}header p{margin:0;color:#c8dcdd}.wrap{max-width:1280px;margin:auto;padding:22px clamp(16px,4vw,48px) 48px}.controls,.grid{display:grid;gap:14px}.controls{grid-template-columns:repeat(auto-fit,minmax(145px,1fr));background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 8px 22px #213b4210}.controls label{font-size:12px;font-weight:700;color:var(--muted);display:grid;gap:5px}select,input{width:100%;padding:9px 10px;border:1px solid #bdcbce;border-radius:8px;background:white;color:var(--ink);font:inherit}select:focus,input:focus{outline:3px solid #176b8730;border-color:var(--blue)}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;min-width:0;box-shadow:0 8px 22px #213b4210}.panel h2{font-size:17px;margin:0 0 12px}.hero{display:flex;align-items:baseline;gap:10px;margin:4px 0 12px}.hero strong{font-size:28px;color:var(--blue)}.hero span,.subtle{color:var(--muted);font-size:13px}.kv{display:grid;grid-template-columns:1fr auto;gap:6px 14px;font-variant-numeric:tabular-nums}.kv b{font-weight:650}.badge{display:inline-block;padding:3px 9px;border-radius:999px;background:#e4eeec;font-size:12px;font-weight:700}.badge.warn{background:#faeee5;color:#8e4a25}.badge.ok{background:#e3f0e9;color:#21654c}.timeline{overflow:auto;max-height:360px}table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:7px 6px;border-bottom:1px solid #e6eceb;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{position:sticky;top:0;background:#f7f9f8}.wide{grid-column:1/-1}.notice{margin:16px 0 0;padding:12px 14px;border-left:4px solid var(--amber);background:#fff8f2;color:#68452f;font-size:13px}.empty{color:var(--muted);padding:10px 0}.spark{width:100%;height:72px;margin-top:6px}.legend{display:flex;gap:14px;font-size:12px;color:var(--muted);margin-bottom:6px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;background:var(--blue)}.dot.truth{background:var(--amber)}@media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style></head><body>
<header><h1>Phase 3: Research Implementation and Experimental Evaluation</h1><p>Ahmed Bilal (23I-2581) · Umar Behram (23I-2604) · Offline evidence replay</p></header>
<main class="wrap"><section class="controls" aria-label="Shared evidence selection">
<label>Task<select id="task"></select></label><label>Seed<select id="seed"></select></label><label>Condition<select id="condition"></select></label><label>Model<select id="model"></select></label><label>Label delay<select id="delay"><option>0</option><option>5</option><option>20</option></select></label><label>As-of clock<input id="clock" type="number" min="1" max="44" value="24"></label><label>Batch<select id="batch"></select></label></section>
<section class="grid">
<article class="panel"><h2>Prospective estimate</h2><div id="estimate"></div></article>
<article class="panel"><h2>Scheduled verification</h2><div id="verification"></div></article>
<article class="panel"><h2>Calibration diagnostic</h2><div id="calibration"></div><div class="legend"><span><i class="dot"></i>Model estimate</span><span><i class="dot truth"></i>Released accuracy</span></div><svg id="spark" class="spark" viewBox="0 0 600 72" role="img" aria-label="Estimate and released accuracy by batch"></svg></article>
<article class="panel"><h2>Input evidence</h2><div id="input"></div></article>
<article class="panel wide"><h2>Same stream at the selected clock</h2><div id="history" class="timeline"></div></article></section>
<p class="notice">This offline replay embeds retrospective release records and filters displayed truth using the selected availability clock. It is not a security boundary or a human-subject evaluation. An unresolved cohort has no displayed accuracy or F1.</p></main>
<script id="evidence-data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('evidence-data').textContent), E=id=>document.getElementById(id), key=(r,d)=>[r.task,r.seed,r.condition,r.model,r.batch,d].join('|');
const releases=new Map(D.releases.map(r=>[key(r,r.delay),r]));
const options=(el,values,selected)=>{const old=el.value;el.replaceChildren(...values.map(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;return o}));el.value=values.includes(old)?old:(values.includes(selected)?selected:values[0]||'')};
const html=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function refreshOptions(){const task=E('task').value;const rows=D.online.filter(r=>r.task===task);options(E('seed'),[...new Set(rows.map(r=>String(r.seed)))].sort((a,b)=>+a-+b),'');const seed=+E('seed').value;const sr=rows.filter(r=>r.seed===seed);options(E('condition'),[...new Set(sr.map(r=>r.condition))].sort(),'');const condition=E('condition').value;const cr=sr.filter(r=>r.condition===condition);options(E('model'),[...new Set(cr.map(r=>r.model))].sort(),'logistic');const model=E('model').value;options(E('batch'),cr.filter(r=>r.model===model).map(r=>String(r.batch)).sort((a,b)=>+a-+b),'1')}
function recRows(){return D.online.filter(r=>r.task===E('task').value&&r.seed===+E('seed').value&&r.condition===E('condition').value&&r.model===E('model').value).sort((a,b)=>a.batch-b.batch)}
function asOf(){return Math.max(1,Math.min(44,+E('clock').value||1))}
function fmt(v,n=4){return Number.isFinite(+v)?(+v).toFixed(n):'—'}
function refresh(){refreshOptions();const clock=asOf(),delay=+E('delay').value,batch=+E('batch').value,rows=recRows(),r=rows.find(x=>x.batch===batch);if(!r)return;const visible=r.observed_at<=clock;const v=releases.get(key(r,delay));const verified=!!v&&v.available_at<=clock;
E('estimate').innerHTML=visible?`<div class="hero"><strong>${fmt(r.cal_estimate)}</strong><span>calibrated AC · 95% plug-in interval ${fmt(r.cal_lower)}–${fmt(r.cal_upper)}</span></div><div class="kv"><span>Raw AC</span><b>${fmt(r.raw_estimate)}</b><span>ATC-style</span><b>${fmt(r.atc_estimate)}</b><span>Historical reference accuracy</span><b>${fmt(r.reference_accuracy)}</b><span>Concern proxy / flag</span><b>${fmt(r.cal_concern)} / <span class="badge ${r.cal_concern_flag?'warn':'ok'}">${r.cal_concern_flag?'flagged':'not flagged'}</span></b></div>`:'<p class="empty">This batch has not been observed by the selected clock.</p>';
E('verification').innerHTML=verified?`<div class="hero"><strong>${fmt(v.accuracy)}</strong><span>accuracy · available at clock ${v.available_at}</span></div><div class="kv"><span>Macro F1</span><b>${fmt(v.macro_f1)}</b><span>Model-specific realised harm</span><b>${v.harm?'yes':'no'}</b><span>Class counts</span><b>${v.class0_n} / ${v.class1_n}</b><span>Evidence age at selected clock</span><b>${clock-v.batch} batches</b></div>`:`<p><span class="badge warn">unresolved</span></p><p class="subtle">Release scheduled at clock ${batch+delay}. Truth is hidden until then.</p>`;
if(verified){const err=Math.abs(r.cal_estimate-v.accuracy),covered=r.cal_lower<=v.accuracy&&v.accuracy<=r.cal_upper;E('calibration').innerHTML=`<div class="kv"><span>Released accuracy</span><b>${fmt(v.accuracy)}</b><span>Absolute estimation error</span><b>${fmt(err)}</b><span>Interval covered</span><b>${covered?'yes':'no'}</b></div>`}else E('calibration').innerHTML='<p class="empty">Reliability for this cohort is unavailable before label release.</p>';
E('input').innerHTML=visible?`<div class="kv"><span>Full KS/PCA alarm</span><b>${r.input_alarm?'yes':'no'}</b><span>Original-feature KS alarm</span><b>${r.original_alarm?'yes':'no'}</b><span>Latent-coordinate rejection</span><b>${r.latent_rejection?'yes':'no'}</b><span>Minimum Holm-adjusted p</span><b>${fmt(r.min_adjusted_p,6)}</b></div>`:'<p class="empty">No input evidence before the cohort is observed.</p>';
const past=rows.filter(x=>x.batch<=clock);const history=past.map(x=>{const release=releases.get(key(x,delay)),shown=!!release&&release.available_at<=clock;return `<tr><td>${x.batch}</td><td>${fmt(x.cal_estimate)}</td><td>${x.input_alarm?'alarm':'—'}</td><td>${shown?fmt(release.accuracy):'<span class="subtle">unresolved</span>'}</td><td>${shown?fmt(release.macro_f1):'<span class="subtle">hidden</span>'}</td><td>${shown?release.harm?'harm':'no harm':'—'}</td></tr>`}).join('');E('history').innerHTML=`<table><thead><tr><th>Batch</th><th>Calibrated estimate</th><th>Input evidence</th><th>Accuracy</th><th>Macro F1</th><th>Verified harm</th></tr></thead><tbody>${history}</tbody></table>`;
const svg=E('spark'),w=600,h=72,known=past.filter(x=>{const z=releases.get(key(x,delay));return z&&z.available_at<=clock});const pts=(arr,field,color)=>arr.length?`<polyline fill="none" stroke="${color}" stroke-width="2" points="${arr.map((x,i)=>`${(i/(Math.max(1,past.length-1)))*(w-8)+4},${h-8-(+x[field])*(h-16)}`).join(' ')}"/>`:'';svg.innerHTML=pts(past,'cal_estimate','#176b87')+pts(known,'accuracy','#d98c5f');}
for(const id of ['task','seed','condition','model','delay','clock','batch']){E(id).addEventListener('input',refresh);E(id).addEventListener('change',refresh)}
options(E('task'),[...new Set(D.online.map(r=>r.task))].sort(),'v1_synthetic');E('clock').addEventListener('input',refresh);E('clock').addEventListener('change',refresh);refresh();
</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__DATA__", data), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.run, args.out)
