from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/"results/phase3/adaptation_schedule_audit_20261117_v1"
REPORT=ROOT/"reports/phase3/adaptation_schedule_audit_20261117_v8"
RUN_V2=ROOT/"results/phase3/adaptation_schedule_audit_20261117_v2"
REPORT_V2=ROOT/"reports/phase3/adaptation_schedule_audit_20261117_v11"
RUN_V4=ROOT/"results/phase3/adaptation_schedule_audit_20261118_v4"
REPORT_V19=ROOT/"reports/phase3/adaptation_schedule_audit_20261118_v19"

def test_locked_schedule_run_has_all_paired_seed_cells_and_no_future_labels():
    batch=pd.read_csv(RUN/"online_batch_audit.csv")
    runs=pd.read_csv(RUN/"seed_metrics.csv")
    receipt=pd.read_json(RUN/"audit_receipt.json",typ="series")
    assert len(receipt["seed_bank"])==30
    assert len(runs)==30*2*3*3
    assert runs.groupby(["mechanism","delay","method"]).seed.nunique().eq(30).all()
    assert len(batch)==30*2*3*3*96
    assert batch.prediction_persisted.all()
    available=batch.max_available_at.notna()
    assert (batch.loc[available,"max_available_at"] < batch.loc[available,"batch"]).all()
    assert batch.updates.max()<=3
    stationary=batch[batch.mechanism=="M0_stationary"]
    assert stationary.loc[stationary.method=="global_error_trigger_refit","updates"].max()==0
    reversal=batch[batch.mechanism=="M4_abrupt_conditional_reversal"]
    assert reversal.loc[reversal.method=="global_error_trigger_refit","updates"].max()==3

def test_report_has_paired_seed_inference_editable_tex_and_pdf():
    paired=pd.read_csv(REPORT/"paired_accuracy_differences.csv")
    assert len(paired)==12
    assert paired.seeds.eq(30).all()
    assert (paired.lower<=paired["mean"]).all() and (paired["mean"]<=paired.upper).all()
    for name in ("adaptation_schedule_audit.tex","adaptation_schedule_audit.pdf","adaptation_schedule.pdf","adaptation_schedule.png","adaptation_schedule.svg"):
        assert (REPORT/name).is_file()

def test_region_local_run_is_label_safe_bounded_and_has_paired_evidence():
    batch=pd.read_csv(RUN_V2/"online_batch_audit.csv")
    runs=pd.read_csv(RUN_V2/"seed_metrics.csv")
    receipt=pd.read_json(RUN_V2/"audit_receipt.json",typ="series")
    assert len(receipt["seed_bank"])==30
    assert len(runs)==30*2*3*4
    assert runs.groupby(["mechanism","delay","method"]).seed.nunique().eq(30).all()
    assert len(batch)==30*2*3*4*96
    assert batch.prediction_persisted.all()
    available=batch.max_available_at.notna()
    assert (batch.loc[available,"max_available_at"]<batch.loc[available,"batch"]).all()
    local=batch[batch.method=="region_error_trigger_refit"]
    assert local.updates.max()<=3
    stationary_runs=runs[(runs.mechanism=="M0_stationary")&(runs.method=="region_error_trigger_refit")]
    assert stationary_runs.groupby("delay").update_count.sum().eq(1).all()
    reversal_runs=runs[(runs.mechanism=="M4_abrupt_conditional_reversal")&(runs.method=="region_error_trigger_refit")]
    assert reversal_runs.update_count.eq(3).all()
    assert runs[runs.method=="region_error_trigger_refit"].local_models_active.max()<=3
    assert "frozen-training-KMeans" in receipt["adaptation_scope"]

def test_region_local_report_records_all_paired_policy_contrasts():
    paired=pd.read_csv(REPORT_V2/"paired_accuracy_differences.csv")
    assert len(paired)==18
    assert paired.seeds.eq(30).all()
    assert (paired.lower<=paired["mean"]).all() and (paired["mean"]<=paired.upper).all()
    local=paired[paired.method=="region_error_trigger_refit"]
    reversal=local[local.mechanism=="M4_abrupt_conditional_reversal"].sort_values("delay")
    assert (reversal.lower>0).all()
    for name in ("adaptation_schedule_audit.tex","adaptation_schedule_audit.pdf","adaptation_schedule.pdf","adaptation_schedule.png","adaptation_schedule.svg"):
        assert (REPORT_V2/name).is_file()


def test_labelbroker_v4_audits_release_clocks_queue_drain_and_forecast_available_labels():
    receipt=json.loads((RUN_V4/"audit_receipt.json").read_text(encoding="utf-8"))
    runs=pd.read_csv(RUN_V4/"seed_metrics.csv")
    batches=pd.read_csv(RUN_V4/"online_batch_audit.csv")
    events=pd.read_csv(RUN_V4/"label_release_events.csv")
    assert receipt["protocol"].endswith("v4-labelbroker-drain-accounting")
    assert len(receipt["seed_bank"])==30 and receipt["seed_bootstrap_inference_valid"]
    assert len(runs)==30*2*3*4
    assert len(batches)==30*2*3*4*96
    assert len(events)==30*2*3*4*96
    assert runs.groupby(["seed","mechanism","delay","method"]).size().eq(1).all()
    assert runs.groupby("delay").released_labels_after_queue_drain.apply(
        lambda values: values.eq(48000).all()
    ).all()
    for delay,expected in {0:47500,5:45000,20:37500}.items():
        assert runs[runs.delay==delay].released_labels_by_last_forecast.eq(expected).all()
    for key,group in events.groupby(["seed","mechanism","delay","method"]):
        delay=int(key[2])
        assert sorted(group.source_batch.astype(int))==list(range(1,97))
        assert group.rows.sum()==48000 and group.ids_sha256.nunique()==96
        live=group[~group.queue_drain];drain=group[group.queue_drain]
        assert (live.available_at<=live.release_processed_clock).all()
        assert (live.prediction_clock>live.release_processed_clock).all()
        assert (live.source_batch<live.prediction_clock).all()
        assert drain.prediction_clock.isna().all()
        assert (drain.available_at<=drain.release_processed_clock).all()
        assert len(live)=={0:95,5:90,20:75}[delay]
        assert len(drain)=={0:1,5:6,20:21}[delay]
    for relative,digest in receipt["output_sha256"].items():
        assert hashlib.sha256((RUN_V4/Path(relative).name).read_bytes()).hexdigest()==digest


def test_v19_report_uses_simultaneous_seed_bootstrap_and_validates_render():
    contrasts=pd.read_csv(REPORT_V19/"paired_accuracy_differences.csv")
    receipt=json.loads((REPORT_V19/"report_receipt.json").read_text(encoding="utf-8"))
    assert len(contrasts)==18 and contrasts.family_size.eq(18).all()
    assert contrasts.ci_method.eq("joint paired-seed bootstrap max-|t|").all()
    assert contrasts.seeds.eq(30).all()
    assert (contrasts.lower<=contrasts["mean"]).all()
    assert (contrasts["mean"]<=contrasts.upper).all()
    local_reversal=contrasts[(contrasts.mechanism=="M4_abrupt_conditional_reversal") &
                             (contrasts.method=="region_error_trigger_refit")]
    assert local_reversal.lower.gt(0).all()
    stationary=contrasts[contrasts.mechanism=="M0_stationary"]
    assert (stationary.lower<=0).all() and (stationary.upper>=0).all()
    assert contrasts.zero_seed_variance.any()
    assert receipt["status"]=="compiled_and_visually_reviewed"
    assert receipt["bootstrap"]["replicates"]==20000
    assert receipt["page_count"]==2
    for name in ("adaptation_schedule_audit.tex","adaptation_schedule_audit.pdf",
                 "adaptation_schedule.png","adaptation_schedule.pdf","adaptation_schedule.svg"):
        assert (REPORT_V19/name).is_file()

