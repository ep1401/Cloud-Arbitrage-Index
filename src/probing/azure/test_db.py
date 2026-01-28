from meta_cai_scheduler_azure import log_probe_result
import datetime

log_probe_result(
    provider="azure",
    region="eastus",
    instance_type="Standard_D2s_v3",
    probe_kind="micro",
    meta_policy="fail_fast",
    max_minutes=60,
    outcome="SmokeTest",
    instance_id="smoke-test-1",
    start_time=datetime.datetime.utcnow(),
    end_time=datetime.datetime.utcnow(),
    duration_minutes=0.1,
    interrupted=False,
    interrupt_bin=None,
    survived_hours=0,
    spot_price_usd=0.01,
    price_delta_6h=0.0,
    sampling_propensity=1.0,
    policy_version="smoke",
    pred_h1_at_launch=0.1,
    pred_risk_5h_at_launch=0.2,
    features_snapshot={"ok": True},
)
print("done")
