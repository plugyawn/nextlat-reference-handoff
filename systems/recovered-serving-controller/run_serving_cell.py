#!/usr/bin/env python3
"""One checkpoint, existing Slim M8 serving path, natural-EOS Countdown.

Reuses the admitted server arguments/lifecycle. Only this client differs from
the old forced-length benchmark: preserve natural EOS and compare complete IDs.
Warmup/load/teardown and the separate HBM replay are outside timed requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--ssh-target")
    p.add_argument("--local-transport", action="store_true")
    p.add_argument("--target-model", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--overlay", required=True)
    p.add_argument("--hot-map", required=True)
    p.add_argument("--image", default="nextlat-sglang:800m-canary")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--checkpoint-identity", type=Path, required=True,
                   help="Local export receipt binding this exact target and sidecar")
    p.add_argument("--phase", choices=("smoke", "aggregate"), default="smoke")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--port", type=int, default=31128)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--no-sudo", action="store_true")
    a = p.parse_args()
    if not a.local_transport and not a.ssh_target:
        p.error("choose --local-transport or --ssh-target")
    if not 1 <= a.repetitions <= 5 or not 2 <= a.max_new_tokens <= 64:
        p.error("use 1..5 repetitions and a 2..64 token safety cap")
    a.sandbox_id, a.direct_runtime = None, False
    a.attention_backend, a.m8_fp8_slim = "fa3", True
    a.memory_sample_ms = 50
    return a


def requests(a, rows, label, sample_memory=False):
    sampling = {**base.SAMPLING, "ignore_eos": False,
                "max_new_tokens": a.max_new_tokens}
    code = f"""
import base64,json,math,subprocess,threading,time,urllib.request
rows=json.loads({json.dumps(rows)!r})
sampling=json.loads({json.dumps(sampling)!r})
samples=[]; errors=[]; stop=threading.Event()
def read_memory():
    try:
        v=float(subprocess.run(['nvidia-smi','-i',{str(a.gpu)!r},'--query-gpu=memory.used','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10).stdout.strip())
        if not math.isfinite(v) or v<0: raise ValueError('invalid memory')
        samples.append(v)
    except Exception as e: errors.append(str(e)); stop.set()
def monitor():
    while not stop.is_set(): read_memory(); stop.wait(0.05)
thread=threading.Thread(target=monitor) if {sample_memory!r} else None
if thread: thread.start()
out=[]
try:
    for row in rows:
        rid={label!r}+'-o'+str(row['ordinal'])
        payload={{'rid':[rid],'input_ids':[row['input_ids']],'sampling_params':[sampling],'return_logprob':False,'stream':False,'log_metrics':True}}
        req=urllib.request.Request('http://127.0.0.1:{a.port}/generate',data=json.dumps(payload,separators=(',',':')).encode(),headers={{'Content-Type':'application/json'}})
        begin=time.perf_counter()
        with urllib.request.urlopen(req,timeout=180) as r: response=json.load(r)
        elapsed=time.perf_counter()-begin
        if not isinstance(response,list) or len(response)!=1: raise RuntimeError('not B1')
        item=response[0]; ids=item.get('output_ids'); meta=item.get('meta_info')
        if not isinstance(ids,list) or not 1<=len(ids)<={a.max_new_tokens}: raise RuntimeError('invalid natural-EOS output')
        if not isinstance(meta,dict) or meta.get('id')!=rid: raise RuntimeError('request identity mismatch')
        if meta.get('completion_tokens')!=len(ids): raise RuntimeError('completion counter mismatch')
        finish=meta.get('finish_reason',{{}}).get('type')
        if finish not in ('stop','length'): raise RuntimeError('abnormal finish '+str(meta))
        if finish=='length' and len(ids)!={a.max_new_tokens}: raise RuntimeError('unexpected length finish')
        if meta.get('cached_tokens',0)!=0: raise RuntimeError('unexpected prefix-cache hit')
        out.append({{'ordinal':row['ordinal'],'problem_hash':row['problem_hash'],'output_ids':ids,'text':item.get('text'),'meta_info':meta,'client_wall_seconds':elapsed}})
finally:
    if thread:
        stop.set(); thread.join(timeout=15)
        if thread.is_alive(): raise RuntimeError('memory thread did not stop')
        read_memory()
value={{'rows':out,'memory':{{'peak_mib':max(samples) if samples else None,'sample_count':len(samples),'errors':errors,'timing_excluded':{sample_memory!r}}}}}
print({serving.RESULT_BEGIN!r})
print(base64.b64encode(json.dumps(value).encode()).decode())
print({serving.RESULT_END!r})
"""
    r = base.remote_python(a, code, timeout=max(240, len(rows) * 185))
    if r.returncode:
        raise RuntimeError(r.stdout[-5000:])
    result = serving._decode_remote_json(r.stdout, label=label)
    if sample_memory and (result["memory"]["errors"] or not result["memory"]["sample_count"]):
        raise RuntimeError(f"HBM sampling failed: {result['memory']}")
    return result


def run_arm(a, rows, arm, repetition):
    label = f"{a.phase}-{arm}-r{repetition}"
    path = a.output_dir / f"{label}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("contract_sha256") != a.contract_sha256:
            raise RuntimeError(f"refuse stale completed receipt {path}")
        return old
    name = f"nextlat-scaling-{label}-{int(time.time())}"
    cmd = (serving._exact_command(a, name=name) if arm == "exact" else
           serving._spec_command(a, name=name, window=8))
    serving._assert_command(cmd, window=None if arm == "exact" else 8)
    cmd.remove("--rm")  # retain the owned crash log until copied
    at = cmd.index("--entrypoint")
    cmd[at:at] = ["-v", f"{a.cache_dir}:/root/.cache"]
    out = {"arm": arm, "repetition": repetition, "command": cmd,
           "contract_sha256": a.contract_sha256}
    handle = None
    try:
        idle = serving._gpu_memory_mib(a)
        handle = base.launch_docker(a, cmd)
        print(f"LAUNCH {label} {handle['id']}", flush=True)
        check = f"""
import json,subprocess,time,urllib.request
end=time.monotonic()+420
while time.monotonic()<end:
    state=subprocess.run({base.docker_prefix(a.no_sudo)!r}+['inspect','--format','{{{{.State.Running}}}}',{handle['id']!r}],check=True,capture_output=True,text=True).stdout.strip()
    if state!='true': raise SystemExit('owned server exited')
    try:
        with urllib.request.urlopen('http://127.0.0.1:{a.port}/model_info',timeout=2) as r: json.load(r)
        raise SystemExit(0)
    except Exception: time.sleep(1)
raise SystemExit('readiness timeout')
"""
        r = base.remote_python(a, check, timeout=435)
        if r.returncode:
            raise RuntimeError(r.stdout[-3000:])
        out["warmup"] = requests(a, rows, label + "-warm")
        out["server_ready"] = serving._server_observation(a)
        ready = serving._gpu_memory_mib(a)
        out["measurement"] = requests(a, rows, label + "-timed")
        out["server_after_timing"] = serving._server_observation(a)
        base.write_json(path.with_suffix(".partial.json"), out)
        if repetition == 0:
            memory = requests(a, rows, label + "-hbm", sample_memory=True)
            out["hbm"] = {"idle_mib": idle, "ready_mib": ready,
                          **memory["memory"], "authority": "nvidia-smi memory.used"}
        out["status"] = "executed"
        base.write_json(path.with_suffix(".partial.json"), out)
    finally:
        if handle:
            logs = base.remote_script(a, shlex.join([*base.docker_prefix(a.no_sudo), "logs", handle["id"]]), timeout=30, check=False)
            (a.output_dir / f"{label}.log").write_text(logs.stdout)
            base.stop(a, handle)
            serving._assert_arm_released(a, handle)
            base.remote_script(a, shlex.join([*base.docker_prefix(a.no_sudo), "rm", handle["id"]]), timeout=30)
        if out.get("status") == "executed":
            out["cleanup"] = "GPU and port release verified"
            base.write_json(path, out)
    return out


def compare(exact, spec):
    a, b = exact["measurement"]["rows"], spec["measurement"]["rows"]
    if len(a) != len(b):
        raise RuntimeError("response count differs")
    divergences, histogram = [], [0] * 9
    total = matches = verifies = accepted = proposed = overhang = 0
    for x, y in zip(a, b):
        if (x["ordinal"], x["problem_hash"]) != (y["ordinal"], y["problem_hash"]):
            raise RuntimeError("problem identity differs")
        xi, yi = x["output_ids"], y["output_ids"]
        total += len(xi)
        matches += sum(t == u for t, u in zip(xi, yi))
        if xi != yi or x["meta_info"]["finish_reason"] != y["meta_info"]["finish_reason"]:
            first = next((i for i, (t, u) in enumerate(zip(xi, yi)) if t != u), min(len(xi), len(yi)))
            divergences.append({"ordinal": x["ordinal"], "first_index": first,
                                "exact_ids": xi, "spec_ids": yi,
                                "exact_finish": x["meta_info"]["finish_reason"],
                                "spec_finish": y["meta_info"]["finish_reason"]})
        m = y["meta_info"]
        v, c, p, h = (m.get(k) for k in ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_correct_drafts_histogram"))
        if not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in (v, c, p)):
            raise RuntimeError(f"missing speculation counters: {m}")
        if not isinstance(h, list) or len(h) > 9 or sum(h) != v or sum(i*n for i, n in enumerate(h)) != c:
            raise RuntimeError("acceptance histogram disagrees")
        tail = 1 + v + c - len(yi)
        if not 0 <= tail <= 8 or p < c:
            raise RuntimeError(f"EOS/tail/counter mismatch: {m}")
        for i, n in enumerate(h):
            histogram[i] += n
        verifies += v; accepted += c; proposed += p; overhang += tail
    if verifies <= 0 or proposed <= 0:
        raise RuntimeError("no actual speculative execution on this fixture")
    exact_s = sum(x["client_wall_seconds"] for x in a)
    spec_s = sum(x["client_wall_seconds"] for x in b)
    if not all(math.isfinite(v) and v > 0 for v in (exact_s, spec_s)):
        raise RuntimeError("invalid client timing")
    return {"strict": not divergences, "matching_tokens": matches,
            "exact_tokens": total, "spec_tokens": sum(len(y["output_ids"]) for y in b),
            "divergences": divergences, "exact_seconds": exact_s,
            "spec_seconds": spec_s, "exact_tps": total / exact_s,
            "spec_tps": sum(len(y["output_ids"]) for y in b) / spec_s,
            "speedup": exact_s / spec_s, "verify_calls": verifies,
            "exact_length_capped_requests": sum(x["meta_info"]["finish_reason"]["type"] == "length" for x in a),
            "spec_length_capped_requests": sum(y["meta_info"]["finish_reason"]["type"] == "length" for y in b),
            "accepted_drafts": accepted, "reported_proposed_drafts": proposed,
            "acceptance_histogram": histogram, "eos_tail_overhang": overhang,
            "post_prefill_tokens_per_verify": (sum(len(y["output_ids"]) for y in b) - len(b)) / verifies,
            "counter_accepted_drafts_per_verify": accepted / verifies}


def main():
    global serving, base
    a = arguments()
    sys.path.insert(0, str(a.repo_root / ".opencode"))
    import nextlat_b1_window_select as serving
    base = serving.base
    if a.local_transport:
        def local_script(args, script, *, timeout, check=True):
            return subprocess.run(["bash", "-s"], input=script, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=check)
        def local_python(args, code, *, timeout):
            return subprocess.run([sys.executable, "-c", code], text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  timeout=timeout)
        base.remote_script, base.remote_python = local_script, local_python
    a.output_dir.mkdir(parents=True, exist_ok=True)
    workload = json.loads(a.workload.read_text())
    rows = workload["inputs"][:2 if a.phase == "smoke" else 16]
    expected = 2 if a.phase == "smoke" else 16
    if len(rows) != expected or len({r["problem_hash"] for r in rows}) != expected:
        raise RuntimeError("fixture must contain distinct registered development problems")
    if any(not r["input_ids"] for r in rows):
        raise RuntimeError("empty prompt")
    checkpoint_identity = json.loads(a.checkpoint_identity.read_text())
    if checkpoint_identity.get("target") != a.target_model or checkpoint_identity.get("sidecar") != a.sidecar:
        raise RuntimeError("export receipt target/sidecar differs from requested pair")
    image_identity = base.remote_script(a, shlex.join([*base.docker_prefix(a.no_sudo), "image", "inspect", "--format", "{{.Id}}", a.image]), timeout=30).stdout.strip()
    runtime = base.remote_python(a, f"""
import hashlib,json
from pathlib import Path
root=Path({a.overlay!r})
files={{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.json')}}
if not files: raise SystemExit('empty runtime overlay')
print(json.dumps({{'overlay_sha256':files,'hot_map_sha256':hashlib.sha256(Path({a.hot_map!r}).read_bytes()).hexdigest()}}))
""", timeout=30)
    if runtime.returncode:
        raise RuntimeError(runtime.stdout[-2000:])
    runtime_identity = json.loads(runtime.stdout)
    contract = {"protocol": "slim-m8-b1-greedy-natural-eos-countdown-v1",
                "target": a.target_model, "sidecar": a.sidecar,
                "overlay": a.overlay, "hot_map": a.hot_map,
                "image": a.image, "image_id": image_identity,
                "runtime_identity": runtime_identity,
                "checkpoint_identity": checkpoint_identity,
                "max_new_tokens": a.max_new_tokens,
                "workload": rows, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "helpers": {str(p.relative_to(a.repo_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (a.repo_root / ".opencode/nextlat_b1_window_select.py", a.repo_root / ".opencode/nextlat_b1_m8_deterministic_canary.py")}}
    a.contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    base.write_json(a.output_dir / "contract.json", contract)
    base.write_json(a.output_dir / "hardware.json", serving._gpu_preflight(a))
    base.write_json(a.output_dir / "image.json", {"image_id": image_identity, "requested": a.image})
    summary = {"protocol": contract["protocol"], "claim": "greedy only, warm natural-EOS development workload, not general deployment",
               "phase": a.phase, "pairs": [], "contract_sha256": a.contract_sha256}
    repeats = 1 if a.phase == "smoke" else a.repetitions
    for rep in range(repeats):
        arms = ("exact", "spec") if rep % 2 == 0 else ("spec", "exact")
        result = {arm: run_arm(a, rows, arm, rep) for arm in arms}
        pair = {"repetition": rep, "order": list(arms), **compare(result["exact"], result["spec"])}
        summary["pairs"].append(pair)
        if rep == 0:
            e, s = result["exact"]["hbm"], result["spec"]["hbm"]
            idle_delta = s["idle_mib"] - e["idle_mib"]
            summary["incremental_ready_mib"] = s["ready_mib"] - e["ready_mib"] - idle_delta
            summary["incremental_peak_mib"] = s["peak_mib"] - e["peak_mib"] - idle_delta
        summary["strict"] = all(p["strict"] for p in summary["pairs"])
        summary["pooled_speedup"] = sum(p["exact_seconds"] for p in summary["pairs"]) / sum(p["spec_seconds"] for p in summary["pairs"])
        summary["exact_tps"] = sum(p["exact_tokens"] for p in summary["pairs"]) / sum(p["exact_seconds"] for p in summary["pairs"])
        summary["spec_tps"] = sum(p["spec_tokens"] for p in summary["pairs"]) / sum(p["spec_seconds"] for p in summary["pairs"])
        summary["speedup_min_median_max"] = [min(p["speedup"] for p in summary["pairs"]), statistics.median(p["speedup"] for p in summary["pairs"]), max(p["speedup"] for p in summary["pairs"])]
        base.write_json(a.output_dir / "summary.json", summary)
        print(json.dumps(pair, sort_keys=True), flush=True)
        if not pair["strict"]:
            print("STRICT_PARITY_FAILED: retain diagnostics, do not promote speedup", flush=True)
            return 2
    print(f"COMPLETE strict={summary['strict']} speedup={summary['pooled_speedup']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
