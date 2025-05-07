import modal, os, json, datetime

#RUN BEFORE REDEPLOY

trace_buffer = modal.Dict.from_name("trace-buffer")
trace_volume = modal.Volume.from_name("ssh-traces", create_if_missing=True)
TRACE_PATH = "/traces"

app = modal.App("DarkCircuit-Flusher")

@app.function(volumes={TRACE_PATH: trace_volume})
def flush_to_volume():
    import os, datetime, json
    os.makedirs(TRACE_PATH, exist_ok=True)
    keys = list(trace_buffer.keys())
    for k in keys:
        try:
            rec = trace_buffer.pop(k)
        except KeyError:
            continue
        if isinstance(rec["timestamp"], (int, float)):
            rec["timestamp"] = datetime.datetime.utcfromtimestamp(rec["timestamp"]).isoformat()
        path = f"{TRACE_PATH}/{rec['trace_id']}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


#TO RUN : modal run flush_traces.py::flush_to_volume
