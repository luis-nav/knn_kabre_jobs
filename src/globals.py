MUTABLE_FEATURES = [
    "ReqCPUS",
    "ReqMem_MB",
    "ReqNodes",
    "TimelimitRaw",
    "submit_hour",
    "submit_dow",
    # partitions and QOS are also mutable (user can change queue)
]

PARTITIONS = [
    "andalan", "andalan-debug", "andalan-long",
    "dribe", "dribe-debug", "dribe-long", "dribe-test",
    "kura", "kura-all", "kura-debug", "kura-long", "kura-test", "kura-wide",
    "nu", "nu-all", "nu-debug", "nu-long", "nu-wide",
    "nukwa", "nukwa-debug", "nukwa-l40s", "nukwa-long", "nukwa-v100", "nukwa-wide",
]

QOS_VALUES = ["normal", "high-priority"]
