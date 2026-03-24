from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import random, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⬅️ izinkan semua asal (sementara)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
FS = 50
WIN_SEC = 2.0
OVERLAP = 0.5
STEP_SEC = WIN_SEC * (1 - OVERLAP)
DASHBOARD_MINUTES = 10
WELDER_COUNT = 15

DETAILED_LABELS = [
    "Lying active",
    "Lying inactive",
    "Standing active",
    "Standing inactive",
    "Sitting active",
    "Sitting inactive",
    "Unsafe condition",
]

AGGREGATED = {
    "Active": {"Lying active", "Standing active", "Sitting active"},
    "Inactive": {"Lying inactive", "Standing inactive", "Sitting inactive"},
    "Unsafe": {"Unsafe condition"},
}

TRANSITIONS = {
    "Lying active": [
        ("Lying active", 8),
        ("Sitting active", 2),
        ("Lying inactive", 4),
        ("Unsafe condition", 0.4),
    ],
    "Lying inactive": [
        ("Lying inactive", 10),
        ("Lying active", 4),
        ("Sitting inactive", 2),
        ("Unsafe condition", 0.2),
    ],
    "Standing active": [
        ("Standing active", 10),
        ("Sitting active", 2),
        ("Standing inactive", 3),
        ("Unsafe condition", 0.6),
    ],
    "Standing inactive": [
        ("Standing inactive", 9),
        ("Standing active", 3),
        ("Sitting inactive", 2),
        ("Unsafe condition", 0.2),
    ],
    "Sitting active": [
        ("Sitting active", 9),
        ("Standing active", 3),
        ("Sitting inactive", 3),
        ("Unsafe condition", 0.5),
    ],
    "Sitting inactive": [
        ("Sitting inactive", 10),
        ("Sitting active", 3),
        ("Lying inactive", 2),
        ("Unsafe condition", 0.2),
    ],
    "Unsafe condition": [
        ("Standing active", 3),
        ("Sitting active", 3),
        ("Standing inactive", 1),
        ("Sitting inactive", 1),
    ],
}


def choice_weighted(options):
    total = sum(w for _, w in options)
    r = random.random() * total
    for val, w in options:
        if (r := r - w) <= 0:
            return val
    return options[-1][0]


def synthesize_timeline(minutes: int, welder_idx: int, start_ts: float):
    step_ms = STEP_SEC * 1000
    windows = int((minutes * 60) / STEP_SEC)
    label = random.choice(DETAILED_LABELS)
    series = []

    for i in range(windows):
        ts = start_ts + i * step_ms
        opts = TRANSITIONS[label]
        label = choice_weighted(opts)

        if label in AGGREGATED["Active"]:
            motion_var = random.uniform(1.2, 2.8)
        elif label in AGGREGATED["Inactive"]:
            motion_var = random.uniform(0.2, 0.8)
        else:
            motion_var = random.uniform(2.0, 4.0)

        series.append(
            {
                "ts": int(ts),
                "label": label,
                "motionVar": round(motion_var, 2),
                "samples": int(FS * WIN_SEC),
            }
        )
    return series


@app.get("/api/dashboard")
def get_dashboard(seed: int = 573):
    random.seed(seed)
    start_ts = datetime.now().timestamp() * 1000 - DASHBOARD_MINUTES * 60 * 1000
    welders = [{"id": f"W-{i+1}", "name": f"Welder {i+1}"} for i in range(WELDER_COUNT)]
    management_data = {w["id"]: synthesize_timeline(DASHBOARD_MINUTES, i, start_ts) for i, w in enumerate(welders)}

    return {
        "mode": "dummy",
        "welders": welders,
        "managementData": management_data,
        "FS": FS,
        "WIN_SEC": WIN_SEC,
        "OVERLAP": OVERLAP,
        "STEP_SEC": STEP_SEC,
        "DASHBOARD_MINUTES": DASHBOARD_MINUTES,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard_server:app", host="0.0.0.0", port=8082, reload=True)
