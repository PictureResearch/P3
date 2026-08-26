from pathlib import Path
from typing import Any
import math
import os

import gymnasium as gym
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gymnasium import spaces
from pydantic import BaseModel, Field, model_validator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_path"
MODEL_PATH = MODEL_DIR / "ppo_swine_v2.zip"
NORMALIZER_PATH = MODEL_DIR / "swine_vec_normalize_v2.pkl"

OBS_DIM = 18
ACTION_NVEC = [2, 2, 2, 2, 2]
ACTION_NAMES = ["heater1", "heater2", "fan1", "fan2", "sprinkler"]


DEFAULT_CONFIG = {
    "herd": {
        "counts": {"piglets": 0, "young": 0, "adults": 8},
        "comfort_zones": {
            "piglets": [28, 34],
            "young": [18, 26],
            "adults": [15, 22],
            "source": "default",
        },
    },
    "eval": {
        "sensors": {"humidity": False, "co2": False, "clock": True},
        "equipment": {"heaters": 2, "fans": 2, "sprinklers": 1},
    },
}


class PredictRequest(BaseModel):
    observation: list[float] | None = Field(
        None,
        min_length=OBS_DIM,
        max_length=OBS_DIM,
        description="Required 18-number model observation.",
    )
    config: dict[str, Any] | None = None
    state: dict[str, Any] | None = None

    @model_validator(mode='before')
    def validate_request(cls, values):
        if values is None:
            raise ValueError("Request body must be provided.")
        if values.get("observation") is None and values.get("state") is None:
            raise ValueError("Either 'observation' or 'state' must be provided.")
        return values


class SwineEnvStub(gym.Env):
    """Tiny env used only so VecNormalize can load its saved statistics."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.full(OBS_DIM, -200, dtype=np.float32),
            high=np.full(OBS_DIM, 200, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(ACTION_NVEC)

    def reset(self, seed=None, options=None):
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}


app = FastAPI(title="P3 SwineRL Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
vec_normalizer = None
load_error = None


@app.on_event("startup")
def load_model():
    global model, vec_normalizer, load_error

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        load_error = f"Model file not found: {MODEL_PATH}"
        return

    if not NORMALIZER_PATH.exists():
        load_error = f"Normalizer file not found: {NORMALIZER_PATH}"
        return

    try:
        dummy_env = DummyVecEnv([lambda: SwineEnvStub()])
        vec_normalizer = VecNormalize.load(str(NORMALIZER_PATH), dummy_env)
        vec_normalizer.training = False
        vec_normalizer.norm_reward = False
        model = PPO.load(str(MODEL_PATH), device="cpu")
    except Exception as exc:
        load_error = f"Could not load model: {exc}"


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "pig_barn_simulator.html")

@app.get("/health")
def health():
    return {
        "name": "P3 SwineRL Model API",
        "model_loaded": model is not None and vec_normalizer is not None,
        "error": load_error,
    }


def deep_merge(base, override):
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_pair(values, default):
    if not isinstance(values, list):
        return default
    return [
        float(values[0]) if len(values) > 0 else default[0],
        float(values[1]) if len(values) > 1 else default[1],
    ]


def build_observation(config, state):
    cfg = deep_merge(DEFAULT_CONFIG, config or {})
    state = state or {}

    counts = cfg["herd"]["counts"]
    sensors = cfg.get("eval", {}).get("sensors", {})

    air_temp = float(state.get("air_temp", state.get("airTemp", 25.0)))
    perceived_temp = float(
        state.get("perceived_temp", state.get("perceivedTemp", air_temp))
    )
    humidity = float(state.get("humidity", 55.0))
    co2 = float(state.get("co2", 800.0))
    clock_hour = float(state.get("clock_hour", state.get("clockHour", 0.0)))

    heaters = get_pair(state.get("heaters", [0, 0]), [0.0, 0.0])
    fans = get_pair(state.get("fans", [0, 0]), [0.0, 0.0])
    sprinkler = float(state.get("sprinkler", 0.0))

    humidity_available = bool(sensors.get("humidity", True))
    co2_available = bool(sensors.get("co2", True))
    clock_available = bool(sensors.get("clock", True))

    if clock_available:
        clock_sin = math.sin(2 * math.pi * clock_hour / 24)
        clock_cos = math.cos(2 * math.pi * clock_hour / 24)
    else:
        clock_sin = 0.0
        clock_cos = 0.0

    return np.array(
        [
            air_temp,
            perceived_temp,
            air_temp - perceived_temp,
            heaters[0],
            heaters[1],
            fans[0],
            fans[1],
            sprinkler,
            float(counts.get("piglets", 0)),
            float(counts.get("young", 0)),
            float(counts.get("adults", 0)),
            humidity / 100.0 if humidity_available else 0.0,
            1.0 if humidity_available else 0.0,
            co2 / 5000.0 if co2_available else 0.0,
            1.0 if co2_available else 0.0,
            clock_sin,
            clock_cos,
            1.0 if clock_available else 0.0,
        ],
        dtype=np.float32,
    )


def mask_action_for_equipment(action_list, config):
    cfg = deep_merge(DEFAULT_CONFIG, config or {})
    equipment = cfg.get("eval", {}).get("equipment", {})

    if int(equipment.get("heaters", 2)) < 2:
        action_list[1] = 0
    if int(equipment.get("heaters", 2)) < 1:
        action_list[0] = 0
    if int(equipment.get("fans", 2)) < 2:
        action_list[3] = 0
    if int(equipment.get("fans", 2)) < 1:
        action_list[2] = 0
    if int(equipment.get("sprinklers", 1)) < 1:
        action_list[4] = 0

    return action_list


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or vec_normalizer is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "Model is not loaded.",
        )

    if req.observation is None:
        observation = build_observation(req.config, req.state)
        observation = observation.reshape(1, -1)
    else:
        observation = np.array(req.observation, dtype=np.float32).reshape(1, -1)

    try:
        normalized_observation = vec_normalizer.normalize_obs(observation)
        action, _ = model.predict(normalized_observation, deterministic=True)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    action_list = [int(value) for value in action[0].tolist()]
    action_list = mask_action_for_equipment(action_list, None)
    return {
        "action": action_list,
        "actions": dict(zip(ACTION_NAMES, action_list)),
        "observation": observation[0].tolist(),
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run("main:app", host=host, port=port, reload=reload)
