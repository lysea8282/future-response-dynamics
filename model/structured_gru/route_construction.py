"""Frozen model, interface loading, and F/CF/P route construction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

ANCHOR = 7
HORIZON = 12
CARRIER_DIM = 192
NORMALIZER_STD = np.asarray([
    0.6748629808425903, 0.6790841221809387, 0.2936058044433594, 0.2959529161453247,
    0.6748629808425903, 0.6790841221809387, 0.2936058044433594, 0.2959529161453247,
], dtype=np.float64)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192), nn.GELU(), nn.Linear(192, 8))

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        return self.network(carrier)


class DeterministicBeliefModel(nn.Module):
    cap = 3.0
    carrier_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.observation_encoder = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 64))
        self.action_encoder = nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, 64))
        self.message_network = nn.Sequential(nn.Linear(192, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 64))
        self.deterministic_update = nn.GRUCell(144, 48)
        self.global_update = nn.GRUCell(192, 64)
        self.decoder = Decoder()
        self.prior_network = nn.Sequential(nn.Linear(176, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 16))
        self.posterior_network = nn.Sequential(nn.Linear(176, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 16))

    @staticmethod
    def unpack(z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return z[..., :48], z[..., 48:64], z[..., 64:112], z[..., 112:128], z[..., 128:]

    @staticmethod
    def pack(r0: torch.Tensor, m0: torch.Tensor, r1: torch.Tensor, m1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return torch.cat((r0, m0, r1, m1, g), dim=-1)

    def step(self, z: torch.Tensor, action: torch.Tensor, observation: torch.Tensor | None = None) -> torch.Tensor:
        r0, m0, r1, m1, g = self.unpack(z)
        q0 = self.message_network(torch.cat((r0, m0, r1, m1, g), dim=-1))
        q1 = self.message_network(torch.cat((r1, m1, r0, m0, g), dim=-1))
        magnitude = torch.linalg.norm(action[..., 2:], dim=-1, keepdim=True)[..., None, :].expand(*action.shape[:-1], 2, 1)
        per_object = torch.cat((action[..., :2, None], action[..., None, 2:].expand(*action.shape[:-1], 2, 2), magnitude), dim=-1)
        ae = self.action_encoder(per_object)
        n0 = self.deterministic_update(torch.cat((m0, q0, ae[..., 0, :]), dim=-1), r0)
        n1 = self.deterministic_update(torch.cat((m1, q1, ae[..., 1, :]), dim=-1), r1)
        ng = self.global_update(torch.cat((n0, m0, n1, m1, self.action_encoder(action)), dim=-1), g)
        p0 = self.cap * torch.tanh(self.prior_network(torch.cat((n0, ng, ae[..., 0, :]), dim=-1)) / self.cap)
        p1 = self.cap * torch.tanh(self.prior_network(torch.cat((n1, ng, ae[..., 1, :]), dim=-1)) / self.cap)
        prior = torch.stack((p0, p1), dim=-2)
        if observation is None:
            middle = prior
        else:
            encoded = self.observation_encoder(observation)
            q0p = self.cap * torch.tanh(self.posterior_network(torch.cat((n0, ng, encoded[..., 0, :]), dim=-1)) / self.cap)
            q1p = self.cap * torch.tanh(self.posterior_network(torch.cat((n1, ng, encoded[..., 1, :]), dim=-1)) / self.cap)
            posterior = torch.stack((q0p, q1p), dim=-2)
            visibility = observation[..., 4:5]
            middle = visibility * posterior + (1.0 - visibility) * prior
        return self.pack(n0, middle[..., 0, :], n1, middle[..., 1, :], ng)

    def transition(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.step(z, action, None)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def load_checkpoint(binding: dict[str, Any], device: torch.device) -> DeterministicBeliefModel:
    path = Path(binding["absolute_path"])
    if path.stat().st_size != int(binding["size_bytes"]) or sha256(path) != binding["sha256"]:
        raise RuntimeError(f"CHECKPOINT_BINDING_FAILURE:{binding['seed']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DeterministicBeliefModel()
    state = payload.get("state_dict")
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise RuntimeError("CHECKPOINT_TOPOLOGY_FAILURE")
    if sum(p.numel() for p in model.parameters()) != 496200:
        raise RuntimeError("PARAMETER_COUNT_FAILURE")
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


@torch.no_grad()
def assimilate(model: DeterministicBeliefModel, observations: np.ndarray, actions: np.ndarray, device: torch.device) -> torch.Tensor:
    obs = torch.as_tensor(observations[:, : ANCHOR + 1], dtype=torch.float32, device=device)
    act = torch.as_tensor(actions[:, : ANCHOR + 1], dtype=torch.float32, device=device)
    z = torch.zeros((len(obs), CARRIER_DIM), dtype=torch.float32, device=device)
    for t in range(ANCHOR + 1):
        z = model.step(z, act[:, t], obs[:, t])
    return z


def primitive_features(records: Iterable[dict[str, Any]]) -> np.ndarray:
    rows = list(records)
    x = np.zeros((len(rows), 13), dtype=np.float64)
    x[:, 0] = 1.0
    for i, row in enumerate(rows):
        j = 1 + 2 * int(row["edit_object_id"]) + int(row["edit_axis"])
        x[i, j] = float(row["counterfactual_values"][0]) - float(row["factual_values"][0])
        x[i, 5:] = np.asarray(row["factual_state_t"], dtype=np.float64).reshape(8)
    return x


def load_interface(seed_dir: Path, manifest_entries: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    names = {"U4": "U4.npy", "feature_mean": "feature_mean.npy", "feature_scale": "feature_scale.npy", "affine_weights": "affine_weights.npy"}
    for key, name in names.items():
        path = seed_dir / name
        binding = manifest_entries[name]
        if sha256(path) != binding["sha256"] or path.stat().st_size != binding["size_bytes"]:
            raise RuntimeError(f"INTERFACE_BINDING_FAILURE:{seed_dir.name}:{name}")
        arrays[key] = np.load(path, allow_pickle=False)
    if arrays["U4"].shape != (192, 4) or arrays["affine_weights"].shape != (13, 4):
        raise RuntimeError("INTERFACE_SHAPE_FAILURE")
    return arrays


def patch_delta(records: list[dict[str, Any]], interface: dict[str, np.ndarray]) -> np.ndarray:
    x = primitive_features(records)
    x[:, 1:] = (x[:, 1:] - interface["feature_mean"][1:]) / interface["feature_scale"][1:]
    x[:, 0] = 1.0
    coefficients = x @ interface["affine_weights"]  # full affine; row 0 is not discarded
    delta = coefficients @ interface["U4"].T
    if delta.shape != (len(records), 192) or not np.isfinite(delta).all():
        raise RuntimeError("PATCH_ASSEMBLY_FAILURE")
    return delta.astype(np.float32)


def construct_routes(model: DeterministicBeliefModel, records: list[dict[str, Any]], interface: dict[str, np.ndarray], device: torch.device) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    actions = np.asarray([r["actions"] for r in records], dtype=np.float32)
    z_f = assimilate(model, np.asarray([r["factual_observations"] for r in records], dtype=np.float32), actions, device)
    z_cf = assimilate(model, np.asarray([r["counterfactual_observations"] for r in records], dtype=np.float32), actions, device)
    delta = patch_delta(records, interface)
    z_p = z_f + torch.as_tensor(delta, dtype=torch.float32, device=device)
    return {"F": z_f, "CF": z_cf, "P": z_p}, delta



