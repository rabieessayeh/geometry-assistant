"""Shared fixtures: small meshes with known dimensions, offline settings and a scripted fake LLM."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import trimesh
from trimesh.creation import box

from app.config import Settings
from app.geometry import Catalog
from scripts import make_sample_models as samples


def wedge() -> trimesh.Trimesh:
    """Prism whose only down-facing face is a 45 degree slope (10 x 5 x 10 mm).

    Its x-z profile is the triangle (0, 0), (10, 10), (0, 10): it touches the
    build plate along one edge and leans over it at 45 degrees.
    """
    vertices = [[0, 0, 0], [10, 0, 10], [0, 0, 10], [0, 5, 0], [10, 5, 10], [0, 5, 10]]
    faces = [
        [0, 1, 2],  # front (-y)
        [3, 5, 4],  # back (+y)
        [0, 2, 5],  # wall (-x)
        [0, 5, 3],
        [2, 1, 4],  # top (+z)
        [2, 4, 5],
        [0, 3, 4],  # slope (+x, -z)
        [0, 4, 1],
    ]
    return trimesh.Trimesh(vertices=np.array(vertices, dtype=float), faces=faces)


def open_box() -> trimesh.Trimesh:
    """A 10 x 20 x 30 mm box without its top: not watertight."""
    mesh = box(bounds=[[0, 0, 0], [10, 20, 30]])
    keep = mesh.face_normals[:, 2] < 0.5
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[keep])


@pytest.fixture(scope="session")
def meshes() -> dict[str, trimesh.Trimesh]:
    """Test meshes, built once: a box, a wedge, an open box and the three sample parts."""
    return {
        "box": box(bounds=[[0, 0, 0], [10, 20, 30]]),
        "wedge": wedge(),
        "open_box": open_box(),
        **{name: build() for name, build in samples.SAMPLES.items()},
    }


@pytest.fixture
def cat(meshes) -> Catalog:
    return Catalog(dict(meshes))


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings that ignore the developer's `.env` and never wait between retries."""
    return Settings(
        _env_file=None,
        llm_base_url="http://llm.invalid/v1",
        llm_api_key="test-key",
        llm_model="fake-model",
        llm_retry_base_s=0,
        samples_dir=tmp_path,
    )


class FakeMessage(NS):
    """Mimics the assistant message object of the OpenAI SDK."""

    def model_dump(self, exclude_none: bool = True) -> dict:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in (self.tool_calls or [])
            ],
        }


def tool_call(name: str, args: dict | str, call_id: str = "c1") -> NS:
    """Build a fake tool call; `args` may be a raw string to simulate malformed JSON."""
    arguments = args if isinstance(args, str) else json.dumps(args)
    return NS(id=call_id, function=NS(name=name, arguments=arguments))


class FakeClient:
    """Scripted LLM: plays `calls` on the first turn, then answers with the tool summary."""

    def __init__(
        self,
        calls: list[NS] | None = None,
        failures: list[Exception] | None = None,
        repeat: bool = False,
    ):
        self.calls = calls or []
        self.failures = failures or []  # exceptions raised before the first success
        self.repeat = repeat  # keep requesting `calls` on every turn
        self.requests: list[list[dict]] = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model: str, messages: list[dict], tools: list[dict]) -> NS:
        if self.failures:
            raise self.failures.pop(0)
        self.requests.append(list(messages))
        if self.calls and (self.repeat or len(self.requests) == 1):
            msg = FakeMessage(content=None, tool_calls=self.calls)
        else:
            last = messages[-1]
            result = json.loads(last["content"]) if last["role"] == "tool" else {}
            text = result.get("summary") or result.get("error") or "No tool was needed."
            msg = FakeMessage(content=text, tool_calls=None)
        return NS(choices=[NS(message=msg)])


@pytest.fixture
def fake_llm(monkeypatch):
    """Return a function installing a `FakeClient` in place of the real LLM client."""

    def install(
        calls: list[NS] | None = None,
        failures: list[Exception] | None = None,
        repeat: bool = False,
    ) -> FakeClient:
        client = FakeClient(calls, failures, repeat)
        monkeypatch.setattr("app.agent.build_client", lambda settings: client)
        return client

    return install
