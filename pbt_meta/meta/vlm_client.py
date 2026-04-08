"""
OllamaVLMClient -- Ollama/Qwen2.5-VL interface for M3 VLM Dispatch.
Wraps Ollama HTTP API to send (image + text) -> get text response.
"""
import base64
import io
import json
import time
import logging
from dataclasses import dataclass
import numpy as np

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    text: str
    latency_ms: float
    model: str
    success: bool
    error: str = ""


class OllamaVLMClient:
    """Client for Ollama vision-language model API."""

    def __init__(self, model="qwen2.5-vl:3b", base_url="http://localhost:11434",
                 timeout_s=30.0, temperature=0.1, max_tokens=200):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_queries = 0
        self.total_latency_ms = 0.0
        self.last_latency_ms = 0.0

    def query(self, image_rgb, system_prompt, user_prompt):
        if requests is None:
            return VLMResponse(text="", latency_ms=0, model=self.model,
                               success=False, error="requests not installed")
        b64_image = self._encode_image(image_rgb)
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "images": [b64_image],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{self.base_url}/api/generate",
                                 json=payload, timeout=self.timeout_s)
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                return VLMResponse(text="", latency_ms=latency_ms, model=self.model,
                                   success=False, error=f"HTTP {resp.status_code}")
            text = resp.json().get("response", "")
            self.total_queries += 1
            self.total_latency_ms += latency_ms
            self.last_latency_ms = latency_ms
            return VLMResponse(text=text, latency_ms=latency_ms,
                               model=self.model, success=True)
        except requests.exceptions.Timeout:
            latency_ms = (time.perf_counter() - t0) * 1000
            return VLMResponse(text="", latency_ms=latency_ms, model=self.model,
                               success=False, error=f"Timeout after {self.timeout_s}s")
        except requests.exceptions.ConnectionError as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return VLMResponse(text="", latency_ms=latency_ms, model=self.model,
                               success=False, error=f"Connection failed: {e}")
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return VLMResponse(text="", latency_ms=latency_ms, model=self.model,
                               success=False, error=f"Error: {e}")

    def _encode_image(self, image_rgb):
        if Image is not None:
            img = Image.fromarray(image_rgb.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        raise ImportError("PIL required for image encoding")

    def health_check(self):
        if requests is None:
            return False
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            return False

    @property
    def avg_latency_ms(self):
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries
