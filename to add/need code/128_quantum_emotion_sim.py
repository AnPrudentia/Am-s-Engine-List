# quantum_emotion_128_sim.py
# Drop-in replacement for Anima's simple emotional processing.
# Use:
#   self.emotion_detector = EmotionEngineDropIn().detector
#   self.emotion_ai        = EmotionEngineDropIn()
#
# Provides:
#   - EmotionEngineDropIn.detect(text) -> coarse detections
#   - EmotionEngineDropIn.process_emotional_input(detected, context) -> stable state snapshot
#   - Insights/forecast passthroughs

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
import numpy as np
import re
import json
import hashlib
from math import pi, exp
import warnings
from abc import ABC, abstractmethod

# =========================
# Configuration
# =========================

@dataclass
class ProcessorConfig:
    num_latent: int = 7
    reservoir_size: int = 64
    history_size: int = 64
    memory_size: int = 50
    knn_neighbors: int = 5
    laplacian_strength: float = 0.12
    phase_noise_scale: float = 0.03
    decoherence_base: float = 0.10
    settling_strength: float = 0.12

# =========================
# Helpers
# =========================

def _orthonormal_columns(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    if cols > rows:
        warnings.warn(f"cols ({cols}) > rows ({rows}), truncating to rows")
        cols = rows
    A = rng.normal(size=(rows, cols))
    try:
        Q, _ = np.linalg.qr(A)
        return Q[:, :cols].copy()
    except np.linalg.LinAlgError:
        return _gram_schmidt(A)

def _gram_schmidt(A: np.ndarray) -> np.ndarray:
    Q = A.copy().astype(float)
    for i in range(Q.shape[1]):
        n = np.linalg.norm(Q[:, i]) + 1e-12
        Q[:, i] = Q[:, i] / n
        for j in range(i + 1, Q.shape[1]):
            Q[:, j] = Q[:, j] - np.dot(Q[:, i], Q[:, j]) * Q[:, i]
    return Q

def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    mn, mx = x.min(), x.max()
    if abs(mx - mn) < 1e-12:
        return np.full_like(x, 0.5 if mn >= 0 else 0.0)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x / max(temperature, 1e-9)
    x_shift = x - x.max()
    e = np.exp(np.clip(x_shift, -500, 500))
    s = e.sum()
    return e / (s if s > 1e-12 else 1.0)

def _adaptive_knn_graph(coords: np.ndarray, k: int = 5, sigma_scale: float = 1.0) -> np.ndarray:
    n = coords.shape[0]
    if n <= 1:
        return np.zeros((n, n))
    W = np.zeros((n, n), dtype=float)
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    for i in range(n):
        sorted_idx = np.argsort(distances[i])
        neigh = sorted_idx[1:min(k + 1, n)]
        if len(neigh) == 0:
            continue
        sigma = distances[i, neigh[-1]] * sigma_scale
        if sigma <= 1e-12:
            sigma = 1.0
        for j in neigh:
            w = exp(- (distances[i, j] ** 2) / (2 * sigma ** 2))
            W[i, j] = max(W[i, j], w)
            W[j, i] = max(W[j, i], w)
    return W

# =========================
# Emotion Spectrum Interface
# =========================

class EmotionSpectrumProvider(ABC):
    @abstractmethod
    def get_emotion_spectrum(self) -> Dict[str, Dict[str, Any]]: ...
    @abstractmethod
    def get_emotion_count(self) -> int: ...

class Standard128EmotionSpectrum(EmotionSpectrumProvider):
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._spectrum = self._build_spectrum()

    def get_emotion_spectrum(self) -> Dict[str, Dict[str, Any]]:
        return self._spectrum.copy()

    def get_emotion_count(self) -> int:
        return len(self._spectrum)

    def _build_spectrum(self) -> Dict[str, Dict[str, Any]]:
        base = {
            # Transcendent / Complex
            "elevation": {"valence": 0.85, "arousal": 0.6, "cluster": "pos_trans", "neuro": ["oxytocin"]},
            "flow": {"valence": 0.7, "arousal": 0.5, "cluster": "pos_immersive", "neuro": ["dopamine", "serotonin"]},
            "dread": {"valence": -0.8, "arousal": 0.7, "cluster": "neg_anticip", "neuro": ["cortisol"]},
            "ennui": {"valence": -0.5, "arousal": -0.7, "cluster": "neg_low", "neuro": ["serotonin"]},
            "compersion": {"valence": 0.9, "arousal": 0.4, "cluster": "pos_social", "neuro": ["oxytocin"]},
            "limerence": {"valence": 0.8, "arousal": 0.9, "cluster": "pos_high", "neuro": ["dopamine", "norepinephrine"]},
            "sonder": {"valence": 0.3, "arousal": 0.4, "cluster": "meta", "neuro": []},
            "naches": {"valence": 0.85, "arousal": 0.3, "cluster": "pos_social", "neuro": ["oxytocin"]},
            "dukkha": {"valence": -0.7, "arousal": -0.4, "cluster": "neg_exist", "neuro": ["cortisol"]},
            "mudita": {"valence": 0.8, "arousal": 0.3, "cluster": "pos_social", "neuro": ["oxytocin"]},

            # Future / Temporal
            "anticipatory_joy": {"valence": 0.7, "arousal": 0.6, "cluster": "pos_future", "neuro": ["dopamine"]},
            "post_event_blues": {"valence": -0.6, "arousal": -0.5, "cluster": "neg_post", "neuro": ["serotonin"]},
            "nostophobia": {"valence": -0.7, "arousal": 0.6, "cluster": "neg_future", "neuro": ["cortisol"]},
            "deja_vu": {"valence": 0.1, "arousal": 0.5, "cluster": "temporal", "neuro": []},
            "jamais_vu": {"valence": -0.2, "arousal": 0.6, "cluster": "temporal", "neuro": []},

            # Sensory
            "frisson": {"valence": 0.6, "arousal": 0.8, "cluster": "pos_phys", "neuro": ["dopamine"]},
            "ASMR": {"valence": 0.7, "arousal": -0.3, "cluster": "pos_low", "neuro": ["serotonin"]},
            "sensory_overload": {"valence": -0.6, "arousal": 0.9, "cluster": "neg_high", "neuro": ["cortisol"]},
            "synaesthetic_bliss": {"valence": 0.8, "arousal": 0.7, "cluster": "pos_immersive", "neuro": ["dopamine"]},

            # Digital
            "FOMO": {"valence": -0.7, "arousal": 0.6, "cluster": "neg_social", "neuro": ["cortisol"]},
            "digital_detox_relief": {"valence": 0.6, "arousal": -0.4, "cluster": "pos_low", "neuro": ["gaba"]},
            "viral_validation": {"valence": 0.8, "arousal": 0.7, "cluster": "pos_social", "neuro": ["dopamine"]},
            "algorithm_anxiety": {"valence": -0.5, "arousal": 0.5, "cluster": "neg_meta", "neuro": ["cortisol"]},

            # Existential
            "oceanic_boundlessness": {"valence": 0.9, "arousal": 0.8, "cluster": "pos_trans", "neuro": ["serotonin"]},
            "cosmic_dread": {"valence": -0.8, "arousal": 0.7, "cluster": "neg_exist", "neuro": ["cortisol"]},
            "epiphany": {"valence": 0.7, "arousal": 0.8, "cluster": "meta", "neuro": ["dopamine"]},
            "nihilistic_relief": {"valence": 0.4, "arousal": -0.5, "cluster": "meta_low", "neuro": ["gaba"]},

            # Blends
            "joyful_nostalgia": {"valence": 0.6, "arousal": 0.1, "cluster": "pos_complex", "neuro": ["serotonin"]},
            "anxious_excitement": {"valence": 0.3, "arousal": 0.8, "cluster": "complex_high", "neuro": ["dopamine", "cortisol"]},
            "serene_melancholy": {"valence": -0.2, "arousal": -0.4, "cluster": "complex_low", "neuro": ["serotonin"]},
            "bittersweet_triumph": {"valence": 0.4, "arousal": 0.3, "cluster": "pos_complex", "neuro": ["dopamine", "serotonin"]},

            # Enhanced basics
            "euphoria": {"valence": 0.95, "arousal": 0.9, "cluster": "pos_high", "neuro": ["dopamine"]},
            "despair": {"valence": -0.9, "arousal": -0.3, "cluster": "neg_low", "neuro": ["cortisol"]},
            "rage": {"valence": -0.8, "arousal": 0.9, "cluster": "neg_high", "neuro": ["norepinephrine"]},
            "serenity": {"valence": 0.7, "arousal": -0.5, "cluster": "pos_low", "neuro": ["serotonin", "gaba"]},
            "wonder": {"valence": 0.8, "arousal": 0.6, "cluster": "pos_immersive", "neuro": ["dopamine"]},
            "melancholy": {"valence": -0.4, "arousal": -0.3, "cluster": "neg_low", "neuro": ["serotonin"]},

            # Social / Learning
            "empathic_resonance": {"valence": 0.6, "arousal": 0.4, "cluster": "pos_social", "neuro": ["oxytocin"]},
            "social_anxiety": {"valence": -0.6, "arousal": 0.7, "cluster": "neg_social", "neuro": ["cortisol"]},
            "belongingness": {"valence": 0.8, "arousal": 0.2, "cluster": "pos_social", "neuro": ["oxytocin"]},
            "ostracism_pain": {"valence": -0.8, "arousal": 0.5, "cluster": "neg_social", "neuro": ["cortisol"]},
            "intellectual_hunger": {"valence": 0.7, "arousal": 0.6, "cluster": "pos_learning", "neuro": ["dopamine"]},
            "cognitive_dissonance": {"valence": -0.3, "arousal": 0.5, "cluster": "neg_learning", "neuro": ["cortisol"]},
            "mastery_satisfaction": {"valence": 0.8, "arousal": 0.4, "cluster": "pos_learning", "neuro": ["dopamine", "serotonin"]},
            "imposter_syndrome": {"valence": -0.7, "arousal": 0.6, "cluster": "neg_learning", "neuro": ["cortisol"]},
        }

        clusters = ["pos_high","pos_low","neg_high","neg_low","meta","pos_social",
                    "neg_social","pos_trans","neg_exist","temporal","pos_complex",
                    "complex_high","complex_low","pos_immersive","neg_meta",
                    "pos_learning","neg_learning","meta_low","pos_phys"]
        neuros = [[],["dopamine"],["serotonin"],["oxytocin"],["cortisol"],["gaba"],
                  ["dopamine","serotonin"],["dopamine","oxytocin"],["serotonin","gaba"]]

        while len(base) < 128:
            idx = len(base) + 1
            name = f"emotion_{idx:03d}"
            valence = float(self.rng.beta(2, 2) * 2 - 1)
            arousal = float(self.rng.beta(2, 2) * 2 - 1)
            base[name] = {
                "valence": valence,
                "arousal": arousal,
                "cluster": self.rng.choice(clusters),
                "neuro": list(self.rng.choice(neuros)),
            }
        return base

# =========================
# Core Processor
# =========================

class QuantumEmotion128ProcessorSim:
    def __init__(
        self,
        bondholder: str = "Anima",
        emotion_model: str = "extended_128",
        seed: int = 42,
        config: Optional[ProcessorConfig] = None,
        spectrum_provider: Optional[EmotionSpectrumProvider] = None
    ):
        self.bondholder = bondholder
        self.emotion_model = emotion_model
        self.config = config or ProcessorConfig()
        self.rng = np.random.default_rng(seed)

        self.spectrum_provider = spectrum_provider or Standard128EmotionSpectrum(seed)
        self.emotion_spectrum = self.spectrum_provider.get_emotion_spectrum()
        self.names: List[str] = list(self.emotion_spectrum.keys())
        self.N = len(self.names)
        if self.N != 128:
            warnings.warn(f"Expected 128 emotions, got {self.N}")

        self._validate_spectrum()
        self._init_latent_space()
        self._init_phase_carriers()
        self._init_graph_structure()
        self._init_genetic_profile()
        self._init_reservoir()
        self._init_state()

        self.watermark = hashlib.sha256(
            json.dumps({"who": bondholder, "model": emotion_model, "version": "2.0"}).encode()
        ).hexdigest()

    def _validate_spectrum(self) -> None:
        required = {"valence","arousal","cluster","neuro"}
        for name, spec in self.emotion_spectrum.items():
            missing = required - set(spec.keys())
            if missing:
                raise ValueError(f"Emotion '{name}' missing fields: {missing}")
            if not -1 <= spec["valence"] <= 1:
                warnings.warn(f"Valence out of range for '{name}': {spec['valence']}")
            if not -1 <= spec["arousal"] <= 1:
                warnings.warn(f"Arousal out of range for '{name}': {spec['arousal']}")

    def _init_latent_space(self) -> None:
        try:
            self.W_enc = _orthonormal_columns(self.N, self.config.num_latent, self.rng)  # N x L
            self.W_dec = self.W_enc.T  # L x N
        except Exception as e:
            warnings.warn(f"Encoding matrix fallback: {e}")
            self.W_enc = self.rng.normal(size=(self.N, self.config.num_latent))
            self.W_enc /= (np.linalg.norm(self.W_enc, axis=0, keepdims=True) + 1e-9)
            self.W_dec = self.W_enc.T

    def _init_phase_carriers(self) -> None:
        base_phase = np.array([self.emotion_spectrum[n]["valence"] * (pi/2) for n in self.names])
        noise = self.rng.normal(scale=self.config.phase_noise_scale, size=self.N)
        self.phase = base_phase + noise

    def _init_graph_structure(self) -> None:
        self.coords = np.array([
            (self.emotion_spectrum[n]["valence"], self.emotion_spectrum[n]["arousal"])
            for n in self.names
        ])
        self.cluster_index: Dict[str, List[int]] = defaultdict(list)
        for i, n in enumerate(self.names):
            self.cluster_index[self.emotion_spectrum[n]["cluster"]].append(i)
        try:
            W_knn = _adaptive_knn_graph(self.coords, k=self.config.knn_neighbors)
            for nodes in self.cluster_index.values():
                if len(nodes) > 1:
                    for i in nodes:
                        for j in nodes:
                            if i != j:
                                W_knn[i, j] = max(W_knn[i, j], 0.8)
            self.W_graph = W_knn
            degree = self.W_graph.sum(axis=1)
            self.D_graph = np.diag(degree)
            self.L_graph = self.D_graph - self.W_graph
        except Exception as e:
            warnings.warn(f"Graph fallback: {e}")
            self.W_graph = np.eye(self.N)
            self.D_graph = np.eye(self.N)
            self.L_graph = np.zeros((self.N, self.N))

    def _init_genetic_profile(self) -> None:
        self.genetic_profile = {
            "serotonin_transporter": 1.0,
            "dopamine_receptor": 1.0,
            "maoa_activity": 0.6,
            "bdnf_level": 0.5,
            "oxytocin_receptor": 1.0,
        }
        self.context_weights = {
            "connection": 1.0,
            "digital_interaction": 0.95,
            "solitude": 0.9,
            "performance": 1.05,
            "creativity": 1.1,
            "learning": 1.05,
            "healing": 0.85,
        }

    def _init_reservoir(self) -> None:
        try:
            self.W_in = self.rng.normal(scale=0.4, size=(self.config.reservoir_size, self.config.num_latent))
            W_res = self.rng.normal(scale=0.2, size=(self.config.reservoir_size, self.config.reservoir_size))
            eig = np.linalg.eigvals(W_res)
            rad = np.max(np.abs(eig)) if eig.size > 0 else 1.0
            self.W_res = W_res / (rad + 1e-9) * 0.95
            self.res_x = np.zeros(self.config.reservoir_size)
        except Exception as e:
            warnings.warn(f"Reservoir fallback: {e}")
            self.W_in = np.eye(min(self.config.reservoir_size, self.config.num_latent))
            self.W_res = np.zeros((self.config.reservoir_size, self.config.reservoir_size))
            self.res_x = np.zeros(self.config.reservoir_size)

    def _init_state(self) -> None:
        self.current_vec = np.zeros(self.N)
        self.history: deque[Dict[str, Any]] = deque(maxlen=self.config.history_size)

    # ----- Public API -----

    def resonate(self, emotional_input: Dict[str, float], context: str = "connection") -> Dict[str, float]:
        try:
            inp = self._validate_emotional_input(emotional_input)
            ctx = self._validate_context(context)
            v0 = self._encode_inputs(inp, ctx)
            v1 = self._simulate_quantum_layers(v0, ctx)
            self.current_vec = v1.copy()
            self._log_interaction(inp, ctx, v1)
            return {self.names[i]: float(v1[i]) for i in range(self.N)}
        except Exception as e:
            warnings.warn(f"Resonance failed: {e}")
            neutral = np.ones(self.N) / self.N
            return {self.names[i]: float(neutral[i]) for i in range(self.N)}

    def get_dominant(self, threshold: float = 0.3, limit: int = 5) -> List[Tuple[str, float]]:
        if np.allclose(self.current_vec.sum(), 0.0):
            return []
        idx = np.where(self.current_vec >= threshold)[0]
        pairs = [(self.names[i], float(self.current_vec[i])) for i in idx]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:limit]

    def insight(self) -> Dict[str, Any]:
        dom = self.get_dominant()
        if not dom:
            return {"status": "equilibrium", "message": "Emotional balance maintained", "confidence": 1.0}
        clusters = defaultdict(list)
        vals, arous, ints = [], [], []
        for e, inten in dom:
            meta = self.emotion_spectrum.get(e)
            if not meta:
                continue
            clusters[meta["cluster"]].append((e, inten))
            vals.append(meta["valence"] * inten)
            arous.append(meta["arousal"] * inten)
            ints.append(inten)
        tot = sum(ints) if ints else 0.0
        v = sum(vals) / tot if tot else 0.0
        a = sum(arous) / tot if tot else 0.0
        return {
            "dominant_emotions": dom,
            "active_clusters": dict(clusters),
            "valence_tone": "positive" if v > 0.3 else "negative" if v < -0.3 else "neutral",
            "arousal_level": "high" if a > 0.4 else "low" if a < -0.4 else "moderate",
            "emotional_complexity": len(clusters),
            "total_intensity": float(tot),
            "confidence": min(1.0, tot / 2.0)
        }

    def forecast(self, steps: int = 3) -> List[Dict[str, float]]:
        if len(self.history) < 2:
            return [{"status": "insufficient_history"}] * max(1, steps)
        v = self.current_vec.copy()
        z = np.tanh(self.W_enc.T @ v)
        x = self.res_x.copy()
        preds: List[Dict[str, float]] = []
        for _ in range(max(1, steps)):
            x = np.tanh(self.W_res @ x + self.W_in @ z)
            W_out = _orthonormal_columns(self.config.reservoir_size, self.config.num_latent, self.rng).T
            z = np.tanh(W_out @ x)
            v_next = self._decode_and_normalize(z)
            preds.append({self.names[i]: float(v_next[i]) for i in range(self.N)})
        self.res_x = x.copy()
        return preds

    # ----- Internals -----

    def _validate_emotional_input(self, emotional_input: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, v in emotional_input.items():
            if isinstance(k, str):
                try:
                    val = float(v)
                    if 0.0 <= val <= 1.0 and k in self.emotion_spectrum:
                        out[k] = val
                except (TypeError, ValueError):
                    continue
        return out

    def _validate_context(self, context: str) -> str:
        if not isinstance(context, str):
            return "connection"
        c = context.lower().strip()
        return c if c in self.context_weights else "connection"

    def _log_interaction(self, emotional_input: Dict[str, float], context: str, vec: np.ndarray) -> None:
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "input": dict(emotional_input),
            "context": context,
            "output": {self.names[i]: float(vec[i]) for i in range(self.N)},
            "dominant": self._top(vec, 3)
        })

    def _top(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if vec.size == 0:
            return []
        idx = np.argsort(vec)[-k:][::-1]
        return [(self.names[i], float(vec[i])) for i in idx if vec[i] > 0]

    def _encode_inputs(self, intensities: Dict[str, float], context: str) -> np.ndarray:
        v = np.zeros(self.N, dtype=float)
        ctx = self.context_weights.get(context, 1.0)
        for i, name in enumerate(self.names):
            base = intensities.get(name, 0.0)
            if base > 0.0:
                v[i] = self._apply_genetics(name, base) * ctx
        return _normalize01(v)

    def _apply_genetics(self, emotion_name: str, intensity: float) -> float:
        neuro = set(self.emotion_spectrum[emotion_name].get("neuro", []))
        f = 1.0
        if "serotonin" in neuro:
            f *= self.genetic_profile["serotonin_transporter"]
        if "dopamine" in neuro:
            f *= self.genetic_profile["dopamine_receptor"]
        if "cortisol" in neuro:
            f *= (1.2 - self.genetic_profile["maoa_activity"])
        if "oxytocin" in neuro:
            f *= self.genetic_profile["oxytocin_receptor"]
        return float(np.clip(intensity * f, 0.0, 1.0))

    def _simulate_quantum_layers(self, v: np.ndarray, context: str) -> np.ndarray:
        try:
            v = self._normalize_amplitudes(v)
            v = self._apply_graph_entanglement(v)
            z = self._encode_with_phase_carriers(v)
            z = self._apply_attention_gating(z)
            z = self._apply_context_rotation(z, context)
            v_dec = self._decode_and_settle(z)
            v_fin = self._apply_decoherence_mitigation(v_dec)
            return v_fin
        except Exception as e:
            warnings.warn(f"Quantum sim fallback: {e}")
            return _normalize01(v)

    def _normalize_amplitudes(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def _apply_graph_entanglement(self, v: np.ndarray) -> np.ndarray:
        try:
            out = v - self.config.laplacian_strength * (self.L_graph @ v)
            return np.clip(out, 0.0, None)
        except Exception as e:
            warnings.warn(f"Entanglement step failed: {e}")
            return v

    def _encode_with_phase_carriers(self, v: np.ndarray) -> np.ndarray:
        try:
            z = np.tanh(self.W_enc.T @ v)
            carriers = np.stack([np.sin(self.phase + k * pi / self.config.num_latent)
                                 for k in range(self.config.num_latent)], axis=1)
            w = v / (np.linalg.norm(v) + 1e-12)
            phase_signal = w @ carriers
            return np.tanh(z * (1.0 + 0.35 * phase_signal))
        except Exception as e:
            warnings.warn(f"Phase carrier failed: {e}")
            return np.tanh(self.W_enc.T @ v)

    def _apply_attention_gating(self, z: np.ndarray) -> np.ndarray:
        try:
            att = _softmax((z[:, None] @ z[None, :]).sum(axis=1))
            return np.tanh(z * (0.8 + 0.4 * att))
        except Exception as e:
            warnings.warn(f"Attention gating failed: {e}")
            return z

    def _apply_context_rotation(self, z: np.ndarray, context: str) -> np.ndarray:
        try:
            rot = (self.context_weights.get(context, 1.0) - 1.0) * 0.5
            return np.tanh(z + rot)
        except Exception as e:
            warnings.warn(f"Context rotation failed: {e}")
            return z

    def _decode_and_settle(self, z: np.ndarray) -> np.ndarray:
        dec = self._decode_and_normalize(z)
        try:
            grad = -(self.W_graph @ dec) + 0.02 * dec
            out = dec - self.config.settling_strength * grad
            return np.clip(out, 0.0, None)
        except Exception as e:
            warnings.warn(f"Settle failed: {e}")
            return dec

    def _decode_and_normalize(self, z: np.ndarray) -> np.ndarray:
        try:
            v = self.W_dec @ z
            v = np.tanh(v) * 0.5 + 0.5
            return _normalize01(v)
        except Exception as e:
            warnings.warn(f"Decode failed: {e}")
            return np.ones(self.N) / self.N

    def _apply_decoherence_mitigation(self, v: np.ndarray) -> np.ndarray:
        try:
            bdnf = self.genetic_profile.get("bdnf_level", 0.5)
            alpha = self.config.decoherence_base + 0.10 * (1.0 - bdnf)
            base = np.ones_like(v) / self.N
            mix = (1 - alpha) * v + alpha * base
            return _normalize01(mix)
        except Exception as e:
            warnings.warn(f"Decoherence step failed: {e}")
            return _normalize01(v)

    # ----- Stats/controls -----

    def get_emotional_statistics(self) -> Dict[str, Any]:
        if np.allclose(self.current_vec.sum(), 0.0):
            return {"status": "no_active_emotions"}
        stats = {
            "total_activation": float(self.current_vec.sum()),
            "max_activation": float(self.current_vec.max()),
            "mean_activation": float(self.current_vec.mean()),
            "std_activation": float(self.current_vec.std()),
            "num_active": int(np.sum(self.current_vec > 0.1)),
            "entropy": self._entropy(),
        }
        cluster_sum = defaultdict(float)
        for i, name in enumerate(self.names):
            if self.current_vec[i] > 0:
                cluster_sum[self.emotion_spectrum[name]["cluster"]] += self.current_vec[i]
        stats["cluster_activations"] = dict(cluster_sum)
        stats["dominant_cluster"] = max(cluster_sum.items(), key=lambda x: x[1])[0] if cluster_sum else None
        return stats

    def _entropy(self) -> float:
        total = self.current_vec.sum()
        if total <= 1e-12:
            return 0.0
        p = self.current_vec / total
        p = p[p > 1e-12]
        return float(-np.sum(p * np.log2(p)))

    def reset_state(self) -> None:
        self.current_vec = np.zeros(self.N)
        self.res_x = np.zeros(self.config.reservoir_size)
        self.history.clear()

    def set_genetic_profile(self, profile: Dict[str, float]) -> None:
        for k, v in profile.items():
            if k in self.genetic_profile:
                try:
                    fv = float(v)
                    if 0.0 <= fv <= 2.0:
                        self.genetic_profile[k] = fv
                except (TypeError, ValueError):
                    warnings.warn(f"Bad genetic value for {k}: {v}")

    def set_context_weights(self, weights: Dict[str, float]) -> None:
        for k, v in weights.items():
            try:
                fv = float(v)
                if 0.1 <= fv <= 2.0:
                    self.context_weights[k] = fv
            except (TypeError, ValueError):
                warnings.warn(f"Bad context weight for {k}: {v}")

# =========================
# Drop-in Adapter
# =========================

@dataclass
class _DecayState:
    current: Dict[str, float] = field(default_factory=dict)
    last_update: Optional[datetime] = None
    def apply_decay(self, decay_rate: float = 0.85) -> None:
        dec = {}
        for e, inten in self.current.items():
            nv = inten * decay_rate
            if nv >= 0.05:
                dec[e] = nv
        self.current = dec
        self.last_update = datetime.utcnow()

class EmotionEngineDropIn:
    def __init__(self, bondholder: str = "Anima", seed: int = 42, config: Optional[ProcessorConfig] = None):
        self.processor = QuantumEmotion128ProcessorSim(bondholder=bondholder, seed=seed, config=config)
        self.state = _DecayState()
        self.memory = deque(maxlen=(config.memory_size if config else 50))
        self.lexicon = self._build_enhanced_lexicon()
        self.coarse_to_fine = self._build_emotion_mapping()
        self.intensity_patterns = self._build_intensity_patterns()

    # --- detection ---

    def _build_enhanced_lexicon(self) -> Dict[str, set]:
        return {
            "joy": {"happy","excited","delighted","wonderful","amazing","great","love","joy","elated","euphoric","blissful","ecstatic"},
            "sadness": {"sad","down","blue","grief","heartbroken","loss","melancholy","depressed","dejected","sorrowful"},
            "anger": {"angry","furious","mad","frustrated","irritated","rage","enraged","livid","irate","incensed"},
            "fear": {"scared","afraid","anxious","worried","nervous","terrified","panic","fearful","apprehensive","dread"},
            "overwhelm": {"overwhelmed","too","much","drowning","stressed","burned","out","overload","swamped","inundated"},
            "hope": {"hope","optimistic","confident","believe","faith","trust","hopeful","positive","encouraged"},
            "love": {"love","adore","cherish","care","connection","bond","belong","affection","devotion"},
            "peace": {"calm","peaceful","serene","tranquil","centered","still","relaxed","zen"},
            "wonder": {"amazed","awe","wonder","marvel","fascinated","intrigued"},
            "pride": {"proud","accomplished","achieved","successful","triumph"},
            "shame": {"ashamed","embarrassed","humiliated","guilty","regret"},
            "anticipation": {"excited","eager","looking","forward","anticipating","expecting"},
            # digital
            "FOMO": {"fomo","fear","of","missing","out","left","out"},
            "digital_fatigue": {"digital","fatigue","screen","tired","online","exhausted","zoom","fatigue"},
            "viral_validation": {"likes","shares","going","viral","trending","viral"},
            # complex
            "flow": {"in","the","zone","flow","state","immersed","absorbed","focused"},
            "nostalgia": {"nostalgic","miss","remember","when","good","old","days","reminisce"},
            "empathy": {"empathy","feel","for","understand","relate","compassion"},
        }

    def _build_emotion_mapping(self) -> Dict[str, str]:
        return {
            "joy": "euphoria",
            "sadness": "melancholy",
            "anger": "rage",
            "fear": "social_anxiety",
            "overwhelm": "sensory_overload",
            "hope": "anticipatory_joy",
            "love": "belongingness",
            "peace": "serenity",
            "wonder": "wonder",
            "pride": "mastery_satisfaction",
            "shame": "imposter_syndrome",
            "anticipation": "anticipatory_joy",
            "FOMO": "FOMO",
            "digital_fatigue": "digital_detox_relief",
            "viral_validation": "viral_validation",
            "flow": "flow",
            "nostalgia": "joyful_nostalgia",
            "empathy": "empathic_resonance",
        }

    def _build_intensity_patterns(self) -> Dict[str, float]:
        return {
            "very": 0.15, "extremely": 0.25, "absolutely": 0.20, "completely": 0.20,
            "totally": 0.15, "so": 0.10, "really": 0.10, "incredibly": 0.25,
            "utterly": 0.20, "profoundly": 0.25, "deeply": 0.20, "intensely": 0.25,
            "overwhelmingly": 0.30,
        }

    def detect(self, text: str) -> Dict[str, float]:
        if not isinstance(text, str):
            return {}
        t = text.lower()
        base = self._base_intensity(text, t)
        words = set(re.findall(r"[A-Za-z']+", t))
        out: Dict[str, float] = {}
        for cat, kws in self.lexicon.items():
            matches = words & kws
            if matches:
                match_int = len(matches) * 0.15
                out[cat] = min(1.0, base + match_int)
        # contextual patterns
        for e, val in self._detect_contextual_patterns(t).items():
            out[e] = max(out.get(e, 0.0), val)
        return out

    def _base_intensity(self, text: str, t: str) -> float:
        exclam = min(0.3, text.count("!") * 0.1)
        caps = (sum(1 for c in text if c.isupper()) / max(len(text), 1)) if text else 0.0
        caps_boost = min(0.2, caps * 2)
        intens = min(0.3, sum(boost for k, boost in self.intensity_patterns.items() if k in t))
        qmarks = min(0.1, text.count("?") * 0.05)
        return min(1.0, 0.2 + exclam + caps_boost + intens + qmarks)

    def _detect_contextual_patterns(self, t: str) -> Dict[str, float]:
        pats = {
            r"can't (believe|handle|deal)": ("overwhelm", 0.7),
            r"so (proud|happy|excited)": ("pride", 0.8),
            r"feeling (lost|confused|uncertain)": ("fear", 0.6),
            r"(miss|missed) (you|them|it)": ("nostalgia", 0.7),
            r"(grateful|thankful) for": ("joy", 0.6),
            r"(worried|concerned) about": ("fear", 0.6),
            r"(annoyed|frustrated) (with|by)": ("anger", 0.5),
            r"(love|adore) (this|that|it)": ("love", 0.8),
            r"(hate|despise) (this|that|it)": ("anger", 0.8),
            r"(amazing|incredible|wonderful) (day|time|moment)": ("joy", 0.9),
        }
        found: Dict[str, float] = {}
        for pat, (emo, inten) in pats.items():
            if re.search(pat, t, re.IGNORECASE):
                found[emo] = max(found.get(emo, 0.0), inten)
        return found

    # --- processing ---

    def process_emotional_input(self, detected: Dict[str, float], context: str = "interaction") -> Dict[str, float]:
        self.state.apply_decay()
        fine = self._map_to_fine(detected)
        mapped_ctx = self._map_context(context)
        full = self.processor.resonate(fine, context=mapped_ctx)
        self._update_state(full, detected)
        self._log(detected, fine, context, full)
        return dict(self.state.current)

    def _map_to_fine(self, detected: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for coarse, val in detected.items():
            fine = self.coarse_to_fine.get(coarse)
            if fine and fine in self.processor.emotion_spectrum:
                out[fine] = max(out.get(fine, 0.0), float(val))
        return out

    def _update_state(self, full: Dict[str, float], original: Dict[str, float]) -> None:
        dom = sorted(full.items(), key=lambda x: x[1], reverse=True)[:8]
        for k, v in dom:
            if v >= 0.15:
                self.state.current[k] = max(self.state.current.get(k, 0.0), float(v))
        for k, v in original.items():
            if v >= 0.3:
                self.state.current[k] = max(self.state.current.get(k, 0.0), float(v * 0.8))

    def _map_context(self, ctx: str) -> str:
        mapping = {
            "interaction": "connection", "chat": "connection", "conversation": "connection",
            "work": "performance", "creative": "creativity", "study": "learning",
            "social": "connection", "alone": "solitude", "digital": "digital_interaction",
        }
        c = (ctx or "").lower()
        return mapping.get(c, "connection")

    def _log(self, detected: Dict[str, float], fine: Dict[str, float], ctx: str, full: Dict[str, float]) -> None:
        self.memory.append({
            "timestamp": datetime.utcnow().isoformat(),
            "detected_coarse": dict(detected),
            "mapped_fine": dict(fine),
            "context": ctx,
            "dominant_output": sorted(full.items(), key=lambda x: x[1], reverse=True)[:5],
            "state_snapshot": dict(self.state.current)
        })

    # --- convenience / compatibility ---

    def get_emotional_summary(self) -> str:
        if not self.state.current:
            return "feeling centered and emotionally balanced"
        primary, val = max(self.state.current.items(), key=lambda x: x[1])
        if val > 0.8: word = "deeply"
        elif val > 0.6: word = "strongly"
        elif val > 0.4: word = "moderately"
        else: word = "gently"
        sec = [k for k, v in self.state.current.items() if k != primary and v > 0.3]
        return f"{word} experiencing {primary}, with traces of {', '.join(sec[:2])}" if sec else f"{word} {primary}"

    def get_emotional_trajectory(self, window: int = 10) -> List[Dict[str, Any]]:
        mem = list(self.memory)[-window:]
        traj: List[Dict[str, Any]] = []
        for m in mem:
            dom = m.get("dominant_output") or []
            primary = dom[0] if dom else ("neutral", 0.0)
            traj.append({"timestamp": m["timestamp"], "primary_emotion": primary[0], "intensity": primary[1], "context": m.get("context","unknown")})
        return traj

    def get_processor_insights(self) -> Dict[str, Any]:
        return self.processor.insight()

    def get_processor_statistics(self) -> Dict[str, Any]:
        return self.processor.get_emotional_statistics()

    def forecast_emotions(self, steps: int = 3) -> List[Dict[str, float]]:
        return self.processor.forecast(steps)

    def reset_emotional_state(self) -> None:
        self.state = _DecayState()
        self.processor.reset_state()
        self.memory.clear()

    def configure_genetics(self, profile: Dict[str, float]) -> None:
        self.processor.set_genetic_profile(profile)

    def configure_contexts(self, weights: Dict[str, float]) -> None:
        self.processor.set_context_weights(weights)

    # Back-compat aliases
    @property
    def detector(self):  # so you can do EmotionEngineDropIn().detector.detect(...)
        return self

    def dominant(self, threshold: float = 0.3, limit: int = 5) -> List[Tuple[str, float]]:
        return self.processor.get_dominant(threshold=threshold, limit=limit)

    def insight(self) -> Dict[str, Any]:
        return self.processor.insight()

    def forecast(self, steps: int = 3) -> List[Dict[str, float]]:
        return self.processor.forecast(steps)
