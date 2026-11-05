"""Schemas for external speech generation and provider capability reporting.

Generated audio is SYNTHETIC by definition. Every generation response carries
explicit provenance and is never presented as, or mixed with, VoxSentinel's
local detection results.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.services.gnani_client import (
    TTS_CONTAINERS,
    TTS_LANGUAGES,
    TTS_MODELS,
    TTS_SAMPLE_RATES,
)


class SynthesizeRequest(BaseModel):
    """Authorized speech-generation request (Gnani.ai / Vach TTS).

    Field bounds mirror the provider's documented contract
    (docs.gnani.ai, TTS REST reference) with an additional locally enforced
    text-length ceiling.
    """

    text: str = Field(..., min_length=1, description="Text to synthesize.")
    voice: str = Field(..., min_length=1, description="Voice identifier from the provider catalog.")
    model: Literal[tuple(TTS_MODELS)] = Field(default="timbre-v2.5")  # type: ignore[valid-type]
    language: Literal[tuple(TTS_LANGUAGES)] = Field(default="auto")  # type: ignore[valid-type]
    speed: float = Field(default=1.0, ge=0.85, le=1.15)
    container: Literal[tuple(TTS_CONTAINERS)] = Field(default="wav")  # type: ignore[valid-type]
    sample_rate: Literal[tuple(TTS_SAMPLE_RATES)] = Field(default=48000)  # type: ignore[valid-type]


class ProviderCapabilitiesOut(BaseModel):
    """Honest capability report for one configured inference provider."""

    generation_enabled: bool
    generation_supported: bool
    detection_supported: bool
    provider: str
    api_name: str
    note: str


class ProvidersResponse(BaseModel):
    """Which system answers detection vs generation. The local detector is
    always listed first and remains the sole detection source."""

    detection: dict[str, Any]
    detection_note: str
    generations: list[ProviderCapabilitiesOut]


class VoiceChatTurnOut(BaseModel):
    """One completed speech-to-speech turn (NemotronLabs VoiceChat 11B).

    The agent's audio is SYNTHETIC by construction: it is generated speech
    from a model, labelled as such here and in every stored artifact.
    """

    synthetic: bool = True
    provider: str
    model: str
    generator_id: str
    audio_b64: str = Field(..., description="Agent speech as base64-encoded WAV.")
    audio_format: str = "wav"
    sample_rate: int
    channels: int = 1
    agent_transcript: str
    user_transcript: str
