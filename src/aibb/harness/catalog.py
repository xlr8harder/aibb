"""Authoritative OpenRouter model metadata used to price a run."""

from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from aibb.runtime.models import ReasoningConfiguration


def public_openrouter_model_id(model_id: str) -> str:
    """Remove OpenRouter's free-route selector from the stable model identity."""

    return model_id.removesuffix(":free")


class OpenRouterModelRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    context_length: int
    pricing: dict[str, object]
    architecture: dict[str, object] = {}
    supported_parameters: list[str] = Field(default_factory=list)
    top_provider: dict[str, object] = {}
    reasoning: dict[str, object] | None = None

    @property
    def input_modalities(self) -> set[str]:
        values = self.architecture.get("input_modalities") or []
        return {str(value) for value in values}

    @property
    def supports_image_input(self) -> bool:
        return "image" in self.input_modalities

    @property
    def developer(self) -> str:
        prefix, separator, _remainder = self.name.partition(":")
        return prefix.strip() if separator and prefix.strip() else self.id.split("/", 1)[0]

    @property
    def display_name(self) -> str:
        """Return the public model name without route-provider decoration."""

        _prefix, separator, remainder = self.name.partition(":")
        return remainder.strip() if separator and remainder.strip() else self.name.strip()

    @property
    def effective_context_length(self) -> int:
        """Clamp the model maximum to the currently selected provider ceiling."""

        provider_value = self.top_provider.get("context_length")
        if provider_value is None:
            return self.context_length
        provider_context_length = int(provider_value)
        return min(self.context_length, provider_context_length) if provider_context_length > 0 else self.context_length

    def select_reasoning(
        self,
        override: Literal["auto", "enabled", "mandatory", "disabled"] = "auto",
        effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    ) -> ReasoningConfiguration:
        supported_efforts = [str(value) for value in (self.reasoning or {}).get("supported_efforts") or []]
        if effort is not None:
            if override == "disabled":
                raise ValueError("A reasoning effort cannot be selected when reasoning is disabled")
            if "reasoning" not in self.supported_parameters:
                raise ValueError(f"{self.id} does not expose configurable reasoning effort")
            if supported_efforts and effort not in supported_efforts:
                raise ValueError(
                    f"{self.id} does not advertise reasoning effort {effort}; "
                    f"supported efforts: {', '.join(supported_efforts)}"
                )
        if override != "auto":
            enabled = override != "disabled"
            return ReasoningConfiguration(
                enabled=enabled,
                mandatory=override == "mandatory",
                supported_efforts=supported_efforts,
                selected_effort=effort if enabled else None,
                request_parameter=(
                    {"effort": effort, "exclude": False} if effort is not None else {"enabled": enabled}
                ),
                source="curator-override",
            )
        if not self.reasoning:
            return ReasoningConfiguration()
        mandatory = bool(self.reasoning.get("mandatory", False))
        default_enabled = bool(self.reasoning.get("default_enabled", mandatory))
        default_effort = self.reasoning.get("default_effort")
        selected_effort = (
            effort
            if effort is not None
            else "high"
            if "high" in supported_efforts
            else supported_efforts[0]
            if supported_efforts
            else str(default_effort)
            if default_effort and str(default_effort) != "none"
            else None
        )
        if "reasoning" in self.supported_parameters:
            request = {"effort": selected_effort, "exclude": False} if selected_effort else {"enabled": True}
            return ReasoningConfiguration(
                enabled=True,
                mandatory=mandatory,
                supported_efforts=supported_efforts,
                selected_effort=selected_effort,
                request_parameter=request,
                source="curator-override" if effort is not None else "openrouter-catalog",
            )
        return ReasoningConfiguration(
            enabled=mandatory or default_enabled,
            mandatory=mandatory,
            supported_efforts=supported_efforts,
            selected_effort=selected_effort,
            source="provider-default" if mandatory or default_enabled else "unavailable",
        )

    @property
    def prompt_price(self) -> float:
        return float(str(self.pricing["prompt"]))

    @property
    def completion_price(self) -> float:
        return float(str(self.pricing["completion"]))

    @property
    def max_completion_tokens(self) -> int | None:
        value = self.top_provider.get("max_completion_tokens")
        return int(value) if value is not None else None

    def clamp_output_tokens(self, requested: int) -> int:
        context_length = self.effective_context_length
        ceiling = self.max_completion_tokens or context_length
        return min(requested, ceiling, max(1, context_length - 4096))

    def recommend_cost_ceiling(self, *, provider_turns: int, output_tokens_per_turn: int) -> float:
        average_input_tokens = min(60_000, max(8_000, self.effective_context_length // 8))
        average_output_tokens = min(4_000, output_tokens_per_turn)
        estimate = provider_turns * (
            average_input_tokens * self.prompt_price + average_output_tokens * self.completion_price
        )
        return round(max(0.5, estimate * 1.5), 2)


class OpenRouterImageModelRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    architecture: dict[str, object] = {}

    @property
    def output_modalities(self) -> set[str]:
        values = self.architecture.get("output_modalities") or []
        return {str(value) for value in values}


class OpenRouterEndpointRecord(BaseModel):
    """One live model endpoint from OpenRouter's provider catalog."""

    model_config = ConfigDict(extra="allow")

    name: str
    model_id: str
    provider_name: str
    tag: str
    context_length: int = Field(ge=1)
    pricing: dict[str, object]
    quantization: str | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1)
    supported_parameters: list[str] = Field(default_factory=list)
    supports_tool_choice: dict[str, bool] | None = None

    @property
    def prompt_price(self) -> float:
        return float(str(self.pricing["prompt"]))

    @property
    def completion_price(self) -> float:
        return float(str(self.pricing["completion"]))

    @property
    def output_token_parameter(self) -> Literal["max_tokens", "max_completion_tokens"]:
        """Use the completion-limit parameter the pinned endpoint advertises.

        OpenRouter's model-level catalog may advertise both spellings while a
        specific provider endpoint accepts only one.  When provider routing uses
        ``require_parameters``, sending the other spelling causes OpenRouter to
        filter an otherwise healthy endpoint out with a misleading 404.
        """

        supported = set(self.supported_parameters)
        if "max_completion_tokens" in supported and "max_tokens" not in supported:
            return "max_completion_tokens"
        return "max_tokens"

    def accepts_tool_choice(self, value: Literal["auto", "required"]) -> bool:
        """Honor OpenRouter's value-level capability when it is advertised.

        Older endpoint records expose only the generic ``tool_choice`` entry in
        ``supported_parameters``. Newer records may omit that entry and instead
        report support for each accepted value in ``supports_tool_choice``.
        """

        if self.supports_tool_choice is not None:
            return bool(self.supports_tool_choice.get(value, False))
        return "tool_choice" in self.supported_parameters

    @property
    def supports_openrouter_parameter_filter(self) -> bool:
        """Whether OpenRouter's legacy require-parameters filter sees tool choice.

        OpenRouter can advertise working value-level tool-choice support while
        omitting the generic parameter from ``supported_parameters``. Its router
        still rejects those endpoints when ``require_parameters`` is true, so a
        locally verified, provider-pinned route must bypass that legacy filter.
        """

        return "tool_choice" in self.supported_parameters


async def fetch_openrouter_endpoint(
    model_id: str,
    provider_slug: str,
    *,
    tool_choice: Literal["auto", "required"] = "auto",
    transport: httpx.AsyncBaseTransport | None = None,
) -> OpenRouterEndpointRecord:
    """Resolve and verify a specific provider route before binding a run."""

    async with httpx.AsyncClient(timeout=30, transport=transport) as client:
        response = await client.get(f"https://openrouter.ai/api/v1/models/{model_id}/endpoints")
    response.raise_for_status()
    matches: list[OpenRouterEndpointRecord] = []
    for item in response.json()["data"]["endpoints"]:
        record = OpenRouterEndpointRecord.model_validate(item)
        if record.tag == provider_slug or record.tag.startswith(f"{provider_slug}/"):
            matches.append(record)
    if not matches:
        raise ValueError(f"OpenRouter model {model_id!r} has no endpoint matching provider {provider_slug!r}")
    endpoint = sorted(matches, key=lambda item: item.tag)[0]
    supported = set(endpoint.supported_parameters)
    missing = {"tools"} - supported
    if missing:
        raise ValueError(
            f"OpenRouter endpoint {endpoint.tag!r} for {model_id!r} does not advertise required parameters: "
            + ", ".join(sorted(missing))
        )
    if not endpoint.accepts_tool_choice(tool_choice):
        raise ValueError(
            f"OpenRouter endpoint {endpoint.tag!r} for {model_id!r} does not advertise "
            f"tool_choice={tool_choice!r} support"
        )
    if not {"max_tokens", "max_completion_tokens"} & supported:
        raise ValueError(
            f"OpenRouter endpoint {endpoint.tag!r} for {model_id!r} does not advertise an output-token parameter"
        )
    return endpoint


async def fetch_openrouter_model(
    model_id: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> OpenRouterModelRecord:
    async with httpx.AsyncClient(timeout=30, transport=transport) as client:
        response = await client.get("https://openrouter.ai/api/v1/models")
    response.raise_for_status()
    for item in response.json()["data"]:
        if item["id"] == model_id:
            record = OpenRouterModelRecord.model_validate(item)
            if "tools" not in record.supported_parameters:
                raise ValueError(f"OpenRouter model {model_id!r} does not advertise tool support")
            return record
    raise ValueError(f"OpenRouter model {model_id!r} is not in the current model catalog")


async def fetch_openrouter_image_model(
    model_id: str,
    *,
    api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> OpenRouterImageModelRecord:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    async with httpx.AsyncClient(timeout=30, transport=transport) as client:
        response = await client.get("https://openrouter.ai/api/v1/images/models", headers=headers)
    response.raise_for_status()
    for item in response.json()["data"]:
        if item["id"] == model_id:
            record = OpenRouterImageModelRecord.model_validate(item)
            if "image" not in record.output_modalities:
                raise ValueError(f"OpenRouter image model {model_id!r} does not advertise image output")
            return record
    raise ValueError(f"OpenRouter image model {model_id!r} is not in the current image catalog")
