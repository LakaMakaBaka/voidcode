from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..skills.models import SkillMetadata
from ..skills.registry import SkillRegistry
from ..tools.contracts import ToolDefinition

type SkillExecutionStatus = Literal["ok", "error"]


@dataclass(frozen=True, slots=True)
class SkillRuntimeContext:
    name: str
    description: str
    content: str
    prompt_context: str
    execution_notes: str = ""
    source_path: str = ""


@dataclass(frozen=True, slots=True)
class SkillRuntimeBindings:
    available_tools: tuple[str, ...] = ()
    hook_phases: tuple[Literal["pre", "post"], ...] = ("pre", "post")
    capabilities: tuple[str, ...] = ("prompt_context",)


@dataclass(frozen=True, slots=True)
class SkillExecutionRequest:
    skill: SkillRuntimeContext
    prompt: str
    session_id: str
    bindings: SkillRuntimeBindings


@dataclass(frozen=True, slots=True)
class SkillExecutionResult:
    name: str
    status: SkillExecutionStatus
    prompt_context: str
    execution_notes: str
    bindings: SkillRuntimeBindings
    source_path: str = ""
    error: str | None = None

    def metadata_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "prompt_context": self.prompt_context,
            "execution_notes": self.execution_notes,
            "bindings": {
                "available_tools": list(self.bindings.available_tools),
                "hook_phases": list(self.bindings.hook_phases),
                "capabilities": list(self.bindings.capabilities),
            },
            "source_path": self.source_path,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@runtime_checkable
class SkillExecutor(Protocol):
    def execute(self, request: SkillExecutionRequest) -> SkillExecutionResult: ...


class PromptAttachmentSkillExecutor:
    def execute(self, request: SkillExecutionRequest) -> SkillExecutionResult:
        return SkillExecutionResult(
            name=request.skill.name,
            status="ok",
            prompt_context=request.skill.prompt_context,
            execution_notes=request.skill.execution_notes,
            bindings=request.bindings,
            source_path=request.skill.source_path,
        )


_WHITESPACE_PATTERN = re.compile(r"[ \t]+")


def _normalize_text(value: str) -> str:
    normalized_lines = [_WHITESPACE_PATTERN.sub(" ", line.strip()) for line in value.splitlines()]
    return "\n".join(line for line in normalized_lines if line).strip()


def build_runtime_context(skill: SkillMetadata) -> SkillRuntimeContext:
    description = _normalize_text(skill.description)
    content = _normalize_text(skill.content)
    execution_notes = content
    prompt_parts = [f"Skill: {skill.name}"]
    if description:
        prompt_parts.append(f"Description: {description}")
    if execution_notes:
        prompt_parts.append(f"Instructions:\n{execution_notes}")
    prompt_context = "\n".join(prompt_parts).strip()

    return SkillRuntimeContext(
        name=skill.name,
        description=description,
        content=content,
        prompt_context=prompt_context,
        execution_notes=execution_notes,
        source_path=str(skill.entry_path),
    )


def build_runtime_contexts(
    registry: SkillRegistry,
    *,
    skill_names: Iterable[str] | None = None,
) -> tuple[SkillRuntimeContext, ...]:
    if skill_names is None:
        skills = registry.all()
    else:
        skills = tuple(registry.resolve(skill_name) for skill_name in skill_names)
    return tuple(build_runtime_context(skill) for skill in skills)


def build_skill_prompt_context(contexts: Iterable[SkillRuntimeContext]) -> str:
    rendered = [context.prompt_context for context in contexts if context.prompt_context]
    if not rendered:
        return ""
    return (
        "Runtime-managed skills are active for this turn. "
        "Apply these instructions in addition to the user's request.\n\n" + "\n\n".join(rendered)
    )


def build_skill_runtime_bindings(
    available_tools: Iterable[ToolDefinition],
) -> SkillRuntimeBindings:
    return SkillRuntimeBindings(
        available_tools=tuple(sorted(tool.name for tool in available_tools)),
    )


def execute_runtime_contexts(
    contexts: Iterable[SkillRuntimeContext],
    *,
    prompt: str,
    session_id: str,
    bindings: SkillRuntimeBindings,
    executor: SkillExecutor,
) -> tuple[SkillExecutionResult, ...]:
    return tuple(
        executor.execute(
            SkillExecutionRequest(
                skill=context,
                prompt=prompt,
                session_id=session_id,
                bindings=bindings,
            )
        )
        for context in contexts
    )


def runtime_context_from_payload(payload: dict[str, str]) -> SkillRuntimeContext:
    name = payload["name"]
    description = payload["description"]
    content = payload["content"]
    prompt_context = payload.get("prompt_context")
    execution_notes = payload.get("execution_notes", content)
    if prompt_context is None:
        prompt_parts = [f"Skill: {name}"]
        if description:
            prompt_parts.append(f"Description: {description}")
        if execution_notes:
            prompt_parts.append(f"Instructions:\n{execution_notes}")
        prompt_context = "\n".join(prompt_parts).strip()
    return SkillRuntimeContext(
        name=name,
        description=description,
        content=content,
        prompt_context=prompt_context,
        execution_notes=execution_notes,
        source_path=payload.get("source_path", ""),
    )


__all__ = [
    "PromptAttachmentSkillExecutor",
    "SkillExecutionRequest",
    "SkillExecutionResult",
    "SkillExecutionStatus",
    "SkillExecutor",
    "SkillRuntimeContext",
    "SkillRuntimeBindings",
    "build_runtime_context",
    "build_runtime_contexts",
    "build_skill_runtime_bindings",
    "build_skill_prompt_context",
    "execute_runtime_contexts",
    "runtime_context_from_payload",
]
