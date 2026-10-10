#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
math_solver_advanced_universal.py

Advanced multi-agent math solver with:
- A_base quick solver producing difficulty / confidence / reasoning flags, etc.
- A hard-rule router that chooses one of three paths:
    * DIRECT: trust A_base (easy questions).
    * BASELINE: run baseline solver once (medium / unclear).
    * PLANNER: call SOFT_PLANNER to decide K and model_list, then run multiple solvers.
- Optional critic / judge agents (for diagnostics; final answer still comes from solver/A_base).

This file is designed to work with prompt_advanced.yaml containing:
    a_base, soft_planner, solver, critic, judge, summarizer

After solving all questions, it also prints a detailed summary:
- accuracy (if dataset has "answer" field),
- route counts (DIRECT / BASELINE / PLANNER),
- agent usage counts (A_base / SoftPlanner / Solver / Summarizer / Critic / Judge),
- A_base difficulty distribution and average confidence,
- total runtime and average runtime per question.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import requests  # type: ignore

# Optional YAML support
try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------


def render_template(template: str, **kwargs: str) -> str:
    """
    Very simple template substitution:
    - For each key, replace all occurrences of "{KEY}" with the value.
    - Unknown {xxx} placeholders are left unchanged (no KeyError).
    """
    result = template
    for key, value in kwargs.items():
        token = "{" + key + "}"
        result = result.replace(token, value)
    return result


def strip_json_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences around the content if present.
    This makes it easier to run json.loads on LLM output.
    """
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return s


def safe_json_loads(text: str) -> Any:
    """
    Safely parse JSON from an LLM string:
    - strip code fences
    - try to fix common issues (trailing commas, invalid backslash escapes)
    - if everything fails, return a fallback dict instead of raising
    """
    # 1) remove markdown fences
    s = strip_json_fences(text).strip()

    # 2) try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 3) remove trailing commas before } or ]
    s2 = re.sub(r",(\s*[}\]])", r"\1", s)
    try:
        return json.loads(s2)
    except json.JSONDecodeError:
        pass

    # 4) fix invalid backslash escapes: \X where X is not a valid JSON escape
    def _fix_backslash(m: re.Match) -> str:
        return "\\\\" + m.group(1)

    s3 = re.sub(r"\\([^\"\\/bfnrtu])", _fix_backslash, s2)
    try:
        return json.loads(s3)
    except json.JSONDecodeError:
        # 5) final fallback: do not crash, return raw text wrapped in dict
        return {"_raw_text": s}


# --------------------------------------------------------------------
# Prompt config
# --------------------------------------------------------------------


@dataclass
class PromptSection:
    system: str
    user: str


@dataclass
class AdvancedPromptConfig:
    a_base: PromptSection
    soft_planner: PromptSection
    solver: PromptSection
    critic: PromptSection
    judge: PromptSection
    summarizer: PromptSection

    @staticmethod
    def _section_from_yaml(data: Dict[str, Any], key: str) -> PromptSection:
        sec = data.get(key)
        if not sec:
            raise KeyError(f"Missing section '{key}' in prompts YAML.")
        return PromptSection(system=sec.get("system", ""), user=sec.get("user", ""))

    @classmethod
    def from_yaml(cls, path: str) -> "AdvancedPromptConfig":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML prompts.")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            a_base=cls._section_from_yaml(raw, "a_base"),
            soft_planner=cls._section_from_yaml(raw, "soft_planner"),
            solver=cls._section_from_yaml(raw, "solver"),
            critic=cls._section_from_yaml(raw, "critic"),
            judge=cls._section_from_yaml(raw, "judge"),
            summarizer=cls._section_from_yaml(raw, "summarizer"),
        )


# --------------------------------------------------------------------
# LLM client (universal API wrapper)
# --------------------------------------------------------------------


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLMClient:
    """
    Universal thin wrapper around multiple chat-completions style APIs.

    Supported providers:
      - "openai"     : OpenAI-compatible endpoint using `openai` library.
      - "deepseek"   : OpenAI-compatible endpoint at https://api.deepseek.com.
      - "openrouter" : OpenRouter API at https://openrouter.ai.
      - "gemini"     : Google Gemini via `google-generativeai`.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key or self._default_api_key()
        self.base_url = base_url
        if self.provider in {"openai", "deepseek"}:
            import openai as _openai  # type: ignore
            _openai.api_key = self.api_key
            if self.provider == "deepseek":
                _openai.base_url = self.base_url or "https://api.deepseek.com"
            elif self.base_url:
                _openai.base_url = self.base_url

    def _default_api_key(self) -> Optional[str]:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        if self.provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if self.provider == "openrouter":
            return os.getenv("OPENROUTER_API_KEY")
        if self.provider == "gemini":
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return os.getenv("OPENAI_API_KEY")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        timeout: int = 600,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        start = time.time()

        # OpenAI / DeepSeek via openai SDK
        if self.provider in {"openai", "deepseek"}:
            import openai as _openai  # type: ignore

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            resp = _openai.chat.completions.create(**kwargs)
            end = time.time()
            content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None) or {}
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            return LLMResponse(
                content=content,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_s=end - start,
            )

        # OpenRouter
        if self.provider == "openrouter":
            if not self.api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set and --api_key is missing.")
            url = self.base_url or "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            end = time.time()
            choices = data.get("choices") or []
            content = ""
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content", "") or ""
            usage = data.get("usage") or {}
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            return LLMResponse(
                content=content,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_s=end - start,
            )

        # Gemini
        if self.provider == "gemini":
            try:
                import google.generativeai as genai  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Please install google-generativeai to use provider='gemini': pip install google-generativeai"
                ) from exc

            if not self.api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY not set and --api_key is missing."
                )

            genai.configure(api_key=self.api_key)
            model_obj = genai.GenerativeModel(model)
            chat = model_obj.start_chat(history=[])

            system_parts = [m["content"] for m in messages if m.get("role") == "system"]
            user_parts = [m["content"] for m in messages if m.get("role") == "user"]
            prompt = ""
            if system_parts:
                prompt += "System instructions:\n" + "\n".join(system_parts) + "\n\n"
            if user_parts:
                prompt += "User:\n" + "\n".join(user_parts)

            resp = chat.send_message(prompt, generation_config={"temperature": temperature})
            end = time.time()
            content = resp.text or ""
            return LLMResponse(
                content=content,
                prompt_tokens=0,
                completion_tokens=0,
                latency_s=end - start,
            )

        # Generic OpenAI-compatible HTTP endpoint
        if self.base_url:
            if not self.api_key:
                raise RuntimeError("API key not set for custom base_url provider.")
            url = self.base_url.rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            end = time.time()
            choices = data.get("choices") or []
            content = ""
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content", "") or ""
            usage = data.get("usage") or {}
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            return LLMResponse(
                content=content,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_s=end - start,
            )

        raise ValueError(f"Unknown provider for LLMClient: {self.provider!r}")


# --------------------------------------------------------------------
# LLMCaller wrapper
# --------------------------------------------------------------------


class LLMCaller:
    """
    Simple wrapper to hide provider details.
    """

    def __init__(
        self,
        default_model: str,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.default_model = default_model
        self.client = LLMClient(provider=provider, api_key=api_key, base_url=base_url)

    def call(
        self,
        model_name: Optional[str],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        model = model_name or self.default_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self.client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.content


# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------


@dataclass
class ABaseResult:
    restatement: str
    quick_reasoning: str
    provisional_answer: str
    difficulty: str
    confidence: float
    flags: List[str]
    summary_for_downstream: str
    difficulty_rationale: str
    problem_type: str
    self_consistency_comment: str
    raw_text: str


@dataclass
class PlannerResult:
    route: str
    k_runs: int
    model_list: List[str]
    planner_comment: str
    raw_text: str


@dataclass
class SolverRunResult:
    solver_answer: str
    solver_reasoning: str
    raw_text: str


@dataclass
class CriticResult:
    verdict: str
    issues: str
    suggestions: str
    corrected_summary: str
    raw_text: str


@dataclass
class JudgeResult:
    label: str
    reason: str
    raw_text: str


@dataclass
class SummarizerResult:
    summary_for_cache: str
    raw_text: str


# --------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------


class ABaseAgent:
    def __init__(self, llm: LLMCaller, prompts: PromptSection) -> None:
        self.llm = llm
        self.prompts = prompts

    def build_user_prompt(self, question: str) -> str:
        # Support both {QUESTION} and {PROBLEM_TEXT} in the template
        return render_template(
            self.prompts.user,
            QUESTION=question,
            PROBLEM_TEXT=question,
        )

    def run(self, question: str, model_name: Optional[str]) -> ABaseResult:
        user_prompt = self.build_user_prompt(question)
        raw = self.llm.call(
            model_name=model_name,
            system_prompt=self.prompts.system,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1024,
        )
        obj = safe_json_loads(raw)
        return ABaseResult(
            restatement=obj.get("Restatement", ""),
            quick_reasoning=obj.get("Quick_reasoning", ""),
            provisional_answer=str(obj.get("Provisional_answer", "")),
            difficulty=obj.get("A_base_difficulty", obj.get("Difficulty", "")),
            confidence=float(obj.get("A_base_confidence", obj.get("Confidence", 0.0))),
            flags=list(obj.get("A_base_reasoning_flags", obj.get("Reasoning_flags", []))),
            summary_for_downstream=obj.get(
                "A_base_summary_for_downstream",
                obj.get("Summary_for_downstream", ""),
            ),
            difficulty_rationale=obj.get(
                "A_base_difficulty_rationale",
                obj.get("Difficulty_rationale", ""),
            ),
            problem_type=obj.get("A_base_problem_type", obj.get("Problem_type", "")),
            self_consistency_comment=obj.get(
                "A_base_self_consistency_comment",
                obj.get("Self_consistency_comment", ""),
            ),
            raw_text=raw,
        )


class SoftPlannerAgent:
    def __init__(self, llm: LLMCaller, prompts: PromptSection) -> None:
        self.llm = llm
        self.prompts = prompts

    def build_user_prompt(
        self,
        question: str,
        a_base: ABaseResult,
        available_models: Sequence[str],
        baseline_model: str,
        hard_model: Optional[str],
        max_runs: int,
    ) -> str:
        base_user = self.prompts.user

        a_base_dict = {
            "Restatement": a_base.restatement,
            "Quick_reasoning": a_base.quick_reasoning,
            "Provisional_answer": a_base.provisional_answer,
            "A_base_difficulty": a_base.difficulty,
            "A_base_confidence": a_base.confidence,
            "A_base_reasoning_flags": a_base.flags,
            "A_base_summary_for_downstream": a_base.summary_for_downstream,
            "A_base_difficulty_rationale": a_base.difficulty_rationale,
            "A_base_problem_type": a_base.problem_type,
            "A_base_self_consistency_comment": a_base.self_consistency_comment,
        }
        a_base_json = json.dumps(a_base_dict, ensure_ascii=False)
        available_models_json = json.dumps(list(available_models), ensure_ascii=False)

        return render_template(
            base_user,
            PROBLEM_TEXT=question,
            A_BASE_JSON=a_base_json,
            AVAILABLE_MODELS_JSON=available_models_json,
            BASELINE_MODEL=baseline_model,
            HARD_MODEL=hard_model or "",
            MAX_RUNS=str(max_runs),
        )

    def run(
        self,
        question: str,
        a_base: ABaseResult,
        model_name: Optional[str],
        available_models: Sequence[str],
        baseline_model: str,
        hard_model: Optional[str],
        max_runs: int,
    ) -> PlannerResult:
        user_prompt = self.build_user_prompt(
            question=question,
            a_base=a_base,
            available_models=available_models,
            baseline_model=baseline_model,
            hard_model=hard_model,
            max_runs=max_runs,
        )
        raw = self.llm.call(
            model_name=model_name,
            system_prompt=self.prompts.system,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1024,
        )
        obj = safe_json_loads(raw)
        route = str(obj.get("Route", obj.get("route", ""))).upper()
        k_runs = int(obj.get("K_runs", obj.get("k_runs", obj.get("K", 1))))
        model_list = [str(x) for x in obj.get("Model_list", obj.get("model_list", []))]
        planner_comment = obj.get(
            "Planner_comment",
            obj.get("planner_comment", obj.get("solve_plan", "")),
        )
        return PlannerResult(
            route=route,
            k_runs=k_runs,
            model_list=model_list,
            planner_comment=planner_comment,
            raw_text=raw,
        )


class SolverAgent:
    def __init__(self, llm: LLMCaller, prompts: PromptSection) -> None:
        self.llm = llm
        self.prompts = prompts

    def build_user_prompt(self, question: str, extra_context: str = "") -> str:
        base = self.prompts.user
        if extra_context:
            inject = textwrap.dedent(
                f"""\
                [CONTEXT_FROM_PREVIOUS_RUNS]
                {extra_context}
                [/CONTEXT_FROM_PREVIOUS_RUNS]
                """
            ).strip()
            base = base + "\n\n" + inject
        return render_template(base, QUESTION=question)

    def run(
        self,
        question: str,
        model_name: Optional[str],
        extra_context: str = "",
        temperature: float = 0.0,
    ) -> SolverRunResult:
        user_prompt = self.build_user_prompt(question, extra_context)
        raw = self.llm.call(
            model_name=model_name,
            system_prompt=self.prompts.system,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=2048,
        )
        obj = safe_json_loads(raw)
        return SolverRunResult(
            solver_answer=str(obj.get("Final_answer", obj.get("final_answer", ""))),
            solver_reasoning=obj.get("Reasoning", obj.get("reasoning", "")),
            raw_text=raw,
        )


class CriticAgent:
    def __init__(self, llm: LLMCaller, prompts: PromptSection) -> None:
        self.llm = llm
        self.prompts = prompts

    def build_user_prompt(self, question: str, solver_output: str) -> str:
        base = self.prompts.user
        inject = textwrap.dedent(
            f"""\
            [SOLVER_OUTPUT]
            {solver_output}
            [/SOLVER_OUTPUT]
            """
        ).strip()
        base = base + "\n\n" + inject
        return render_template(
            base,
            QUESTION=question,
            SOLVER_OUTPUT_JSON=solver_output,
        )

    def run(self, question: str, solver_output: str, model_name: Optional[str]) -> CriticResult:
        user_prompt = self.build_user_prompt(question, solver_output)
        raw = self.llm.call(
            model_name=model_name,
            system_prompt=self.prompts.system,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1024,
        )
        obj = safe_json_loads(raw)
        return CriticResult(
            verdict=obj.get("Critic_verdict", ""),
            issues=obj.get("Critic_issues", ""),
            suggestions=obj.get("Critic_suggestions", ""),
            corrected_summary=obj.get("Corrected_summary_for_cache", ""),
            raw_text=raw,
        )


class JudgeAgent:
    def __init__(self, llm: LLMCaller, prompts: PromptSection) -> None:
        self.llm = llm
        self.prompts = prompts

    def build_user_prompt(
        self,
        question: str,
        solver_output: str,
        critic_output: str,
    ) -> str:
        base = self.prompts.user
        inject = textwrap.dedent(
            f"""\
            [SOLVER_OUTPUT]
            {solver_output}
            [/SOLVER_OUTPUT]

            [CRITIC_OUTPUT]
            {critic_output}
            [/CRITIC_OUTPUT]
            """
        ).strip()
        base = base + "\n\n" + inject
        return render_template(
            base,
            QUESTION=question,
            SOLVER_OUTPUT_JSON=solver_output,
            CRITIC_OUTPUT_JSON=critic_output,
        )

    def run(
        self,
        question: str,
        solver_output: str,
        critic_output: str,
        model_name: Optional[str],
    ) -> JudgeResult:
        user_prompt = self.build_user_prompt(question, solver_output, critic_output)
        raw = self.llm.call(
            model_name=model_name,
            system_prompt=self.prompts.system,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=512,
        )
        obj = safe_json_loads(raw)
        return JudgeResult(
            label=obj.get("Judge_label", ""),
            reason=obj.get("Judge_reason", ""),
            raw_text=raw,
        )


class SummarizerAgent:
    def __init__(self, llm: LLMCaller, prompts: PromptSection) -> None:
        self.llm = llm
        self.prompts = prompts

    def build_user_prompt(
        self,
        question: str,
        a_base: ABaseResult,
        solver_runs: Sequence[SolverRunResult],
    ) -> str:
        base = self.prompts.user

        runs_block_lines: List[str] = []
        for i, r in enumerate(solver_runs, start=1):
            runs_block_lines.append(f"[RUN_{i}_ANSWER] {r.solver_answer}")
            runs_block_lines.append(f"[RUN_{i}_REASONING] {r.solver_reasoning}")
        runs_block = "\n".join(runs_block_lines)

        inject = textwrap.dedent(
            f"""\
            [A_BASE_METADATA]
            restatement: {a_base.restatement}
            provisional_answer: {a_base.provisional_answer}
            difficulty: {a_base.difficulty}
            confidence: {a_base.confidence}
            flags: {",".join(a_base.flags)}
            summary_for_downstream: {a_base.summary_for_downstream}
            difficulty_rationale: {a_base.difficulty_rationale}
            problem_type: {a_base.problem_type}
            self_consistency_comment: {a_base.self_consistency_comment}
            [/A_BASE_METADATA]

            [SOLVER_RUNS]
            {runs_block}
            [/SOLVER_RUNS]
            """
        ).strip()

        base = base + "\n\n" + inject
        return render_template(base, QUESTION=question)

    def run(
        self,
        question: str,
        a_base: ABaseResult,
        solver_runs: Sequence[SolverRunResult],
        model_name: Optional[str],
    ) -> SummarizerResult:
        user_prompt = self.build_user_prompt(question, a_base, solver_runs)
        raw = self.llm.call(
            model_name=model_name,
            system_prompt=self.prompts.system,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1024,
        )
        obj = safe_json_loads(raw)
        return SummarizerResult(
            summary_for_cache=obj.get("Summary_for_cache", ""),
            raw_text=raw,
        )


# --------------------------------------------------------------------
# Router logic
# --------------------------------------------------------------------


POSITIVE_FLAGS = {"NO_ISSUE", "DOUBLE_CHECKED"}
NEGATIVE_FLAGS = {"GUESSED", "INCOMPLETE_REASONING", "CONTRADICTION_DETECTED"}


def parse_a_base_result(raw_text: str) -> ABaseResult:
    """
    Parse A_base JSON output.

    Accepts both A_base_* keys and legacy keys without the prefix.
    """
    obj = safe_json_loads(raw_text)
    return ABaseResult(
        restatement=obj.get("Restatement", ""),
        quick_reasoning=obj.get("Quick_reasoning", ""),
        provisional_answer=str(obj.get("Provisional_answer", "")),
        difficulty=obj.get("A_base_difficulty", obj.get("Difficulty", "")),
        confidence=float(obj.get("A_base_confidence", obj.get("Confidence", 0.0))),
        flags=list(obj.get("A_base_reasoning_flags", obj.get("Reasoning_flags", []))),
        summary_for_downstream=obj.get(
            "A_base_summary_for_downstream",
            obj.get("Summary_for_downstream", ""),
        ),
        difficulty_rationale=obj.get(
            "A_base_difficulty_rationale",
            obj.get("Difficulty_rationale", ""),
        ),
        problem_type=obj.get("A_base_problem_type", obj.get("Problem_type", "")),
        self_consistency_comment=obj.get(
            "A_base_self_consistency_comment",
            obj.get("Self_consistency_comment", ""),
        ),
        raw_text=raw_text,
    )


def should_use_direct_route(a_base: ABaseResult) -> bool:
    """
    Hard-rule router for DIRECT path.
    """
    if a_base.difficulty.upper() != "EASY":
        return False
    if a_base.confidence < 0.8:
        return False
    flag_set = {f.upper() for f in a_base.flags}
    if flag_set & NEGATIVE_FLAGS:
        return False
    return True


def should_use_planner_route(a_base: ABaseResult) -> bool:
    """
    Hard-rule router for PLANNER path.
    """
    diff = a_base.difficulty.upper()
    if diff in {"HARD", "VERY_HARD"}:
        return True
    flag_set = {f.upper() for f in a_base.flags}
    if "CONTRADICTION_DETECTED" in flag_set:
        return True
    return False


# --------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------


@dataclass
class AdvancedRunResult:
    question: str
    final_answer: str
    route: str
    a_base: ABaseResult
    planner: Optional[PlannerResult]
    solver_runs: List[SolverRunResult]
    critic: Optional[CriticResult]
    judge: Optional[JudgeResult]
    summarizer: Optional[SummarizerResult]


def run_advanced_for_question(
    question: str,
    cfg: AdvancedPromptConfig,
    llm: LLMCaller,
    baseline_model: str,
    a_base_model: Optional[str],
    planner_model: Optional[str],
    hard_model: Optional[str],
    max_hard_runs: int,
    use_critic: bool,
    use_judge: bool,
) -> AdvancedRunResult:
    """
    Orchestrate the advanced solver logic for a single question.
    """

    # 1) A_base
    a_base_agent = ABaseAgent(llm, cfg.a_base)
    a_base_model_name = a_base_model or baseline_model
    a_base_res = a_base_agent.run(question, a_base_model_name)

    # 2) Hard-rule router
    if should_use_direct_route(a_base_res):
        route = "DIRECT"
    elif should_use_planner_route(a_base_res):
        route = "PLANNER"
    else:
        route = "BASELINE"

    planner_res: Optional[PlannerResult] = None
    solver_runs: List[SolverRunResult] = []
    critic_res: Optional[CriticResult] = None
    judge_res: Optional[JudgeResult] = None
    summarizer_res: Optional[SummarizerResult] = None

    solver_agent = SolverAgent(llm, cfg.solver)
    critic_agent = CriticAgent(llm, cfg.critic) if use_critic else None
    judge_agent = JudgeAgent(llm, cfg.judge) if use_judge else None
    summarizer_agent = SummarizerAgent(llm, cfg.summarizer)

    # 3) Execute chosen route
    if route == "DIRECT":
        solver_runs.append(
            SolverRunResult(
                solver_answer=a_base_res.provisional_answer,
                solver_reasoning=a_base_res.quick_reasoning,
                raw_text=a_base_res.raw_text,
            )
        )
        final_answer = a_base_res.provisional_answer

    elif route == "BASELINE":
        solver_res = solver_agent.run(
            question=question,
            model_name=baseline_model,
            extra_context=a_base_res.summary_for_downstream,
            temperature=0.0,
        )
        solver_runs.append(solver_res)
        final_answer = solver_res.solver_answer

        if critic_agent and judge_agent:
            critic_res = critic_agent.run(question, solver_res.raw_text, baseline_model)
            judge_res = judge_agent.run(
                question,
                solver_res.raw_text,
                critic_res.raw_text,
                baseline_model,
            )

    else:  # PLANNER
        planner_agent = SoftPlannerAgent(llm, cfg.soft_planner)
        soft_planner_model = planner_model or baseline_model

        available_solver_models: List[str] = []
        if baseline_model:
            available_solver_models.append(baseline_model)
        if hard_model and hard_model not in available_solver_models:
            available_solver_models.append(hard_model)

        planner_res = planner_agent.run(
            question=question,
            a_base=a_base_res,
            model_name=soft_planner_model,
            available_models=available_solver_models,
            baseline_model=baseline_model,
            hard_model=hard_model,
            max_runs=max_hard_runs,
        )

        k = max(1, min(planner_res.k_runs, max_hard_runs))
        model_list = planner_res.model_list or [hard_model or baseline_model]

        for i in range(k):
            m = model_list[i % len(model_list)]
            solver_res = solver_agent.run(
                question=question,
                model_name=m,
                extra_context=a_base_res.summary_for_downstream,
                temperature=0.2,
            )
            solver_runs.append(solver_res)

        summarizer_res = summarizer_agent.run(
            question=question,
            a_base=a_base_res,
            solver_runs=solver_runs,
            model_name=baseline_model,
        )

        answers = [s.solver_answer for s in solver_runs if s.solver_answer]
        if answers:
            counts: Dict[str, int] = {}
            for ans in answers:
                counts[ans] = counts.get(ans, 0) + 1
            final_answer = max(counts.items(), key=lambda kv: kv[1])[0]
        else:
            final_answer = a_base_res.provisional_answer

        if critic_agent and judge_agent:
            chosen_run = next(
                (s for s in solver_runs if s.solver_answer == final_answer),
                solver_runs[0],
            )
            critic_res = critic_agent.run(question, chosen_run.raw_text, baseline_model)
            judge_res = judge_agent.run(
                question,
                chosen_run.raw_text,
                critic_res.raw_text,
                baseline_model,
            )

    return AdvancedRunResult(
        question=question,
        final_answer=final_answer,
        route=route,
        a_base=a_base_res,
        planner=planner_res,
        solver_runs=solver_runs,
        critic=critic_res,
        judge=judge_res,
        summarizer=summarizer_res,
    )


# --------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL or JSON list.
    Each example must have at least 'question' field.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Try JSONL
    try:
        objs: List[Dict[str, Any]] = []
        for ln in lines:
            objs.append(json.loads(ln))
        if objs and isinstance(objs[0], dict):
            return objs
    except Exception:
        pass
    # Fallback: JSON list
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported data format. Use JSONL or JSON list.")


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced multi-agent math solver with A_base + soft planner."
    )
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--baseline_model", type=str, required=True)
    parser.add_argument("--a_base_model", type=str, default=None)
    parser.add_argument("--planner_model", type=str, default=None)
    parser.add_argument("--hard_model", type=str, default=None)
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider: openai, deepseek, openrouter, gemini",
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument(
        "--max_hard_runs",
        type=int,
        default=3,
        help="Upper bound on K when using the PLANNER route.",
    )
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_critic", action="store_true")
    parser.add_argument("--no_judge", action="store_true")

    args = parser.parse_args()

    cfg = AdvancedPromptConfig.from_yaml(args.prompts)
    llm = LLMCaller(
        default_model=args.baseline_model,
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    results: List[Dict[str, Any]] = []

    overall_start = time.time()

    if args.data:
        dataset = load_dataset(args.data)
        for idx, ex in enumerate(dataset):
            q = str(ex.get("question", ""))
            if not q:
                continue
            t0 = time.time()
            res = run_advanced_for_question(
                question=q,
                cfg=cfg,
                llm=llm,
                baseline_model=args.baseline_model,
                a_base_model=args.a_base_model,
                planner_model=args.planner_model,
                hard_model=args.hard_model,
                max_hard_runs=args.max_hard_runs,
                use_critic=not args.no_critic,
                use_judge=not args.no_judge,
            )
            elapsed = time.time() - t0
            solver_calls = len(res.solver_runs)
            result_row: Dict[str, Any] = {
                "index": idx,
                "question": q,
                "final_answer": res.final_answer,
                "route": res.route,
                "elapsed_s": elapsed,
                "solver_calls": solver_calls,
                "planner_used": res.planner is not None,
                "summarizer_used": res.summarizer is not None,
                "critic_used": res.critic is not None,
                "judge_used": res.judge is not None,
                "a_base": {
                    "difficulty": res.a_base.difficulty,
                    "confidence": res.a_base.confidence,
                    "flags": res.a_base.flags,
                    "provisional_answer": res.a_base.provisional_answer,
                },
            }
            results.append(result_row)
    else:
        if not args.question:
            raise ValueError("Either --question or --data must be provided.")
        q = args.question
        t0 = time.time()
        res = run_advanced_for_question(
            question=q,
            cfg=cfg,
            llm=llm,
            baseline_model=args.baseline_model,
            a_base_model=args.a_base_model,
            planner_model=args.planner_model,
            hard_model=args.hard_model,
            max_hard_runs=args.max_hard_runs,
            use_critic=not args.no_critic,
            use_judge=not args.no_judge,
        )
        elapsed = time.time() - t0
        solver_calls = len(res.solver_runs)
        result_row = {
            "index": 0,
            "question": q,
            "final_answer": res.final_answer,
            "route": res.route,
            "elapsed_s": elapsed,
            "solver_calls": solver_calls,
            "planner_used": res.planner is not None,
            "summarizer_used": res.summarizer is not None,
            "critic_used": res.critic is not None,
            "judge_used": res.judge is not None,
            "a_base": {
                "difficulty": res.a_base.difficulty,
                "confidence": res.a_base.confidence,
                "flags": res.a_base.flags,
                "provisional_answer": res.a_base.provisional_answer,
            },
        }
        results.append(result_row)

    # write or print per-question results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        if len(results) == 1:
            print(json.dumps(results[0], ensure_ascii=False, indent=2))
        else:
            for row in results:
                print(json.dumps(row, ensure_ascii=False))
                print("-" * 80)

    # ------------------------------------------------------------------
    # Detailed summary statistics
    # ------------------------------------------------------------------
    overall_elapsed = time.time() - overall_start

    if results:
        total = len(results)

        # Load gold answers if available
        gold_answers: List[str] = []
        if args.data:
            try:
                gold_data = load_dataset(args.data)
                for ex in gold_data:
                    gold_answers.append(str(ex.get("answer", "")).strip())
            except Exception:
                gold_answers = []

        correct = 0
        route_count: Dict[str, int] = {"DIRECT": 0, "BASELINE": 0, "PLANNER": 0, "OTHER": 0}
        difficulty_count: Dict[str, int] = {}
        total_conf = 0.0
        conf_count = 0
        total_solver_calls = 0
        planner_calls = 0
        summarizer_calls = 0
        critic_calls = 0
        judge_calls = 0
        total_elapsed = 0.0

        for i, r in enumerate(results):
            # correctness
            if i < len(gold_answers):
                if str(r.get("final_answer", "")).strip() == gold_answers[i]:
                    correct += 1

            # routes
            route = str(r.get("route", "")).upper()
            if route in route_count:
                route_count[route] += 1
            else:
                route_count["OTHER"] += 1

            # A_base difficulty / confidence
            a_base = r.get("a_base", {})
            diff = str(a_base.get("difficulty", "")).upper()
            if diff:
                difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
            conf = a_base.get("confidence", None)
            try:
                conf_val = float(conf)
                total_conf += conf_val
                conf_count += 1
            except Exception:
                pass

            # agent usage counts
            sc = int(r.get("solver_calls", 0))
            total_solver_calls += sc
            if r.get("planner_used"):
                planner_calls += 1
            if r.get("summarizer_used"):
                summarizer_calls += 1
            if r.get("critic_used"):
                critic_calls += 1
            if r.get("judge_used"):
                judge_calls += 1

            # timing
            try:
                total_elapsed += float(r.get("elapsed_s", 0.0))
            except Exception:
                pass

        accuracy = (correct / total * 100.0) if total > 0 and gold_answers else None
        avg_conf = (total_conf / conf_count) if conf_count > 0 else None
        avg_solver_calls = (total_solver_calls / total) if total > 0 else 0.0
        avg_time_per_q = (total_elapsed / total) if total > 0 else 0.0

        print("\n===== ADVANCED SOLVER SUMMARY =====")
        print(f"Total questions            : {total}")
        if gold_answers:
            print(f"Correct                    : {correct}")
            print(f"Accuracy                   : {accuracy:.1f}%")
        else:
            print("Accuracy                   : (no gold answers available)")
        print(f"Routes count               : {route_count}")
        print(f"A_base difficulty counts   : {difficulty_count if difficulty_count else 'N/A'}")
        if avg_conf is not None:
            print(f"A_base avg confidence      : {avg_conf:.3f}")
        else:
            print("A_base avg confidence      : N/A")
        print(f"A_base calls               : {total}  (always 1 per question)")
        print(f"SoftPlanner used (questions): {planner_calls}")
        print(f"Summarizer used (questions): {summarizer_calls}")
        print(f"Critic used (questions)    : {critic_calls}")
        print(f"Judge used (questions)     : {judge_calls}")
        print(f"Solver total calls         : {total_solver_calls}")
        print(f"Solver avg calls / question: {avg_solver_calls:.2f}")
        print(f"Total runtime (s)          : {overall_elapsed:.2f}")
        print(f"Sum of per-question time   : {total_elapsed:.2f}")
        print(f"Avg time / question (s)    : {avg_time_per_q:.2f}")
        print("===================================\n")


if __name__ == "__main__":
    main()
