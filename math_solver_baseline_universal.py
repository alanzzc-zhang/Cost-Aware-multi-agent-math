#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline multi-agent math solver (API version).

Agents: Retriever (tool), Solver, Critic, Judge, Coordinator.
- Prompts are stored in a separate YAML file (see prompts.yaml template).
- LLM calls are done via the OpenAI Python client (Chat Completions).
- Evaluates a JSON dataset of math word problems (GSM8K / U-MATH style).

Usage (example):

    pip install openai pyyaml

    python math_solver_baseline.py \
        --model gpt-5.1-mini \
        --data gsm8k_sample10.json \
        --prompts prompts.yaml \
        --out_jsonl results.jsonl

Expected input JSON format (list of examples):

[
  {
    "id": 0,
    "question": "Alice has 3 apples ...",
    "answer": "12"
  },
  ...
]

Each example must have:
- "question" : the text of the math word problem
- "answer"   : the ground truth numeric answer (string or number)
- "id"       : optional identifier (if missing, index is used)

Output JSONL (one line per example) will contain:
- id
- question
- gold
- prediction_raw
- prediction_value
- correct_numeric (0/1)
- judge_label
- judge_reason
- agent_calls
- prompt_tokens
- completion_tokens
- latency_wall_s_sum
- solver_output
- critic_output
- judge_output

Example of a single JSONL line:

{
  "id": 0,
  "question": "Alice has ...",
  "gold": "12",
  "prediction_raw": "The answer is 12.",
  "prediction_value": "12",
  "correct_numeric": 1,
  "judge_label": "CORRECT",
  "judge_reason": "Matches the numeric answer.",
  "agent_calls": 3,
  "prompt_tokens": 150,
  "completion_tokens": 120,
  "latency_wall_s_sum": 3.50,
  "solver_output": "... raw text ...",
  "critic_output": "... raw text ...",
  "judge_output": "... raw text ..."
}

Example dataset entry for frontier-like problems:

{
    "id": 0,
    "question": "Let a sequence (a_n) be defined by a_0 = 0, a_1 = 1 and a_{n+1} = 4 a_n - a_{n-1} + 3 for n >= 1. Find a closed form for a_n and then compute a_6.",
    "answer": "219"
}

IMPORTANT:
- This script assumes your prompts.yaml defines the keys:
    solver.system
    solver.user
    critic.system
    critic.user
    judge.system
    judge.user

  and that solver/critic/judge outputs follow the field labels described
  in your prompt design, e.g.:

  Solver output JSON fields:
    Restatement
    Reasoning
    Final answer
    Summary_for_cache

  Critic output JSON fields:
    Critic_verdict
    Critic_issues
    Critic_suggestions
    Corrected_summary_for_cache

  Judge output JSON fields:
    Judge_label
    Judge_reason

"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


# ================================
# Simple helper functions
# ================================


def safe_strip(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_answer(x: Any) -> str:
    """Normalize numeric-ish answers for comparison.

    - Strip spaces
    - If it can be parsed as int, return canonical int string
    - If it can be parsed as float, return canonical float string
    - Otherwise, lowercase the string.

    This is a conservative normalization, mainly to trim superficial differences.
    """
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    # Remove trailing period if it's clearly a sentence final.
    if s.endswith(".") and all(c.isdigit() or c in "+-." for c in s[:-1]):
        s = s[:-1]

    # Try integer
    try:
        v = int(s)
        return str(v)
    except Exception:
        pass

    # Try float
    try:
        v = float(s)
        # Avoid scientific notation for simple floats.
        if v.is_integer():
            return str(int(v))
        return str(v)
    except Exception:
        pass

    return s.lower()


def extract_last_number(text: Any) -> Optional[str]:
    """Extract the last integer or float from a string.

    This is a fallback numeric parser for solver outputs that are not fully structured
    but end with "... so the answer is 42.".
    """
    if not text:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    if not nums:
        return None
    return nums[-1]


def sum_tokens(metrics_list: List[Dict[str, Any]]) -> Tuple[int, int]:
    p_total = 0
    c_total = 0
    for m in metrics_list:
        p_total += int(m.get("prompt_tokens", 0) or 0)
        c_total += int(m.get("completion_tokens", 0) or 0)
    return p_total, c_total


# ================================
# LLM client
# ================================


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLMClient:
    """A small wrapper for different provider backends.

    Currently supports:
    - openai (Chat Completions API)
    - deepseek (OpenAI-compatible)
    - openrouter (OpenAI-compatible)
    - gemini (Google Generative AI)
    - generic_http (simple POST with JSON)
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url

        if self.provider in ("openai", "deepseek", "openrouter"):
            if openai is None:
                raise RuntimeError("openai package not available. Please pip install openai.")
            # Environment variable names
            env_map = {
                "openai": "OPENAI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }
            env_name = env_map.get(self.provider, "OPENAI_API_KEY")
            key = api_key or os.getenv(env_name, "")
            if not key:
                raise RuntimeError(f"{env_name} not set and no --api_key provided.")
            openai.api_key = key

            # base_url
            if base_url:
                # New OpenAI client style
                openai.base_url = base_url  # type: ignore[attr-defined]

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai  # type: ignore[import]

                key = api_key or os.getenv("GEMINI_API_KEY", "")
                if not key:
                    raise RuntimeError("GEMINI_API_KEY not set and no --api_key provided.")
                genai.configure(api_key=key)
                self._gemini = genai
            except Exception as e:
                raise RuntimeError(f"Failed to import/configure google.generativeai: {e}")

        elif self.provider == "generic_http":
            # Generic HTTP endpoint:
            #   base_url: full endpoint URL
            #   api_key:  optional header "Authorization: Bearer ..."
            if not base_url:
                raise RuntimeError("generic_http requires --base_url to be provided.")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        timeout: float = 120.0,
    ) -> LLMResponse:
        """Call the underlying provider with a chat-completion-like interface."""
        if self.provider in ("openai", "deepseek", "openrouter"):
            return self._chat_openai_like(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif self.provider == "gemini":
            return self._chat_gemini(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        elif self.provider == "generic_http":
            return self._chat_generic_http(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ----------------
    # OpenAI-like
    # ----------------
    def _chat_openai_like(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> LLMResponse:
        assert openai is not None
        t0 = time.time()
        resp = openai.chat.completions.create(  # type: ignore[call-arg]
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        t1 = time.time()
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=t1 - t0,
        )

    # ----------------
    # Gemini
    # ----------------
    def _chat_gemini(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> LLMResponse:
        genai = self._gemini
        # Gemini uses a different content format.
        contents = []
        system_prompt = None
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                # We'll prepend this to the user text.
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt += "\n\n" + content
            else:
                contents.append({"role": role, "parts": [content]})

        if system_prompt:
            # Prepend the system prompt as a user message.
            contents.insert(0, {"role": "user", "parts": [system_prompt]})

        # Set generation config
        gen_config = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens

        gmodel = genai.GenerativeModel(model)
        t0 = time.time()
        resp = gmodel.generate_content(contents, generation_config=gen_config)
        t1 = time.time()

        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        return LLMResponse(
            content=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=t1 - t0,
        )

    # ----------------
    # Generic HTTP
    # ----------------
    def _chat_generic_http(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> LLMResponse:
        """A simple JSON POST interface for generic HTTP endpoints.

        Expects:
          POST base_url
          JSON payload with fields: model, messages, temperature, max_tokens
          Returns JSON with:
              choices[0].message.content
              usage.prompt_tokens
              usage.completion_tokens
        """
        if not self.base_url:
            raise RuntimeError("generic_http provider requires base_url.")
        url = self.base_url
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        t0 = time.time()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        t1 = time.time()

        choices = data.get("choices") or []
        content = ""
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content", "") or ""
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=t1 - t0,
        )


# ================================
# Prompt config
# ================================


@dataclass
class AgentPrompts:
    system: str
    user: str


@dataclass
class PromptConfig:
    solver: AgentPrompts
    critic: AgentPrompts
    judge: AgentPrompts

    @staticmethod
    def from_yaml(path: str) -> "PromptConfig":
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)

        def _block(name: str) -> AgentPrompts:
            blk = y[name]
            return AgentPrompts(system=blk["system"], user=blk["user"])

        return PromptConfig(
            solver=_block("solver"),
            critic=_block("critic"),
            judge=_block("judge"),
        )


# ================================
# Retriever (stub)
# ================================


def retrieve_context(question: str) -> str:
    """Very simple stub retriever.

    For now, we just return an empty string.
    In a real system, you could connect this to a vector DB.
    """
    return ""


# ================================
# Parsing helpers
# ================================


@dataclass
class SolverOutput:
    raw_text: str
    restatement: str
    reasoning: str
    final_answer_raw: str
    final_answer_value: str
    summary_for_cache: str
    metrics: Dict[str, Any]


def _strip_code_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```") and s.endswith("```"):
        inner = s[3:-3].strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        return inner
    return s


def _parse_sections(raw: str, headers: List[str]) -> Dict[str, str]:
    """Parse free-form text with headings like "Restatement:".

    Returns a dict: heading -> text (without the heading).
    """
    text = raw.replace("\r\n", "\n")
    result: Dict[str, str] = {h: "" for h in headers}
    current = None
    lines: List[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        # Detect new section
        matched_header = None
        for h in headers:
            if stripped.lower().startswith(h.lower() + ":"):
                matched_header = h
                break
        if matched_header is not None:
            # Save previous section
            if current is not None:
                result[current] = "\n".join(lines).strip()
            current = matched_header
            lines = [stripped[len(matched_header) + 1 :].lstrip()]
        else:
            lines.append(line)

    # Save last
    if current is not None:
        result[current] = "\n".join(lines).strip()

    return result


def parse_solver_output(text: str, metrics: Dict[str, Any]) -> SolverOutput:
    """Parse solver output.

    Supports two formats:
    1) Preferred: a JSON object like
       {"Restatement": ..., "Reasoning": ..., "Final answer": ..., "Summary_for_cache": ...}
    2) Fallback: the older section format with headers "Restatement:", "Reasoning:", etc.
    """
    raw = text or ""
    data = None

    # First, try to parse as JSON (strip possible code fences).
    candidate = raw.strip()
    # Remove surrounding markdown fences if present
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.strip("`").strip()
        # In case there's a language tag like ```json
        if "\n" in candidate:
            first_line, rest = candidate.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                candidate = rest.strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            data = obj
    except Exception:
        data = None

    if data is not None:
        restatement = str(data.get("Restatement", "") or "")
        reasoning = str(data.get("Reasoning", "") or "")
        # Support both "Final answer" and "Final_answer" keys in solver JSON
        final_raw = str(
            data.get("Final answer", data.get("Final_answer", "")) or ""
        )
        summary = str(data.get("Summary_for_cache", "") or "") or "NONE"
    else:
        # Fallback to section parsing with textual headers.
        sections = _parse_sections(
            raw,
            [
                "Restatement",
                "Reasoning",
                "Final answer",
                "Final_answer",
                "Summary_for_cache",
            ],
        )
        restatement = sections.get("Restatement", "")
        reasoning = sections.get("Reasoning", "")
        # Fallback: accept either header name
        final_raw = (
            sections.get("Final answer", "")
            or sections.get("Final_answer", "")
            or ""
        )
        summary = sections.get("Summary_for_cache", "") or "NONE"

    # numeric fallback for value
    value = extract_last_number(final_raw) or final_raw.strip()

    return SolverOutput(
        raw_text=raw,
        restatement=restatement,
        reasoning=reasoning,
        final_answer_raw=final_raw,
        final_answer_value=value,
        summary_for_cache=summary,
        metrics=metrics,
    )


@dataclass
class CriticOutput:
    raw_text: str
    verdict: str
    issues: str
    suggestions: str
    corrected_summary: str
    metrics: Dict[str, Any]


def parse_critic_output(text: str, metrics: Dict[str, Any]) -> CriticOutput:
    """Parse critic output.

    Preferred format: JSON object with keys
        Critic_verdict, Critic_issues, Critic_suggestions, Corrected_summary_for_cache
    Fallback: older section format with the same field names followed by colons.
    """
    raw = text or ""
    data = None

    candidate = raw.strip()
    if candidate.startswith("```") and candidate.endswith("```"):

        candidate = candidate.strip("`").strip()
        if "\n" in candidate:
            first_line, rest = candidate.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                candidate = rest.strip()

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            data = obj
    except Exception:
        data = None

    if data is not None:
        verdict = str(data.get("Critic_verdict", "") or "").upper()
        issues = str(data.get("Critic_issues", "") or "")
        suggestions = str(data.get("Critic_suggestions", "") or "")
        corrected = str(data.get("Corrected_summary_for_cache", "") or "") or "NONE"
    else:
        sections = _parse_sections(
            raw,
            [
                "Critic_verdict",
                "Critic_issues",
                "Critic_suggestions",
                "Corrected_summary_for_cache",
            ],
        )
        verdict = sections.get("Critic_verdict", "").strip().upper()
        issues = sections.get("Critic_issues", "")
        suggestions = sections.get("Critic_suggestions", "")
        corrected = sections.get("Corrected_summary_for_cache", "") or "NONE"

    return CriticOutput(
        raw_text=raw,
        verdict=verdict,
        issues=issues,
        suggestions=suggestions,
        corrected_summary=corrected,
        metrics=metrics,
    )


@dataclass
class JudgeOutput:
    raw_text: str
    label: str
    reason: str
    metrics: Dict[str, Any]


def parse_judge_output(text: str, metrics: Dict[str, Any]) -> JudgeOutput:
    """Parse judge output.

    Preferred format: JSON object with keys Judge_label, Judge_reason.
    Fallback: older section format with the same field names followed by colons.
    """
    raw = text or ""
    data = None

    candidate = raw.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.strip("`").strip()
        if "\n" in candidate:
            first_line, rest = candidate.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                candidate = rest.strip()

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            data = obj
    except Exception:
        data = None

    if data is not None:
        label = str(data.get("Judge_label", "") or "").upper()
        reason = str(data.get("Judge_reason", "") or "")
    else:
        sections = _parse_sections(raw, ["Judge_label", "Judge_reason"])
        label = sections.get("Judge_label", "").strip().upper()
        reason = sections.get("Judge_reason", "")

    return JudgeOutput(
        raw_text=raw,
        label=label,
        reason=reason,
        metrics=metrics,
    )


# ================================
# Agents
# ================================


class SolverAgent:
    def __init__(self, client: LLMClient, prompts: AgentPrompts, model: str, temperature: float = 0.2):
        self.client = client
        self.prompts = prompts
        self.model = model
        self.temperature = temperature

    def solve(
        self,
        question: str,
        context: str = "",
        max_tokens: Optional[int] = None,
    ) -> SolverOutput:
        """Run solver with the given question and optional retrieved context."""
        user_msg = self.prompts.user.format(QUESTION=question, CONTEXT=context)
        messages = [
            {"role": "system", "content": self.prompts.system},
            {"role": "user", "content": user_msg},
        ]
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        metrics = {
            "role": "solver",
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_s": resp.latency_s,
        }
        return parse_solver_output(resp.content, metrics)


class CriticAgent:
    def __init__(self, client: LLMClient, prompts: AgentPrompts, model: str, temperature: float = 0.2):
        self.client = client
        self.prompts = prompts
        self.model = model
        self.temperature = temperature

    def critique(
        self,
        question: str,
        solver_output: SolverOutput,
        max_tokens: Optional[int] = None,
    ) -> CriticOutput:
        """Run critic on the question and a given solver_output."""
        user_msg = self.prompts.user.format(QUESTION=question, SOLVER_OUTPUT_JSON=solver_output.raw_text)
        messages = [
            {"role": "system", "content": self.prompts.system},
            {"role": "user", "content": user_msg},
        ]
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        metrics = {
            "role": "critic",
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_s": resp.latency_s,
        }
        return parse_critic_output(resp.content, metrics)


class JudgeAgent:
    def __init__(self, client: LLMClient, prompts: AgentPrompts, model: str, temperature: float = 0.2):
        self.client = client
        self.prompts = prompts
        self.model = model
        self.temperature = temperature

    def judge(
        self,
        question: str,
        solver_output: SolverOutput,
        critic_output: CriticOutput,
        max_tokens: Optional[int] = None,
    ) -> JudgeOutput:
        """Run judge on the question, solver_output, and critic_output."""
        user_msg = self.prompts.user.format(
            QUESTION=question,
            SOLVER_OUTPUT_JSON=solver_output.raw_text,
            CRITIC_OUTPUT_JSON=critic_output.raw_text,
        )
        messages = [
            {"role": "system", "content": self.prompts.system},
            {"role": "user", "content": user_msg},
        ]
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        metrics = {
            "role": "judge",
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_s": resp.latency_s,
        }
        return parse_judge_output(resp.content, metrics)


# ================================
# Example result container
# ================================


@dataclass
class ExampleResult:
    qid: Any
    question: str
    gold: str
    pred_answer_raw: str
    pred_answer_value: str
    correct_numeric: int
    judge_label: str
    judge_reason: str
    agent_calls: int
    prompt_tokens: int
    completion_tokens: int
    latency_sum_s: float
    solver_output: str
    critic_output: str
    judge_output: str


def run_baseline_for_example(
    solver: SolverAgent,
    critic: CriticAgent,
    judge: JudgeAgent,
    question: str,
    gold: str,
    qid: Any = None,
    use_retriever: bool = True,
) -> ExampleResult:
    """Run the full solver->critic->judge pipeline for a single example."""
    per_metrics: List[Dict[str, Any]] = []
    agent_calls = 0

    # Retrieve context if enabled
    context = retrieve_context(question) if use_retriever else ""

    # 1) Solver
    solver_out = solver.solve(question=question, context=context)
    per_metrics.append(solver_out.metrics)
    agent_calls += 1

    # 2) Critic
    critic_out = critic.critique(question=question, solver_output=solver_out)
    per_metrics.append(critic_out.metrics)
    agent_calls += 1

    # 3) Judge
    judge_out = judge.judge(question=question, solver_output=solver_out, critic_output=critic_out)
    per_metrics.append(judge_out.metrics)
    agent_calls += 1

    # Aggregate metrics
    p_total, c_total = sum_tokens(per_metrics)
    latency_sum = sum(m.get("latency_s", 0.0) for m in per_metrics)

    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(solver_out.final_answer_value)
    correct_numeric = int(gold_norm == pred_norm)

    return ExampleResult(
        qid=qid,
        question=question,
        gold=gold,
        pred_answer_raw=solver_out.final_answer_raw,
        pred_answer_value=solver_out.final_answer_value,
        correct_numeric=correct_numeric,
        judge_label=judge_out.label,
        judge_reason=judge_out.reason,
        agent_calls=agent_calls,
        prompt_tokens=p_total,
        completion_tokens=c_total,
        latency_sum_s=latency_sum,
        solver_output=solver_out.raw_text,
        critic_output=critic_out.raw_text,
        judge_output=judge_out.raw_text,
    )


# ================================
# Main
# ================================


def main():
    parser = argparse.ArgumentParser(description="Baseline multi-agent math solver (API version).")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. gpt-4o-mini, gemini-1.5-pro")
    parser.add_argument("--data", type=str, required=True, help="Path to JSON dataset")
    parser.add_argument("--prompts", type=str, default="prompts.yaml", help="Path to prompts.yaml")
    parser.add_argument("--out_jsonl", type=str, default="baseline_results.jsonl", help="Output JSONL file path")
    parser.add_argument("--use_retriever", action="store_true", help="Enable simple retriever context")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider: openai, deepseek, openrouter, gemini, generic_http",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional API key (otherwise read from environment variables).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Optional custom base URL (for OpenAI-compatible or generic_http).",
    )
    args = parser.parse_args()

    # Load dataset
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load prompts
    prompt_cfg = PromptConfig.from_yaml(args.prompts)

    # LLM client
    client = LLMClient(provider=args.provider, api_key=args.api_key, base_url=args.base_url)

    # Agents (for baseline we use the same model for all three agents)
    solver_agent = SolverAgent(client, prompt_cfg.solver, model=args.model, temperature=args.temperature)
    critic_agent = CriticAgent(client, prompt_cfg.critic, model=args.model, temperature=args.temperature)
    judge_agent = JudgeAgent(client, prompt_cfg.judge, model=args.model, temperature=args.temperature)

    results: List[ExampleResult] = []

    for idx, ex in enumerate(data):
        qid = ex.get("id", idx)
        question = ex.get("question", "")
        gold = ex.get("answer", "")

        print(f"[{idx+1}/{len(data)}] Running baseline on id={qid}...")
        ex_res = run_baseline_for_example(
            solver=solver_agent,
            critic=critic_agent,
            judge=judge_agent,
            question=question,
            gold=gold,
            qid=qid,
            use_retriever=args.use_retriever,
        )
        results.append(ex_res)

    # Save JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as fj:
        for r in results:
            row = {
                "id": r.qid,
                "question": r.question,
                "gold": r.gold,
                "prediction_raw": r.pred_answer_raw,
                "prediction_value": r.pred_answer_value,
                "correct_numeric": r.correct_numeric,
                "judge_label": r.judge_label,
                "judge_reason": r.judge_reason,
                "agent_calls": r.agent_calls,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "latency_wall_s_sum": r.latency_sum_s,
                "solver_output": r.solver_output,
                "critic_output": r.critic_output,
                "judge_output": r.judge_output,
            }
            fj.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print detailed summary (similar style to advanced solver)
    if results:
        total = len(results)
        total_correct = sum(r.correct_numeric for r in results)
        acc = (total_correct / total * 100.0) if total > 0 else 0.0

        # Agent calls / latency
        total_calls = sum(r.agent_calls for r in results)
        avg_calls = total_calls / total if total > 0 else 0.0

        total_lat = sum(r.latency_sum_s for r in results)
        avg_lat = total_lat / total if total > 0 else 0.0

        # Token statistics
        total_prompt_tokens = sum(r.prompt_tokens for r in results)
        total_completion_tokens = sum(r.completion_tokens for r in results)
        total_tokens = total_prompt_tokens + total_completion_tokens
        avg_prompt_tokens = total_prompt_tokens / total if total > 0 else 0.0
        avg_completion_tokens = total_completion_tokens / total if total > 0 else 0.0
        avg_tokens = total_tokens / total if total > 0 else 0.0

        # Judge label distribution
        judge_counts: Dict[str, int] = {}
        for r in results:
            lbl = (r.judge_label or "").upper()
            if not lbl:
                lbl = "UNKNOWN"
            judge_counts[lbl] = judge_counts.get(lbl, 0) + 1

        print("\n===== BASELINE SUMMARY =====")
        print(f"Model                       : {args.model}")
        print(f"Total questions             : {total}")
        print(f"Correct (numeric exact)     : {total_correct}")
        print(f"Accuracy                    : {acc:.1f}%")
        print(f"Total agent calls           : {total_calls}")
        print(f"Avg agent calls / question  : {avg_calls:.2f}")
        print(f"Total latency (s, sum over examples) : {total_lat:.2f}")
        print(f"Avg latency / question (s)           : {avg_lat:.2f}")
        print(f"Total prompt tokens         : {total_prompt_tokens}")
        print(f"Total completion tokens     : {total_completion_tokens}")
        print(f"Total tokens                : {total_tokens}")
        print(f"Avg prompt tokens / question    : {avg_prompt_tokens:.1f}")
        print(f"Avg completion tokens / question: {avg_completion_tokens:.1f}")
        print(f"Avg total tokens / question     : {avg_tokens:.1f}")
        print(f"Judge label counts          : {judge_counts}")
        print(f"Results JSONL written to    : {args.out_jsonl}")
        print("======================================")
    else:
        print("No data examples found.")

if __name__ == "__main__":
    main()
