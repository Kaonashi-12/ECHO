"""Real cross-dataset prompt/completion streams for mask training."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True)
class TextEpisode:
    support_domain: str
    target_domain: str
    retain_domain: str
    support_prompts: list[str]
    support_completions: list[str]
    target_prompts: list[str]
    target_completions: list[str]
    retain_prompts: list[str]
    retain_completions: list[str]

    @property
    def support_texts(self) -> list[str]:
        return [
            prompt + completion
            for prompt, completion in zip(self.support_prompts, self.support_completions)
        ]

    @property
    def target_texts(self) -> list[str]:
        return [
            prompt + completion
            for prompt, completion in zip(self.target_prompts, self.target_completions)
        ]

    @property
    def retain_texts(self) -> list[str]:
        return [
            prompt + completion
            for prompt, completion in zip(self.retain_prompts, self.retain_completions)
        ]


@dataclass(frozen=True)
class PromptCompletionExample:
    prompt: str
    completion: str
    source_id: str = ""


class RealMathCrossDatasetStream:
    """Cross-dataset real text stream for formal mask experiments."""

    def __init__(
        self,
        domains: list[dict[str, Any]],
        retain_domains: list[dict[str, Any]] | None = None,
        seed: int = 0,
    ) -> None:
        self.rng = random.Random(seed)
        self.domains = {
            spec["name"]: _load_examples(spec)
            for spec in domains
        }
        self.retain_domains = {
            spec["name"]: _load_examples(spec)
            for spec in (retain_domains or [])
        }
        self.domain_names = [name for name, examples in self.domains.items() if examples]
        self.retain_names = [
            name for name, examples in self.retain_domains.items() if examples
        ]
        if len(self.domain_names) < 2:
            raise ValueError("RealMathCrossDatasetStream needs at least two non-empty domains")
        if retain_domains and not self.retain_names:
            raise ValueError("retain_domains were configured but no retain examples loaded")

    def sample_episode(
        self,
        support_size: int,
        target_size: int,
        retain_size: int,
    ) -> TextEpisode:
        support_domain = self.rng.choice(self.domain_names)
        target_domain = self.rng.choice(
            [name for name in self.domain_names if name != support_domain]
        )
        retain_domain = self.rng.choice(self.retain_names) if self.retain_names else target_domain
        support = self._sample_many(self.domains[support_domain], support_size)
        target = self._sample_many(self.domains[target_domain], target_size)
        retain_source = (
            self.retain_domains[retain_domain]
            if self.retain_names
            else self.domains[target_domain]
        )
        retain = self._sample_many(retain_source, retain_size)
        return TextEpisode(
            support_domain=support_domain,
            target_domain=target_domain,
            retain_domain=retain_domain,
            support_prompts=[example.prompt for example in support],
            support_completions=[example.completion for example in support],
            target_prompts=[example.prompt for example in target],
            target_completions=[example.completion for example in target],
            retain_prompts=[example.prompt for example in retain],
            retain_completions=[example.completion for example in retain],
        )

    def _sample_many(
        self,
        examples: list[PromptCompletionExample],
        size: int,
    ) -> list[PromptCompletionExample]:
        return [self.rng.choice(examples) for _ in range(size)]


def _load_examples(spec: dict[str, Any]) -> list[PromptCompletionExample]:
    from datasets import load_dataset

    path = spec["path"]
    name = spec.get("config")
    split = spec.get("split", "train")
    dataset = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    max_examples = spec.get("max_examples")
    kind = spec.get("kind", spec["name"])
    completion_mode = spec.get("completion_mode", "final_answer")
    examples: list[PromptCompletionExample] = []
    for index, row in enumerate(dataset):
        if max_examples is not None and len(examples) >= int(max_examples):
            break
        try:
            example = _format_example(
                kind,
                row,
                f"{spec['name']}:{index}",
                completion_mode=completion_mode,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if example.prompt.strip() and example.completion.strip():
            examples.append(example)
    if not examples:
        raise ValueError(f"No examples loaded for domain {spec['name']!r} from {path}")
    return examples


def _format_example(
    kind: str,
    row: dict[str, Any],
    source_id: str,
    completion_mode: str = "final_answer",
) -> PromptCompletionExample:
    kind = kind.lower()
    if kind == "gsm8k":
        return _qa(row["question"], _math_answer(row["answer"], completion_mode), source_id)
    if kind == "svamp":
        question = row.get("question_concat") or f"{row['Body']} {row['Question']}"
        answer = _equation_answer(
            equation=row.get("Equation"),
            final_answer=row["Answer"],
            completion_mode=completion_mode,
        )
        return _qa(question, answer, source_id)
    if kind == "asdiv":
        question = f"{row['body']} {row['question']}"
        answer = _equation_answer(
            equation=row.get("formula"),
            final_answer=row["answer"],
            completion_mode=completion_mode,
        )
        return _qa(question, answer, source_id)
    if kind in {"mawps", "multiarith"}:
        question = row.get("question") or row.get("Question")
        answer = row.get("answer", row.get("Answer", row.get("final_ans")))
        return _qa(str(question), _math_answer(answer, completion_mode), source_id)
    if kind in {"arc", "ai2_arc", "arc_challenge", "arc_easy"}:
        return _multiple_choice(row["question"], row["choices"], row["answerKey"], source_id)
    if kind in {"openbookqa", "openbook"}:
        return _multiple_choice(row["question_stem"], row["choices"], row["answerKey"], source_id)
    if kind in {"commonsenseqa", "commonsense_qa", "csqa"}:
        return _multiple_choice(row["question"], row["choices"], row["answerKey"], source_id)
    raise ValueError(f"Unsupported real dataset kind: {kind!r}")


def _qa(question: str, answer: str, source_id: str) -> PromptCompletionExample:
    question = " ".join(str(question).strip().split())
    answer = str(answer).strip()
    return PromptCompletionExample(
        prompt=f"Question: {question}\nAnswer:",
        completion=f" {answer}",
        source_id=source_id,
    )


def _math_answer(answer: Any, completion_mode: str) -> str:
    if completion_mode in {"final", "final_answer", "answer_only"}:
        return _final_answer(answer)
    if completion_mode in {"full", "full_solution", "solution"}:
        return _normalize_solution(answer)
    raise ValueError(f"Unsupported completion_mode={completion_mode!r}")


def _equation_answer(
    equation: Any,
    final_answer: Any,
    completion_mode: str,
) -> str:
    final = _final_answer(final_answer)
    if completion_mode in {"final", "final_answer", "answer_only"}:
        return final
    if completion_mode not in {"full", "full_solution", "solution"}:
        raise ValueError(f"Unsupported completion_mode={completion_mode!r}")
    equation_text = _normalize_solution(equation)
    if not equation_text:
        return final
    return f"{equation_text}\nFinal answer: {final}"


def _normalize_solution(answer: Any) -> str:
    lines = [
        " ".join(line.strip().split())
        for line in str(answer).strip().splitlines()
        if line.strip()
    ]
    return "\n".join(lines)


def _final_answer(answer: Any) -> str:
    text = str(answer).strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    return " ".join(text.split())


def _multiple_choice(
    question: str,
    choices: dict[str, Any],
    answer_key: str,
    source_id: str,
) -> PromptCompletionExample:
    labels = [str(label) for label in choices["label"]]
    texts = [str(text) for text in choices["text"]]
    answer_label = str(answer_key)
    answer_text = ""
    if answer_label in labels:
        answer_text = texts[labels.index(answer_label)]
    elif answer_label.isdigit():
        index = int(answer_label) - 1
        if 0 <= index < len(texts):
            answer_label = labels[index]
            answer_text = texts[index]
    choice_lines = "\n".join(
        f"{label}. {' '.join(text.strip().split())}"
        for label, text in zip(labels, texts)
    )
    question = " ".join(str(question).strip().split())
    completion = f" {answer_label}"
    if answer_text:
        completion += f". {answer_text}"
    return PromptCompletionExample(
        prompt=f"Question: {question}\nChoices:\n{choice_lines}\nAnswer:",
        completion=completion,
        source_id=source_id,
    )
