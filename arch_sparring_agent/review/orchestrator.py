"""Orchestrates the 5-phase architecture review process."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from botocore.exceptions import BotoCoreError
from strands import Agent
from strands.models import BedrockModel

from ..agents import safe_invoke
from ..agents.architecture_agent import create_architecture_agent, run_architecture
from ..agents.question_agent import create_question_agent, run_questions
from ..agents.requirements_agent import create_requirements_agent
from ..agents.review_agent import create_review_agent, run_review
from ..agents.sparring_agent import create_sparring_agent, run_sparring
from ..config import (
    AGENT_ARCHITECTURE,
    AGENT_QUESTION,
    AGENT_REQUIREMENTS,
    AGENT_REVIEW,
    AGENT_SPARRING,
    DEFAULT_MODEL,
    DEFAULT_REASONING_LEVEL,
    create_model,
)
from ..exceptions import MODEL_ERRORS
from ..infra import SharedConfig
from .context_condenser import (
    extract_architecture_findings,
    extract_phase_findings,
    extract_requirements,
)
from .grounding import GuardrailsChecker, create_guardrails_checker

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Structured output from a completed review run."""

    review: str
    full_session: str
    requirements_summary: str
    requirements_findings: str
    architecture_summary: str
    architecture_findings: str
    qa_context: str
    qa_findings: str
    sparring_context: str
    sparring_findings: str
    agents_used: list[str] = field(default_factory=list)


_VERIFY_DEFAULTS_PROMPT = """You are an AWS service defaults expert. For each item listed under
"Features Not Found" below, determine whether the AWS service provides this
feature BY DEFAULT even without explicit CloudFormation configuration.

For each item:
- If the service DOES provide it by default, move it to "Features Verified
  (via service default)" with a one-line explanation.
- If the service does NOT provide it by default, keep it in "Features Not Found".

Preserve the "Components" and original "Features Verified" sections unchanged.
Only reclassify items from "Features Not Found" when you are confident the
service default covers the requirement. When uncertain, leave the item in
"Features Not Found".

Return the full updated findings with all three sections."""


class ReviewOrchestrator:
    """Orchestrates multi-agent architecture review.

    Accepts pre-built agents via the constructor for testability.
    Use the ``create()`` classmethod for the standard construction path.
    """

    def __init__(
        self,
        requirements_agent: Agent,
        architecture_agent: Agent,
        question_agent: Agent,
        sparring_agent: Agent,
        review_agent: Agent,
        standard_model: BedrockModel,
        output_fn: Callable[[str], None] | None = None,
        *,
        has_kb: bool = False,
        guardrails: GuardrailsChecker | None = None,
    ):
        self.requirements_agent = requirements_agent
        self.architecture_agent = architecture_agent
        self.question_agent = question_agent
        self.sparring_agent = sparring_agent
        self.review_agent = review_agent
        self.standard_model = standard_model
        self.output_fn = output_fn
        self._has_kb = has_kb
        self._guardrails = guardrails
        self.captured_output: list[str] = []

    @classmethod
    def create(
        cls,
        documents_dir: str,
        templates_dir: str,
        diagrams_dir: str,
        shared_config: SharedConfig,
        model_name: str = DEFAULT_MODEL,
        source_dir: str | None = None,
        output_fn: Callable[[str], None] | None = None,
        reasoning_level: str = DEFAULT_REASONING_LEVEL,
        profile: dict[str, Any] | None = None,
    ) -> ReviewOrchestrator:
        """Build all agents and return a configured orchestrator."""
        logger.info("Using Policy Engine: %s", shared_config.policy_engine_id)

        standard_model = create_model(model_name, reasoning=False)
        if reasoning_level == "off":
            reasoning_model = standard_model
        else:
            reasoning_model = create_model(
                model_name, reasoning=True, reasoning_level=reasoning_level
            )

        kb_id = shared_config.knowledge_base_id
        kb_region = shared_config.region if kb_id else None

        requirements_agent = create_requirements_agent(documents_dir, standard_model)
        architecture_agent = create_architecture_agent(
            templates_dir,
            diagrams_dir,
            reasoning_model,
            source_dir=source_dir,
            knowledge_base_id=kb_id,
            region=kb_region,
            profile=profile,
        )
        question_agent = create_question_agent(
            standard_model,
            templates_dir=templates_dir,
            source_dir=source_dir,
        )
        sparring_agent = create_sparring_agent(standard_model, profile=profile)
        review_agent = create_review_agent(
            reasoning_model,
            knowledge_base_id=kb_id,
            region=kb_region,
            profile=profile,
        )

        guardrails = create_guardrails_checker(
            shared_config.guardrail_id,
            shared_config.guardrail_version,
            region=shared_config.region,
        )
        if guardrails:
            logger.info(
                "Contextual grounding checks enabled (guardrail: %s)",
                guardrails.guardrail_id,
            )

        return cls(
            requirements_agent=requirements_agent,
            architecture_agent=architecture_agent,
            question_agent=question_agent,
            sparring_agent=sparring_agent,
            review_agent=review_agent,
            standard_model=standard_model,
            output_fn=output_fn,
            has_kb=bool(kb_id and kb_region),
            guardrails=guardrails,
        )

    @staticmethod
    def _strip_redacted(text: str) -> str:
        """Remove [REDACTED] reasoning traces from model output.

        Bedrock models with extended thinking enabled emit reasoning traces
        as [REDACTED] placeholders in the response text.
        """
        cleaned = re.sub(r"^[.\s]*\[REDACTED\][.\s]*$", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _capture(self, content: str) -> None:
        """Capture output for session export and optionally emit it."""
        content = self._strip_redacted(content)
        self.captured_output.append(content)
        if self.output_fn:
            self.output_fn(content)

    def _check_grounding(self, phase_name: str, source: str, content: str) -> None:
        """Run contextual grounding check on condenser output if guardrails are enabled."""
        if not self._guardrails:
            return

        query = f"Extract {phase_name} findings from architecture review analysis"
        result = self._guardrails.check_grounding(source, query, content)

        if result.passed:
            logger.info(
                "Grounding check passed for %s (score=%.2f)",
                phase_name,
                result.grounding_score,
            )
        else:
            logger.warning(
                "Grounding check FAILED for %s (score=%.2f): %s",
                phase_name,
                result.grounding_score,
                result.details,
            )

    def _verify_against_defaults(self, arch_findings: str) -> str:
        """Filter false positives by verifying 'Features Not Found' against AWS service defaults.

        Uses a focused model call whose only job is to check whether flagged
        items are actually provided by the service by default.
        """
        if "Features Not Found" not in arch_findings:
            return arch_findings

        not_found_idx = arch_findings.index("Features Not Found")
        after_header = arch_findings[not_found_idx + len("Features Not Found") :]
        after_header = after_header.lstrip(":# \n")
        if not after_header or after_header.lower().startswith("none"):
            logger.info("No gaps to verify — skipping service-defaults check")
            return arch_findings

        verifier = Agent(
            name="DefaultsVerifier",
            model=self.standard_model,
            callback_handler=None,
            system_prompt=_VERIFY_DEFAULTS_PROMPT,
            tools=[],
        )
        try:
            result = str(verifier(arch_findings))
            logger.info("Service-defaults verification complete")
            return result
        except (*MODEL_ERRORS, BotoCoreError):
            logger.warning("Service-defaults verification failed, using unverified findings")
            return arch_findings

    def run_review(self) -> ReviewResult:
        """Execute the 5-phase review process."""
        self.captured_output = []

        self._capture("=" * 60)
        self._capture("ARCHITECTURE REVIEW SESSION")
        self._capture("=" * 60 + "\n")

        # Phase 1: Requirements
        self._capture("## Phase 1: Requirements Analysis\n")
        req_summary = safe_invoke(
            self.requirements_agent,
            "Analyze all documents. Summarize requirements and constraints.",
        )
        self._capture(req_summary)

        req_findings = extract_requirements(req_summary, self.standard_model)
        self._check_grounding("requirements", req_summary, req_findings)

        # Phase 2: Architecture
        self._capture("\n## Phase 2: Architecture Analysis\n")
        arch_summary = run_architecture(self.architecture_agent, req_findings, has_kb=self._has_kb)
        self._capture(arch_summary)

        arch_findings = extract_architecture_findings(arch_summary, self.standard_model)
        self._check_grounding("architecture", arch_summary, arch_findings)
        arch_findings = self._verify_against_defaults(arch_findings)

        # Phase 3: Clarifying Questions
        self._capture("\n## Phase 3: Clarifying Questions\n")
        qa_context = run_questions(self.question_agent, arch_findings)
        self._capture(f"\n{qa_context}")

        qa_findings = extract_phase_findings(qa_context, "Q&A", self.standard_model)
        self._check_grounding("Q&A", qa_context, qa_findings)

        # Phase 4: Architecture Sparring
        self._capture("\n## Phase 4: Architecture Sparring\n")
        sparring_context = run_sparring(self.sparring_agent, arch_findings, qa_findings)
        self._capture(f"\n{sparring_context}")

        sparring_findings = extract_phase_findings(
            sparring_context, "Sparring", self.standard_model
        )
        self._check_grounding("sparring", sparring_context, sparring_findings)

        # Phase 5: Final Review
        self._capture("\n## Phase 5: Final Review\n")
        self._capture("=" * 60)
        review_text = run_review(
            self.review_agent, req_findings, arch_findings, qa_findings, sparring_findings
        )
        self._capture(review_text)
        self._capture("=" * 60)

        return ReviewResult(
            review=review_text,
            full_session=self.get_full_session(),
            requirements_summary=req_summary,
            requirements_findings=req_findings,
            architecture_summary=arch_summary,
            architecture_findings=arch_findings,
            qa_context=qa_context,
            qa_findings=qa_findings,
            sparring_context=sparring_context,
            sparring_findings=sparring_findings,
            agents_used=[
                AGENT_REQUIREMENTS,
                AGENT_ARCHITECTURE,
                AGENT_QUESTION,
                AGENT_SPARRING,
                AGENT_REVIEW,
            ],
        )

    def get_full_session(self) -> str:
        """Return captured session output for file export."""
        return "\n".join(self.captured_output)
