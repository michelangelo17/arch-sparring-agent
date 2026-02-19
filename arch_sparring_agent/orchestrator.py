"""Orchestrates the 5-phase architecture review process."""

import logging
import re
from collections.abc import Callable

from .agents.architecture_agent import create_architecture_agent
from .agents.ci_agents import (
    create_ci_question_agent,
    create_ci_review_agent,
    create_ci_sparring_agent,
    generate_ci_review,
    run_ci_questions,
    run_ci_sparring,
)
from .agents.question_agent import create_question_agent, run_questions
from .agents.requirements_agent import create_requirements_agent
from .agents.review_agent import create_review_agent, generate_review
from .agents.sparring_agent import create_sparring_agent, run_sparring
from .config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_LEVEL,
    create_model,
)
from .context_condenser import (
    extract_architecture_findings,
    extract_phase_findings,
    extract_requirements,
)
from .infra import SharedConfig

logger = logging.getLogger(__name__)

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

    Infrastructure (Gateway, Policy Engine, Cedar policies) must already
    exist — created via ``arch-review deploy``.
    """

    def __init__(
        self,
        documents_dir: str,
        templates_dir: str,
        diagrams_dir: str,
        shared_config: SharedConfig,
        model_name: str = DEFAULT_MODEL,
        ci_mode: bool = False,
        source_dir: str | None = None,
        output_fn: Callable[[str], None] | None = None,
        reasoning_level: str = DEFAULT_REASONING_LEVEL,
    ):
        self.documents_dir = documents_dir
        self.templates_dir = templates_dir
        self.diagrams_dir = diagrams_dir
        self.source_dir = source_dir
        self.region = shared_config.region
        self.ci_mode = ci_mode
        self.output_fn = output_fn
        self.model_name = model_name
        self.policy_engine_id = shared_config.policy_engine_id

        logger.info("Using Policy Engine: %s", self.policy_engine_id)

        # Models: reasoning enabled for analysis-heavy agents, off for extraction/summarization
        standard_model = create_model(model_name, reasoning=False)
        if reasoning_level == "off":
            reasoning_model = standard_model
        else:
            reasoning_model = create_model(
                model_name, reasoning=True, reasoning_level=reasoning_level
            )

        kb_id = shared_config.knowledge_base_id
        kb_region = shared_config.region if kb_id else None

        self.requirements_agent = create_requirements_agent(documents_dir, standard_model)
        self.architecture_agent = create_architecture_agent(
            templates_dir,
            diagrams_dir,
            reasoning_model,
            source_dir=source_dir,
            knowledge_base_id=kb_id,
            region=kb_region,
        )

        if ci_mode:
            self.question_agent = create_ci_question_agent(standard_model)
            self.sparring_agent = create_ci_sparring_agent(standard_model)
            self.review_agent = create_ci_review_agent(reasoning_model)
        else:
            self.question_agent = create_question_agent(
                standard_model,
                templates_dir=templates_dir,
                source_dir=source_dir,
            )
            self.sparring_agent = create_sparring_agent(standard_model)
            self.review_agent = create_review_agent(
                reasoning_model,
                knowledge_base_id=kb_id,
                region=kb_region,
            )

        self.standard_model = standard_model
        self.captured_output: list[str] = []

    @staticmethod
    def _strip_redacted(text: str) -> str:
        """Remove [REDACTED] reasoning traces from model output."""
        # Remove lines that are just [REDACTED] (with optional surrounding whitespace/dots)
        cleaned = re.sub(r"^[.\s]*\[REDACTED\][.\s]*$", "", text, flags=re.MULTILINE)
        # Collapse runs of 3+ blank lines into 2
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _capture(self, content: str) -> None:
        """Capture output for session export and optionally emit it."""
        content = self._strip_redacted(content)
        self.captured_output.append(content)
        if self.output_fn:
            self.output_fn(content)

    def _verify_against_defaults(self, arch_findings: str) -> str:
        """Filter false positives by verifying 'Features Not Found' against AWS service defaults.

        Uses a focused model call whose only job is to check whether flagged
        items are actually provided by the service by default.
        """
        if "Features Not Found" not in arch_findings:
            return arch_findings

        from strands import Agent

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
        except Exception:
            logger.warning("Service-defaults verification failed, using unverified findings")
            return arch_findings

    def run_review(self) -> dict:
        """Execute the 5-phase review process."""
        self.captured_output = []

        mode_label = "CI/CD MODE" if self.ci_mode else "INTERACTIVE MODE"
        self._capture("=" * 60)
        self._capture(f"ARCHITECTURE REVIEW SESSION ({mode_label})")
        self._capture("=" * 60 + "\n")

        # Phase 1: Requirements
        self._capture("## Phase 1: Requirements Analysis\n")
        if self.ci_mode:
            req_prompt = "Analyze documents. List key requirements in bullet points. Max 20 lines."
        else:
            req_prompt = """
            Analyze all documents. Summarize requirements and constraints. Keep it under 600 words.
            """
        req_result = self.requirements_agent(req_prompt)
        req_summary = str(req_result)
        self._capture(req_summary)

        # Extract structured requirements for downstream phases
        req_findings = extract_requirements(req_summary, self.standard_model)

        # Phase 2: Architecture
        self._capture("\n## Phase 2: Architecture Analysis\n")
        if self.ci_mode:
            arch_prompt = f"""Analyze templates/diagrams/source code.

REQUIREMENTS TO VERIFY:
{req_findings}

Tasks:
1. List infrastructure components
2. For each requirement above, search source code to verify it's implemented
3. Output a "Features Verified" section listing what you found evidence for

Format:
### Components
- Component list

### Features Verified
- Feature: [evidence found, e.g. "language caching via SUMMARY#${{language}} key in checkCache.ts"]

### Features Not Found
- Feature: [searched but no evidence]"""
        else:
            arch_prompt = f"""Analyze all templates, diagrams, and source code.

REQUIREMENTS:
{req_findings}

Summarize architecture, patterns, and verify which requirements have implementations."""
        arch_result = self.architecture_agent(arch_prompt)
        arch_summary = str(arch_result)
        self._capture(arch_summary)

        # Extract structured architecture findings for downstream phases
        arch_findings = extract_architecture_findings(arch_summary, self.standard_model)

        # Verify "Features Not Found" against AWS service defaults
        arch_findings = self._verify_against_defaults(arch_findings)

        # Phase 3: Questions/Gaps
        phase3_title = "Identified Gaps" if self.ci_mode else "Clarifying Questions"
        self._capture(f"\n## Phase 3: {phase3_title}\n")
        if self.ci_mode:
            qa_context = run_ci_questions(self.question_agent, arch_findings)
        else:
            qa_context = run_questions(self.question_agent, req_findings, arch_findings)
            self._capture(f"\n{qa_context}")

        # Extract structured Q&A findings
        qa_findings = extract_phase_findings(qa_context, "Q&A", self.standard_model)

        # Phase 4: Sparring/Challenges
        phase4_title = "Risk Analysis" if self.ci_mode else "Architecture Sparring"
        self._capture(f"\n## Phase 4: {phase4_title}\n")
        if self.ci_mode:
            sparring_context = run_ci_sparring(self.sparring_agent, qa_findings)
        else:
            sparring_context = run_sparring(
                self.sparring_agent, req_findings, arch_findings, qa_findings
            )
            self._capture(f"\n{sparring_context}")

        # Extract structured sparring findings
        sparring_findings = extract_phase_findings(
            sparring_context, "Sparring", self.standard_model
        )

        # Phase 5: Final Review
        self._capture("\n## Phase 5: Final Review\n")
        self._capture("=" * 60)
        if self.ci_mode:
            review_text = generate_ci_review(self.review_agent, qa_findings, sparring_findings)
        else:
            review_text = generate_review(
                self.review_agent, req_findings, arch_findings, qa_findings, sparring_findings
            )
        self._capture(review_text)
        self._capture("=" * 60)

        return {
            "review": review_text,
            "full_session": self.get_full_session(),
            "requirements_summary": req_summary,
            "requirements_findings": req_findings,
            "architecture_summary": arch_summary,
            "architecture_findings": arch_findings,
            "gaps": qa_context,
            "gaps_findings": qa_findings,
            "risks": sparring_context,
            "risks_findings": sparring_findings,
            "ci_mode": self.ci_mode,
            "agents_used": [
                "RequirementsAnalyst",
                "ArchitectureEvaluator",
                "QuestionAgent",
                "SparringAgent",
                "ReviewAgent",
            ],
        }

    def get_full_session(self) -> str:
        """Return captured session output for file export."""
        return "\n".join(self.captured_output)
