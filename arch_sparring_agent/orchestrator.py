"""Orchestrates the 5-phase architecture review process."""

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
    MODEL_ID,
    check_model_access,
    get_inference_profile_arn,
    setup_architecture_review_policies,
)
from .context_condenser import (
    extract_architecture_findings,
    extract_phase_findings,
    extract_requirements,
)


class ReviewOrchestrator:
    """Orchestrates multi-agent architecture review."""

    def __init__(
        self,
        documents_dir: str,
        templates_dir: str,
        diagrams_dir: str,
        model_id: str = MODEL_ID,
        region: str = "eu-central-1",
        ci_mode: bool = False,
        source_dir: str | None = None,
        skip_policy_check: bool = False,
    ):
        self.documents_dir = documents_dir
        self.templates_dir = templates_dir
        self.diagrams_dir = diagrams_dir
        self.source_dir = source_dir
        self.region = region
        self.ci_mode = ci_mode

        inference_profile_arn = get_inference_profile_arn(model_id)
        self.model_id = inference_profile_arn or model_id

        if not check_model_access(model_id):
            raise RuntimeError(f"Model {model_id} not accessible.")

        print("\n" + "=" * 60)
        print("Setting up Policy Engine and Policies")
        print("=" * 60)
        self.policy_engine_id = setup_architecture_review_policies(region=region)
        if self.policy_engine_id:
            print(f"\n✓ Policy Engine ID: {self.policy_engine_id}")
        elif skip_policy_check:
            print(
                "\n⚠️  Policy Engine setup failed (--skip-policy-check enabled).\n"
                "   Agents will NOT have Cedar tool-access restrictions.\n"
                "   Running without security policy enforcement."
            )
        else:
            raise RuntimeError(
                "Policy Engine setup failed. Cedar policies could not be created.\n"
                "Agents cannot run without security policy enforcement.\n"
                "To bypass this check (development only), use --skip-policy-check."
            )
        print("=" * 60 + "\n")

        self.requirements_agent = create_requirements_agent(documents_dir, self.model_id)
        self.architecture_agent = create_architecture_agent(
            templates_dir, diagrams_dir, self.model_id, source_dir=source_dir
        )

        if ci_mode:
            self.question_agent = create_ci_question_agent(self.model_id)
            self.sparring_agent = create_ci_sparring_agent(self.model_id)
            self.review_agent = create_ci_review_agent(self.model_id)
        else:
            self.question_agent = create_question_agent(self.model_id)
            self.sparring_agent = create_sparring_agent(self.model_id)
            self.review_agent = create_review_agent(self.model_id)

        self.captured_output = []

    def _capture(self, content: str):
        """Capture and print output for session export."""
        self.captured_output.append(content)
        print(content)

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
        req_findings = extract_requirements(req_summary, self.model_id)

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
        arch_findings = extract_architecture_findings(arch_summary, self.model_id)

        # Phase 3: Questions/Gaps
        phase3_title = "Identified Gaps" if self.ci_mode else "Clarifying Questions"
        self._capture(f"\n## Phase 3: {phase3_title}\n")
        if self.ci_mode:
            qa_context = run_ci_questions(self.question_agent, req_findings, arch_findings)
        else:
            qa_context = run_questions(self.question_agent, req_findings, arch_findings)
            self._capture(f"\n{qa_context}")

        # Extract structured Q&A findings
        qa_findings = extract_phase_findings(qa_context, "Q&A", self.model_id)

        # Phase 4: Sparring/Challenges
        phase4_title = "Risk Analysis" if self.ci_mode else "Architecture Sparring"
        self._capture(f"\n## Phase 4: {phase4_title}\n")
        if self.ci_mode:
            sparring_context = run_ci_sparring(
                self.sparring_agent, req_findings, arch_findings, qa_findings
            )
        else:
            sparring_context = run_sparring(
                self.sparring_agent, req_findings, arch_findings, qa_findings
            )
            self._capture(f"\n{sparring_context}")

        # Extract structured sparring findings
        sparring_findings = extract_phase_findings(sparring_context, "Sparring", self.model_id)

        # Phase 5: Final Review
        self._capture("\n## Phase 5: Final Review\n")
        self._capture("=" * 60)
        if self.ci_mode:
            review_text = generate_ci_review(
                self.review_agent, req_findings, arch_findings, qa_findings, sparring_findings
            )
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
