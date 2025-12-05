"""Orchestrates the 5-phase architecture review process."""

from .agents.architecture_agent import create_architecture_agent
from .agents.question_agent import create_question_agent, run_questions
from .agents.requirements_agent import create_requirements_agent
from .agents.review_agent import create_review_agent, generate_review
from .agents.sparring_agent import create_sparring_agent, run_sparring
from .config import MODEL_ID, check_model_access, get_inference_profile_arn, setup_agentcore_memory


class ReviewOrchestrator:
    """Orchestrates multi-agent architecture review."""

    def __init__(
        self,
        documents_dir: str,
        templates_dir: str,
        diagrams_dir: str,
        model_id: str = MODEL_ID,
        enable_memory: bool = True,
        region: str = "eu-central-1",
    ):
        self.documents_dir = documents_dir
        self.templates_dir = templates_dir
        self.diagrams_dir = diagrams_dir
        self.region = region

        # Resolve inference profile ARN for Nova 2 Lite
        inference_profile_arn = get_inference_profile_arn(model_id)
        self.model_id = inference_profile_arn if inference_profile_arn else model_id

        if not check_model_access(model_id):
            raise RuntimeError(f"Model {model_id} not accessible.")

        # Project-scoped memory (derived from working directory)
        self.memory_config = None
        if enable_memory:
            import re
            from pathlib import Path

            # Sanitize: letters, numbers, underscores only; max 48 chars
            project_name = Path.cwd().name
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", project_name)[:40]
            memory_name = f"Review_{safe_name}"
            self.memory_config, memory_id = setup_agentcore_memory(
                region=region, memory_name=memory_name
            )
            if self.memory_config:
                print(f"✓ Memory: {memory_name}")

        # Initialize agents
        self.requirements_agent = create_requirements_agent(
            documents_dir, self.model_id, memory_config=self.memory_config
        )
        self.architecture_agent = create_architecture_agent(
            templates_dir, diagrams_dir, self.model_id, memory_config=self.memory_config
        )
        self.question_agent = create_question_agent(self.model_id)
        self.sparring_agent = create_sparring_agent(self.model_id)
        self.review_agent = create_review_agent(self.model_id)

        self.captured_output = []

    def _capture(self, content: str):
        """Capture and print output for session export."""
        self.captured_output.append(content)
        print(content)

    def run_review(self, interactive: bool = True) -> dict:
        """Execute the 5-phase review process."""
        self.captured_output = []

        self._capture("=" * 60)
        self._capture("ARCHITECTURE REVIEW SESSION")
        self._capture("=" * 60 + "\n")

        # Phase 1: Requirements
        self._capture("## Phase 1: Requirements Analysis\n")
        req_result = self.requirements_agent(
            "Analyze all documents. Summarize requirements and constraints."
        )
        req_summary = str(req_result)
        self._capture(req_summary)

        # Phase 2: Architecture
        self._capture("\n## Phase 2: Architecture Analysis\n")
        arch_result = self.architecture_agent(
            "Analyze all templates and diagrams. Summarize architecture and patterns."
        )
        arch_summary = str(arch_result)
        self._capture(arch_summary)

        qa_context = ""
        sparring_context = ""

        if interactive:
            # Phase 3: Questions
            self._capture("\n## Phase 3: Clarifying Questions\n")
            qa_context = run_questions(self.question_agent, req_summary, arch_summary)
            self._capture(f"\n{qa_context}")

            # Phase 4: Sparring
            self._capture("\n## Phase 4: Architecture Sparring\n")
            sparring_context = run_sparring(
                self.sparring_agent, req_summary, arch_summary, qa_context
            )
            self._capture(f"\n{sparring_context}")

        # Phase 5: Final Review
        self._capture("\n## Phase 5: Final Review\n")
        self._capture("=" * 60)
        review_text = generate_review(
            self.review_agent, req_summary, arch_summary, qa_context, sparring_context
        )
        self._capture(review_text)
        self._capture("=" * 60)

        return {
            "review": review_text,
            "full_session": self.get_full_session(),
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
