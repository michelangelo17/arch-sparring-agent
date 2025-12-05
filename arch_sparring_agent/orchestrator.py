from .agents.architecture_agent import create_architecture_agent
from .agents.moderator_agent import create_moderator_agent
from .agents.requirements_agent import create_requirements_agent
from .config import MODEL_ID, check_model_access, get_inference_profile_arn, setup_agentcore_memory


class ReviewOrchestrator:
    """Orchestrates the multi-agent architecture review process."""

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

        # Resolve inference profile ARN if needed (required for Nova 2 Lite)
        inference_profile_arn = get_inference_profile_arn(model_id)
        self.model_id = inference_profile_arn if inference_profile_arn else model_id

        # Verify model access (using original ID for check)
        if not check_model_access(model_id):
            raise RuntimeError(
                f"Model {model_id} not accessible. Please request access in Bedrock console."
            )

        # Set up shared memory config for all agents (avoid creating 3 times)
        self.memory_config = None
        self.memory_id = None
        if enable_memory:
            self.memory_config, self.memory_id = setup_agentcore_memory(region=region)
            if self.memory_config:
                print(f"✓ Shared AgentCore Memory enabled (ID: {self.memory_id})")

        # Initialize agents with shared memory config
        self.requirements_agent = create_requirements_agent(
            documents_dir,
            self.model_id,
            memory_config=self.memory_config,
        )
        self.architecture_agent = create_architecture_agent(
            templates_dir,
            diagrams_dir,
            self.model_id,
            memory_config=self.memory_config,
        )
        self.moderator_agent = create_moderator_agent(
            self.requirements_agent,
            self.architecture_agent,
            self.model_id,
            memory_config=self.memory_config,
        )

    def run_review(self, interactive: bool = True) -> dict:
        """Run the complete architecture review. Interactive mode is default."""
        print("Starting architecture review...")
        print("=" * 60)

        # The moderator agent handles everything
        # It will call the other agents as needed via its tools
        if interactive:
            review_prompt = """Conduct an interactive architecture review.

Steps:
1. Get requirements summary from RequirementsAnalyst
2. Get architecture summary from ArchitectureEvaluator
3. Identify 3-5 gaps or unclear areas
4. For each gap: call ask_user_question with ONE question, then STOP and wait
5. After each answer, ask the next question or produce final review

CRITICAL: After calling ask_user_question, STOP. Do not continue until you receive the answer.

When all questions are answered, produce the final review."""
        else:
            review_prompt = """Conduct a comprehensive architecture review:

1. Analyze all requirements documents
2. Analyze CloudFormation templates and architecture diagrams
3. Compare requirements against implementation
4. Identify gaps and risks
5. Provide prioritized recommendations

Produce a structured review document."""

        print("Moderator agent is coordinating the review...")
        if interactive:
            print("(Interactive mode: agent will ask clarifying questions)\n")

        result = self.moderator_agent(review_prompt)

        print("\n" + "=" * 60)
        print("Review complete!")

        return {
            "review": result,
            "agents_used": ["RequirementsAnalyst", "ArchitectureEvaluator", "ReviewModerator"],
        }
