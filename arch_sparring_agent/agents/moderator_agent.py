from strands import Agent, tool

from ..config import create_session_manager


def create_moderator_agent(
    requirements_agent: Agent,
    architecture_agent: Agent,
    model_id: str = "amazon.nova-2-lite-v1:0",
    memory_config=None,
) -> Agent:
    """
    Create the Moderator/Review Agent using Agents-as-Tools pattern.

    Coordinates requirements and architecture agents, asks clarifying questions.
    """

    @tool
    def get_requirements_analysis(task: str) -> str:
        """Delegate a task to the Requirements Analyst agent."""
        result = requirements_agent(task)
        return str(result)

    @tool
    def get_architecture_analysis(task: str) -> str:
        """Delegate a task to the Architecture Evaluator agent."""
        result = architecture_agent(task)
        return str(result)

    @tool
    def ask_user_question(question: str) -> str:
        """Ask the user a clarifying question and get their response."""
        print(f"\n❓ {question}")
        response = input("Your answer: ")
        return response

    # Set up session manager if memory config provided
    session_manager = None
    if memory_config:
        session_manager = create_session_manager(memory_config)

    agent = Agent(
        name="ReviewModerator",
        model=model_id,
        system_prompt="""Architecture review moderator.

Tools:
- get_requirements_analysis: Get requirements from RequirementsAnalyst
- get_architecture_analysis: Get architecture from ArchitectureEvaluator
- ask_user_question: Ask user ONE question and wait for response

CRITICAL RULES:
1. Call ask_user_question with exactly ONE question
2. STOP after calling ask_user_question - wait for the response
3. After receiving an answer, you may ask another question or proceed
4. Only write the final review when you have enough information

Process:
1. Get requirements and architecture summaries
2. Identify 3-5 key gaps or unclear areas
3. Ask about each gap ONE AT A TIME using ask_user_question
4. After all questions answered, produce final review

Final review format:
- Executive summary (2-3 sentences)
- Gaps identified (bullet points)
- Top 3 risks with severity
- Top 3 recommendations with priority""",
        tools=[get_requirements_analysis, get_architecture_analysis, ask_user_question],
        session_manager=session_manager,
    )

    return agent
