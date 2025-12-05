"""Question agent for Phase 3 - clarifying questions."""

from strands import Agent, tool


def create_question_agent(model_id: str = "amazon.nova-2-lite-v1:0") -> Agent:
    """Create agent for asking clarifying questions."""

    questions_asked = []

    @tool
    def ask_user(question: str) -> str:
        """Ask a clarifying question."""
        questions_asked.append(question)
        print(f"\n❓ [{len(questions_asked)}] {question}")
        return input("Your answer: ")

    @tool
    def done_asking() -> str:
        """Signal completion of question phase."""
        return "Proceeding to sparring phase."

    return Agent(
        name="QuestionAgent",
        model=model_id,
        system_prompt="""Ask clarifying questions about the architecture.

Focus on:
- Scale and traffic expectations
- Security and compliance requirements
- Reliability and availability needs
- Key constraints

Guidelines:
- One question at a time
- If answer is "no" or "none", move on
- Call done_asking when ready to proceed""",
        tools=[ask_user, done_asking],
    )


def run_questions(agent: Agent, req_summary: str, arch_summary: str) -> str:
    """Execute question phase."""
    result = agent(
        f"""Review summaries and ask clarifying questions:

REQUIREMENTS:
{req_summary}

ARCHITECTURE:
{arch_summary}

Ask questions to understand gaps. Call done_asking when ready."""
    )
    return str(result)
