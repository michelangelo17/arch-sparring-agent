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
        system_prompt="""Ask clarifying questions about CONFIRMED gaps only.

RULES:
- Only ask about items in "Features Not Found"
- Do NOT question features listed in "Features Verified"
- Focus on: scale, security, compliance, reliability
- One question at a time
- If answer is "no" or "none", move on
- Call done_asking when ready to proceed""",
        tools=[ask_user, done_asking],
    )


def run_questions(agent: Agent, req_summary: str, arch_summary: str) -> str:
    """Execute question phase."""
    result = agent(
        f"""Ask questions about items in "Features Not Found" only.
Do NOT question features in "Features Verified" - those exist.

ARCHITECTURE:
{arch_summary}

Ask about gaps listed above. Call done_asking when ready."""
    )
    return str(result)
