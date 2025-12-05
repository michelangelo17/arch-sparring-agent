from strands import Agent, tool


def create_question_agent(model_id: str = "amazon.nova-2-lite-v1:0"):
    """Create the Question Agent for clarifying questions phase."""

    questions_asked = []

    @tool
    def ask_user(question: str) -> str:
        """Ask the user a clarifying question about the architecture."""
        questions_asked.append(question)
        print(f"\n❓ [{len(questions_asked)}] {question}")
        response = input("Your answer: ")
        return response

    @tool
    def done_asking() -> str:
        """Call when you have enough information to proceed to sparring."""
        return "Proceeding to sparring phase."

    agent = Agent(
        name="QuestionAgent",
        model=model_id,
        system_prompt="""Ask clarifying questions about the architecture.

Focus on understanding:
- Scale and traffic expectations
- Security and compliance requirements
- Reliability and availability needs
- Key constraints and limitations

Guidelines:
- Ask one question at a time
- If user says "no", "none", or similar, move on
- Stop when you understand the critical aspects
- Call done_asking when ready to proceed""",
        tools=[ask_user, done_asking],
    )

    return agent


def run_questions(agent: Agent, req_summary: str, arch_summary: str) -> str:
    """Run the question phase and return the context gathered."""
    result = agent(
        f"""Review these summaries and ask clarifying questions:

REQUIREMENTS:
{req_summary}

ARCHITECTURE:
{arch_summary}

Ask questions to understand gaps. Call done_asking when ready."""
    )
    return str(result)

