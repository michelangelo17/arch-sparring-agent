from unittest.mock import ANY, MagicMock, patch

import pytest

from arch_sparring_agent.agents.requirements_agent import create_requirements_agent


@pytest.fixture
def requirements_agent_mocks():
    mock_parser_cls = patch(
        "arch_sparring_agent.tools.document_parser.DocumentParser"
    ).start()
    mock_parser = mock_parser_cls.return_value

    mock_agent_cls = patch("arch_sparring_agent.agents.requirements_agent.Agent").start()
    patch(
        "arch_sparring_agent.agents.requirements_agent.tool", side_effect=lambda x: x
    ).start()

    try:
        yield type(
            "Mocks",
            (),
            {
                "mock_parser_cls": mock_parser_cls,
                "mock_parser": mock_parser,
                "mock_agent_cls": mock_agent_cls,
            },
        )()
    finally:
        patch.stopall()


def test_create_requirements_agent(requirements_agent_mocks):
    m = requirements_agent_mocks
    create_requirements_agent("docs_dir", "test-model")

    m.mock_agent_cls.assert_called()
    args, kwargs = m.mock_agent_cls.call_args
    assert kwargs["name"] == "RequirementsAnalyst"
    assert len(kwargs["tools"]) == 2

    tools = kwargs["tools"]
    read_doc_tool = tools[0]
    list_docs_tool = tools[1]

    if list_docs_tool.__name__ == "read_document":
        read_doc_tool, list_docs_tool = list_docs_tool, read_doc_tool

    assert list_docs_tool.__name__ == "list_available_documents"
    assert read_doc_tool.__name__ == "read_document"

    m.mock_parser_cls.assert_called()
    m.mock_parser.list_documents.return_value = ["doc1.md", "doc2.md"]

    docs = list_docs_tool()
    assert docs == ["doc1.md", "doc2.md"]
    m.mock_parser.list_documents.assert_called()


def test_read_document_short(requirements_agent_mocks):
    m = requirements_agent_mocks
    create_requirements_agent("docs_dir", "test-model")
    tools = m.mock_agent_cls.call_args[1]["tools"]
    read_doc_tool = next(t for t in tools if t.__name__ == "read_document")

    m.mock_parser.read_markdown_file.return_value = {
        "content": "Short content",
        "metadata": {},
    }

    result = read_doc_tool("test.md")
    assert "Short content" in result
    assert "Summarized" not in result


def test_read_document_long_needs_summary(requirements_agent_mocks):
    m = requirements_agent_mocks
    m.mock_agent_cls.reset_mock()

    create_requirements_agent("docs_dir", "test-model")
    tools = m.mock_agent_cls.call_args[1]["tools"]
    read_doc_tool = next(t for t in tools if t.__name__ == "read_document")

    m.mock_agent_cls.reset_mock()

    long_content = "A" * 30000
    m.mock_parser.read_markdown_file.return_value = {"content": long_content, "metadata": {}}

    mock_summarizer = MagicMock()
    mock_summarizer.return_value = "Summary of content"
    m.mock_agent_cls.return_value = mock_summarizer

    result = read_doc_tool("long.md")

    m.mock_agent_cls.assert_called_with(
        name="DocSummarizer",
        model="test-model",
        callback_handler=None,
        system_prompt=ANY,
        tools=[],
    )

    mock_summarizer.assert_called()
    assert "Summary of content" in result
    assert "(Summarized)" in result
