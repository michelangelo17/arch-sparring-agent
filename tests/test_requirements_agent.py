import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock strands and others to prevent ImportError
for mod in [
    "strands",
    "strands.types",
    "strands.types.exceptions",
    "botocore",
    "botocore.exceptions",
    "arch_sparring_agent.tools.document_parser",
    "frontmatter",
    "bedrock_agentcore",
    "bedrock_agentcore.memory",
    "bedrock_agentcore.memory.integrations.strands",
    "bedrock_agentcore.memory.integrations.strands.config",
    "bedrock_agentcore.memory.integrations.strands.session_manager",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from arch_sparring_agent.agents.requirements_agent import create_requirements_agent  # noqa: E402


class TestRequirementsAgent(unittest.TestCase):
    def setUp(self):
        self.mock_parser_cls = patch(
            "arch_sparring_agent.tools.document_parser.DocumentParser"
        ).start()
        self.mock_parser = self.mock_parser_cls.return_value

        # Patch Agent and tool in the module under test
        self.mock_agent_cls = patch("arch_sparring_agent.agents.requirements_agent.Agent").start()
        self.mock_tool = patch(
            "arch_sparring_agent.agents.requirements_agent.tool", side_effect=lambda x: x
        ).start()

    def tearDown(self):
        patch.stopall()

    def test_create_requirements_agent(self):
        create_requirements_agent("docs_dir")

        # Check Agent was created
        self.mock_agent_cls.assert_called()
        args, kwargs = self.mock_agent_cls.call_args
        self.assertEqual(kwargs["name"], "RequirementsAnalyst")
        self.assertEqual(len(kwargs["tools"]), 2)  # read_document, list_available_documents

        # Verify tools are what we expect
        tools = kwargs["tools"]
        read_doc_tool = tools[0]
        list_docs_tool = tools[1]

        # Find which is which
        if list_docs_tool.__name__ == "read_document":
            read_doc_tool, list_docs_tool = list_docs_tool, read_doc_tool

        self.assertEqual(list_docs_tool.__name__, "list_available_documents")
        self.assertEqual(read_doc_tool.__name__, "read_document")

        self.mock_parser_cls.assert_called()
        self.mock_parser.list_documents.return_value = ["doc1.md", "doc2.md"]

        docs = list_docs_tool()
        self.assertEqual(docs, ["doc1.md", "doc2.md"])
        self.mock_parser.list_documents.assert_called()

    def test_read_document_short(self):
        create_requirements_agent("docs_dir")
        # Get the tool from the call args
        tools = self.mock_agent_cls.call_args[1]["tools"]
        read_doc_tool = next(t for t in tools if t.__name__ == "read_document")

        self.mock_parser.read_markdown_file.return_value = {
            "content": "Short content",
            "metadata": {},
        }

        result = read_doc_tool("test.md")
        self.assertIn("Short content", result)
        self.assertNotIn("Summarized", result)

    def test_read_document_long_needs_summary(self):
        # Reset Agent calls from create_requirements_agent
        self.mock_agent_cls.reset_mock()

        create_requirements_agent("docs_dir")
        tools = self.mock_agent_cls.call_args[1]["tools"]
        read_doc_tool = next(t for t in tools if t.__name__ == "read_document")

        # Clear mock calls again
        self.mock_agent_cls.reset_mock()

        long_content = "A" * 30000
        self.mock_parser.read_markdown_file.return_value = {"content": long_content, "metadata": {}}

        # Setup the summarizer agent mock
        mock_summarizer = MagicMock()
        mock_summarizer.return_value = "Summary of content"
        self.mock_agent_cls.return_value = mock_summarizer

        result = read_doc_tool("long.md")

        # Verify summarizer agent was created
        self.mock_agent_cls.assert_called_with(
            name="DocSummarizer",
            model="amazon.nova-2-lite-v1:0",
            system_prompt=unittest.mock.ANY,
            tools=[],
        )

        # Verify summarizer was called
        mock_summarizer.assert_called()
        self.assertIn("Summary of content", result)
        self.assertIn("(Summarized)", result)


if __name__ == "__main__":
    unittest.main()
