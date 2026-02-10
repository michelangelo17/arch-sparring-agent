"""Unit tests for tool modules: CloudFormationAnalyzer, DocumentParser, SourceAnalyzer."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from arch_sparring_agent.exceptions import ToolError
from arch_sparring_agent.tools.cfn_analyzer import CloudFormationAnalyzer
from arch_sparring_agent.tools.document_parser import DocumentParser
from arch_sparring_agent.tools.source_analyzer import SourceAnalyzer


class CloudFormationAnalyzerTests(unittest.TestCase):
    """Tests for CloudFormationAnalyzer."""

    def test_list_templates_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = CloudFormationAnalyzer(tmp)
            self.assertEqual(analyzer.list_templates(), [])

    def test_list_templates_finds_yaml_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "template.yaml").write_text("{}")
            (p / "stack.yml").write_text("{}")
            (p / "cdk.out").mkdir()
            (p / "cdk.out" / "out.json").write_text("{}")
            analyzer = CloudFormationAnalyzer(tmp)
            names = analyzer.list_templates()
            self.assertIn("template.yaml", names)
            self.assertIn("stack.yml", names)
            self.assertIn("cdk.out/out.json", names)

    def test_read_template_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.yaml"
            p.write_text("Resources:\n  Bucket: {}")
            analyzer = CloudFormationAnalyzer(tmp)
            out = analyzer.read_template("t.yaml")
            self.assertEqual(out, "Resources:\n  Bucket: {}")

    def test_read_template_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = CloudFormationAnalyzer(tmp)
            with self.assertRaises(ToolError) as ctx:
                analyzer.read_template("../../../etc/passwd")
            self.assertIn("Path traversal detected", str(ctx.exception))

    def test_read_template_file_too_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "large.yaml"
            p.write_text("x" * 20)  # 20 chars
            analyzer = CloudFormationAnalyzer(tmp)
            with patch("arch_sparring_agent.tools.cfn_analyzer.CFN_MAX_CHARS", 10):
                with self.assertRaises(ToolError) as ctx:
                    analyzer.read_template("large.yaml")
                self.assertIn("exceeds the", str(ctx.exception))


class DocumentParserTests(unittest.TestCase):
    """Tests for DocumentParser."""

    def test_list_documents_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            parser = DocumentParser(tmp)
            self.assertEqual(parser.list_documents(), [])

    def test_list_documents_finds_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "req.md").write_text("# Req")
            (p / "other.txt").write_text("x")
            parser = DocumentParser(tmp)
            names = parser.list_documents()
            self.assertEqual(names, ["req.md"])

    def test_read_markdown_file_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "doc.md"
            p.write_text("---\ntitle: x\n---\n# Body")
            parser = DocumentParser(tmp)
            mock_doc = MagicMock()
            mock_doc.content = "# Body"
            mock_doc.metadata = {"title": "x"}
            with patch(
                "arch_sparring_agent.tools.document_parser.frontmatter.loads",
                return_value=mock_doc,
            ):
                out = parser.read_markdown_file("doc.md")
            self.assertEqual(out["filename"], "doc.md")
            self.assertEqual(out["content"], "# Body")
            self.assertEqual(out["metadata"], {"title": "x"})

    def test_read_markdown_file_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parser = DocumentParser(tmp)
            with self.assertRaises(ToolError) as ctx:
                parser.read_markdown_file("../../../etc/passwd")
            self.assertIn("Path traversal detected", str(ctx.exception))

    def test_read_markdown_file_too_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "large.md"
            p.write_text("x" * 20)
            parser = DocumentParser(tmp)
            with patch("arch_sparring_agent.tools.document_parser.DOC_MAX_CHARS", 10):
                with self.assertRaises(ToolError) as ctx:
                    parser.read_markdown_file("large.md")
                self.assertIn("exceeds the", str(ctx.exception))


class SourceAnalyzerTests(unittest.TestCase):
    """Tests for SourceAnalyzer."""

    def test_list_source_files_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = SourceAnalyzer(tmp)
            self.assertEqual(analyzer.list_source_files(), [])

    def test_list_source_files_finds_supported_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "main.py").write_text("print(1)")
            (p / "src").mkdir()
            (p / "src" / "lib.ts").write_text("export {}")
            (p / "script.sh").write_text("echo x")
            analyzer = SourceAnalyzer(tmp)
            names = analyzer.list_source_files()
            self.assertIn("main.py", names)
            self.assertIn("src/lib.ts", names)
            self.assertNotIn("script.sh", names)

    def test_list_source_files_excludes_node_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "main.py").write_text("x")
            (p / "node_modules" / "pkg").mkdir(parents=True)
            (p / "node_modules" / "pkg" / "index.js").write_text("x")
            analyzer = SourceAnalyzer(tmp)
            names = analyzer.list_source_files()
            self.assertIn("main.py", names)
            self.assertFalse(any("node_modules" in n for n in names))

    def test_read_source_file_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "handler.py"
            p.write_text("def handler(event, context): pass")
            analyzer = SourceAnalyzer(tmp)
            out = analyzer.read_source_file("handler.py")
            self.assertIn("def handler", out)

    def test_read_source_file_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = SourceAnalyzer(tmp)
            with self.assertRaises(ToolError) as ctx:
                analyzer.read_source_file("../../../etc/passwd")
            self.assertIn("Path traversal detected", str(ctx.exception))

    def test_read_source_file_too_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "large.py"
            p.write_text("x" * 20)
            analyzer = SourceAnalyzer(tmp)
            with patch("arch_sparring_agent.tools.source_analyzer.SOURCE_FILE_MAX_CHARS", 10):
                with self.assertRaises(ToolError) as ctx:
                    analyzer.read_source_file("large.py")
                self.assertIn("exceeds the", str(ctx.exception))

    def test_search_source_finds_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "a.py").write_text("def process_order():\n    pass")
            (p / "b.py").write_text("def other():\n    pass")
            analyzer = SourceAnalyzer(tmp)
            out = analyzer.search_source("process_order")
            self.assertIn("a.py", out)
            self.assertIn("process_order", out)
            self.assertIn("L1:", out)

    def test_search_source_no_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.py"
            p.write_text("def foo(): pass")
            analyzer = SourceAnalyzer(tmp)
            out = analyzer.search_source("nonexistent_token_xyz")
            self.assertEqual(out, "No matches found for: nonexistent_token_xyz")


if __name__ == "__main__":
    unittest.main()
