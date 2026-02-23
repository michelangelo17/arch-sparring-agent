"""Unit tests for tool modules: CloudFormationAnalyzer, DocumentParser, SourceAnalyzer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arch_sparring_agent.exceptions import ToolError
from arch_sparring_agent.tools.cfn_analyzer import CloudFormationAnalyzer
from arch_sparring_agent.tools.document_parser import DocumentParser
from arch_sparring_agent.tools.source_analyzer import SourceAnalyzer


def test_list_templates_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        analyzer = CloudFormationAnalyzer(tmp)
        assert analyzer.list_templates() == []


def test_list_templates_finds_yaml_json():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "template.yaml").write_text("{}")
        (p / "stack.yml").write_text("{}")
        (p / "cdk.out").mkdir()
        (p / "cdk.out" / "out.json").write_text("{}")
        analyzer = CloudFormationAnalyzer(tmp)
        names = analyzer.list_templates()
        assert "template.yaml" in names
        assert "stack.yml" in names
        assert "cdk.out/out.json" in names


def test_read_template_success():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.yaml"
        p.write_text("Resources:\n  Bucket: {}")
        analyzer = CloudFormationAnalyzer(tmp)
        out = analyzer.read_template("t.yaml")
        assert out == "Resources:\n  Bucket: {}"


def test_read_template_path_traversal_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        analyzer = CloudFormationAnalyzer(tmp)
        with pytest.raises(ToolError) as exc_info:
            analyzer.read_template("../../../etc/passwd")
        assert "Path traversal detected" in str(exc_info.value)


def test_read_template_file_too_large():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "large.yaml"
        p.write_text("x" * 20)
        analyzer = CloudFormationAnalyzer(tmp)
        with patch("arch_sparring_agent.tools.cfn_analyzer.CFN_MAX_BYTES", 10):
            with pytest.raises(ToolError) as exc_info:
                analyzer.read_template("large.yaml")
            assert "exceeds the" in str(exc_info.value)


def test_list_documents_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        parser = DocumentParser(tmp)
        assert parser.list_documents() == []


def test_list_documents_finds_md():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "req.md").write_text("# Req")
        (p / "other.txt").write_text("x")
        parser = DocumentParser(tmp)
        names = parser.list_documents()
        assert names == ["req.md"]


def test_read_markdown_file_success():
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
        assert out["filename"] == "doc.md"
        assert out["content"] == "# Body"
        assert out["metadata"] == {"title": "x"}


def test_read_markdown_file_path_traversal_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        parser = DocumentParser(tmp)
        with pytest.raises(ToolError) as exc_info:
            parser.read_markdown_file("../../../etc/passwd")
        assert "Path traversal detected" in str(exc_info.value)


def test_read_markdown_file_too_large():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "large.md"
        p.write_text("x" * 20)
        parser = DocumentParser(tmp)
        with patch("arch_sparring_agent.tools.document_parser.DOC_MAX_BYTES", 10):
            with pytest.raises(ToolError) as exc_info:
                parser.read_markdown_file("large.md")
            assert "exceeds the" in str(exc_info.value)


def test_list_source_files_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        analyzer = SourceAnalyzer(tmp)
        assert analyzer.list_source_files() == []


def test_list_source_files_finds_supported_extensions():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "main.py").write_text("print(1)")
        (p / "src").mkdir()
        (p / "src" / "lib.ts").write_text("export {}")
        (p / "script.sh").write_text("echo x")
        analyzer = SourceAnalyzer(tmp)
        names = analyzer.list_source_files()
        assert "main.py" in names
        assert "src/lib.ts" in names
        assert "script.sh" not in names


def test_list_source_files_excludes_node_modules():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "main.py").write_text("x")
        (p / "node_modules" / "pkg").mkdir(parents=True)
        (p / "node_modules" / "pkg" / "index.js").write_text("x")
        analyzer = SourceAnalyzer(tmp)
        names = analyzer.list_source_files()
        assert "main.py" in names
        assert not any("node_modules" in n for n in names)


def test_read_source_file_success():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "handler.py"
        p.write_text("def handler(event, context): pass")
        analyzer = SourceAnalyzer(tmp)
        out = analyzer.read_source_file("handler.py")
        assert "def handler" in out


def test_read_source_file_path_traversal_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        analyzer = SourceAnalyzer(tmp)
        with pytest.raises(ToolError) as exc_info:
            analyzer.read_source_file("../../../etc/passwd")
        assert "Path traversal detected" in str(exc_info.value)


def test_read_source_file_too_large():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "large.py"
        p.write_text("x" * 20)
        analyzer = SourceAnalyzer(tmp)
        with patch("arch_sparring_agent.tools.source_analyzer.SOURCE_MAX_BYTES", 10):
            with pytest.raises(ToolError) as exc_info:
                analyzer.read_source_file("large.py")
            assert "exceeds the" in str(exc_info.value)


def test_search_source_finds_pattern():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "a.py").write_text("def process_order():\n    pass")
        (p / "b.py").write_text("def other():\n    pass")
        analyzer = SourceAnalyzer(tmp)
        out = analyzer.search_source("process_order")
        assert "a.py" in out
        assert "process_order" in out
        assert "L1:" in out


def test_search_source_no_matches():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.py"
        p.write_text("def foo(): pass")
        analyzer = SourceAnalyzer(tmp)
        out = analyzer.search_source("nonexistent_token_xyz")
        assert out == "No matches found for: nonexistent_token_xyz"
