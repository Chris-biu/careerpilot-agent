from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import rag_ingest


class ReadTextFileTests(unittest.TestCase):
    """测试：统一文档入口能够把不同格式交给正确的读取器。"""

    def test_returns_text_from_docx_reader(self) -> None:
        """传入 DOCX 路径时，应该返回 DOCX 读取器提取的文本。"""

        expected_text = "# 脱敏简历\n\n技能：Python、RAG"

        # DOCX 转换属于外部依赖，本测试只验证 read_text_file() 的格式分派。
        with patch.object(
            rag_ingest,
            "read_docx_file",
            return_value=expected_text,
        ):
            try:
                result = rag_ingest.read_text_file("resume.docx")
            except ValueError as error:
                self.fail(f"DOCX 文件仍被当成不支持的格式：{error}")

        self.assertEqual(result, expected_text)


class ReadDocxFileTests(unittest.TestCase):
    """测试：DOCX 专用读取器能够返回 MarkItDown 提取的 Markdown。"""

    def test_returns_text_content_from_markitdown(self) -> None:
        """读取器应该返回转换结果中的 text_content。"""

        expected_text = "# 脱敏简历\n\n## 技能\n\n- Python\n- RAG"

        # MarkItDown.convert() 的真实返回对象提供 text_content 属性。
        # 这里隔离第三方库和磁盘文件，只验证我们自己的读取器
        # 是否正确取出该属性并返回给上层代码。
        converter = Mock()
        converter.convert.return_value = SimpleNamespace(
            text_content=expected_text,
        )

        with patch.object(
            rag_ingest,
            "MarkItDown",
            return_value=converter,
        ):
            result = rag_ingest.read_docx_file("resume.docx")

        self.assertEqual(result, expected_text)


if __name__ == "__main__":
    unittest.main()
