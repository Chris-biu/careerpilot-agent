import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import rag_query

# 导入需要验证的检索函数和 Prompt 构造函数。
# 这两个函数不会调用 API，因此测试可以完全离线运行。
from rag_query import (
    build_rag_prompt,
    parse_cli_arguments,
    retrieve_relevant_chunks,
)


class RetrieveRelevantChunksTests(unittest.TestCase):
    """测试：程序能从多个文本片段中找出最相关的片段。"""

    def test_returns_chunk_with_overlapping_keywords(self) -> None:
        """问题包含 Python、RAG 时，应优先返回包含这两个词的片段。"""
        chunks = [
            "Python and RAG are used for document retrieval.",
            "Communication skills are valuable in a team.",
        ]

        # top_k=1：只要求程序返回相关度最高的一个片段。
        result = retrieve_relevant_chunks(
            "How are Python and RAG used?",
            chunks,
            top_k=1,
        )

        # 第一个片段与问题共享 Python、RAG；第二个没有共享关键词。
        self.assertEqual(result, [chunks[0]])

    def test_keeps_a_chunk_when_no_keywords_overlap(self) -> None:
        """没有共同关键词时，仍然返回得分最高的片段。"""
        chunks = [
            "Python and RAG are used for document retrieval.",
        ]

        # 这个问题与片段没有任何共同关键词，
        # 因此该片段的关键词重合得分是 0。
        result = retrieve_relevant_chunks(
            "What is the weather today?",
            chunks,
            top_k=1,
        )

        # 第一版检索不会返回空列表，
        # 而是保留现有片段，让模型仍然拥有资料上下文。
        self.assertEqual(result, [chunks[0]])

    def test_builds_prompt_with_question_and_context(self) -> None:
        """Prompt 必须同时包含用户问题和检索到的资料片段。"""
        question = "Which technology is mentioned?"
        context_chunks = ["The document mentions Python."]

        # 调用 Prompt 构造函数，得到准备发送给模型的完整文本。
        prompt = build_rag_prompt(question, context_chunks)

        # 检查 Prompt 没有丢失问题或作为依据的资料。
        self.assertIn(question, prompt)
        self.assertIn(context_chunks[0], prompt)


class ParseCliArgumentsTests(unittest.TestCase):
    """测试：程序能够读取终端传入的文件路径和问题。"""

    def test_parses_document_path_and_question(self) -> None:
        """两个命令行参数应该被分别保存。"""

        # 这个列表模拟用户在终端中传入的两个参数。
        arguments = parse_cli_arguments(
            [
                "sample.pdf",
                "What does the document mention?",
            ]
        )

        # 检查文件路径和问题没有混淆或丢失。
        self.assertEqual(arguments.document_path, "sample.pdf")
        self.assertEqual(
            arguments.question,
            "What does the document mention?",
        )


class FormatSourceCitationsTests(unittest.TestCase):
    """测试：回答能够明确展示资料来源和对应的检索片段。"""

    def test_formats_document_name_and_chunk_numbers(self) -> None:
        """引用文本应该包含文件名，以及从 1 开始的片段编号。"""

        # 模拟程序从 sample.pdf 中检索出了两个相关片段。
        # 测试只验证引用格式，不读取文件，也不会调用大模型 API。
        citations = rag_query.format_source_citations(
            "documents/sample.pdf",
            chunk_count=2,
        )

        # 用户必须能明确看见这是引用来源，而不是普通回答内容。
        self.assertIn("引用来源：", citations)

        # 只显示文件名可以避免把用户电脑上的完整目录暴露到输出中。
        self.assertIn("sample.pdf", citations)
        self.assertNotIn("documents/sample.pdf", citations)

        # 两个检索片段都应该拥有可见编号，方便对应上方打印的片段。
        self.assertIn("检索片段 1", citations)
        self.assertIn("检索片段 2", citations)


class MainOutputTests(unittest.TestCase):
    """测试：完整问答流程会把引用来源打印到终端。"""

    def test_prints_source_citations_after_model_answer(self) -> None:
        """模型回答之后应该显示来源文件和检索片段编号。"""

        # StringIO 是内存中的文本缓冲区。
        # redirect_stdout() 会把 print() 的内容临时写入这里，
        # 让测试可以检查终端输出，而不用真正打开另一个终端。
        terminal_output = StringIO()

        # main() 原本会读取文件并调用 DeepSeek。
        # 测试用固定返回值替换这些外部操作，避免访问磁盘和消耗 API 额度；
        # 检索、Prompt 构造和最终打印流程仍然运行真实代码。
        with (
            patch.object(
                rag_query,
                "parse_cli_arguments",
                return_value=Namespace(
                    document_path="documents/sample.pdf",
                    question="Which technology is mentioned?",
                ),
            ),
            patch.object(
                rag_query,
                "read_text_file",
                return_value="The document mentions Python.",
            ),
            patch.object(
                rag_query,
                "split_text",
                return_value=["The document mentions Python."],
            ),
            patch.object(
                rag_query,
                "ask_llm",
                return_value="Python is mentioned.",
            ),
            redirect_stdout(terminal_output),
        ):
            rag_query.main()

        output = terminal_output.getvalue()

        # 这两个断言证明引用不仅被生成，而且真正出现在最终输出中。
        self.assertIn("引用来源：", output)
        self.assertIn("sample.pdf（检索片段 1）", output)


if __name__ == "__main__":
    # 允许直接运行此文件时也执行测试。
    unittest.main()
