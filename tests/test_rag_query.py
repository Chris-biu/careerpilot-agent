import unittest

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


if __name__ == "__main__":
    # 允许直接运行此文件时也执行测试。
    unittest.main()
