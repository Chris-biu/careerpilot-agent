import argparse
import re
from pathlib import Path

# 复用文档读取和文本切分功能。
# 这些功能已经在 rag_ingest.py 中实现，
# 这里不重复编写。
from rag_ingest import read_text_file, split_text

# 复用 DeepSeek 模型调用功能。
# ask_llm() 会负责把 Prompt 发送给大模型。
from llm_client import ask_llm


def extract_keywords(text: str) -> set[str]:
    """从英文文本中提取小写关键词，供第一版检索使用。"""
    # 统一转成小写，避免 Python 与 python 被当成不同的词。
    normalized_text = text.lower()

    # [a-z0-9]+ 表示提取连续的英文字符或数字。
    # 第一版只服务当前英文测试资料；中文语义检索会在后续 Embedding 阶段解决。
    return set(re.findall(r"[a-z0-9]+", normalized_text))


def parse_cli_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """解析终端传入的文档路径和用户问题。"""

    # 创建一个参数解析器。
    # 它负责定义参数规则，并在格式错误时给出提示。
    parser = argparse.ArgumentParser(
        description="从文档中检索相关片段，并回答用户问题。",
    )

    # 第一个位置参数：要读取的文档路径。
    parser.add_argument(
        "document_path",
        help="待读取的 TXT、Markdown 或 PDF 文件路径",
    )

    # 第二个位置参数：用户提出的问题。
    parser.add_argument(
        "question",
        help="希望模型回答的问题",
    )

    # 测试时传入列表；实际运行时传入 None，
    # argparse 会自动读取终端中的参数。
    return parser.parse_args(arguments)


def format_source_citations(
    document_path: str,
    chunk_count: int,
) -> str:
    """生成面向用户的引用来源文本。"""

    # Path.name 只保留路径末尾的文件名。
    # 例如 documents/sample.pdf 会变成 sample.pdf，
    # 这样既能说明资料来源，也不会暴露用户电脑上的完整目录。
    document_name = Path(document_path).name

    # range() 默认从 0 开始，但终端中展示的检索片段从 1 开始编号。
    # 因此这里显式使用 1 作为起点，并让终点比片段数量多 1。
    citation_lines = [
        f"- {document_name}（检索片段 {index}）"
        for index in range(1, chunk_count + 1)
    ]

    # 把标题和每一条来源用换行符连接，形成便于阅读的结构化文本。
    return "引用来源：\n" + "\n".join(citation_lines)


def retrieve_relevant_chunks(
    question: str,
    chunks: list[str],
    top_k: int = 2,
) -> list[str]:
    """根据关键词重合数量，返回最相关的前 top_k 个文本片段。"""
    # 先把问题转换成关键词集合，后续所有片段都和它比较。
    question_keywords = extract_keywords(question)

    # 每一项保存“得分 + 原始片段”，例如：(2, "Python and RAG ...")。
    scored_chunks: list[tuple[int, str]] = []

    for chunk in chunks:
        # & 是集合交集：只留下同时出现在问题和片段中的关键词。
        overlap_count = len(question_keywords & extract_keywords(chunk))
        scored_chunks.append((overlap_count, chunk))

    # 得分高的片段排在前面；相同得分时保留原来的资料顺序。
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    # 只取前 top_k 个，并去掉每项中用于排序的数字得分。
    return [chunk for _, chunk in scored_chunks[:top_k]]


def build_rag_prompt(
    question: str,
    context_chunks: list[str],
) -> str:
    """把检索到的文本片段和用户问题组合成 Prompt。"""

    # 用分隔线连接多个文本片段，
    # 让模型能够清楚地区分不同资料。
    context = "\n\n---\n\n".join(context_chunks)

    # f-string 会把 {context} 和 {question}
    # 替换成实际的资料内容和用户问题。
    return f"""You are a helpful assistant.
Answer the question using only the provided context.
If the context does not contain the answer, say that the context does not provide it.

Context:
{context}

Question:
{question}
"""


def main() -> None:
    """运行用户指定文档和问题的 RAG 演示。"""

    # 读取用户在终端中传入的两个参数。
    arguments = parse_cli_arguments()

    # 使用用户传入的文件路径读取文档，
    # 不再固定读取 sample.pdf。
    document_text = read_text_file(arguments.document_path)

    # 把完整文本切成多个较小片段。
    # chunk_size=200：每个片段最多包含 200 个字符。
    # chunk_overlap=20：相邻片段重复保留 20 个字符，
    # 避免一句话刚好被切断后完全丢失上下文。
    chunks = split_text(
        document_text,
        chunk_size=200,
        chunk_overlap=20,
    )

    # 空列表在 if 判断中等同于 False。
    # 如果没有得到任何片段，就不能继续进行检索。
    if not chunks:
        raise ValueError("文档中没有可以检索的文本内容")

    # 使用用户在终端中传入的问题检索相关片段。
    # 默认返回关键词重合数量最高的两个片段。
    relevant_chunks = retrieve_relevant_chunks(
        arguments.question,
        chunks,
    )

    # 先把检索结果打印出来，
    # 方便我们确认程序究竟把哪些资料交给了大模型。
    print("检索到的资料片段：")

    # enumerate() 可以在遍历片段时，同时生成编号。
    # start=1 表示编号从 1 开始，而不是从 0 开始。
    for index, chunk in enumerate(relevant_chunks, start=1):
        print(f"\n--- 片段 {index} ---")
        print(chunk)

    # 使用同一个用户问题和检索到的资料片段构造最终 Prompt。
    prompt = build_rag_prompt(
        arguments.question,
        relevant_chunks,
    )

    # 把 Prompt 发送给 DeepSeek，
    # 并使用 answer 保存模型返回的回答文本。
    answer = ask_llm(prompt)

    # 先打印标题，再输出模型回答。
    # 开头的 \n 会先换行，让终端输出更容易阅读。
    print("\n模型回答：")
    print(answer)

    # 根据用户传入的文档路径和实际检索到的片段数量，
    # 生成结构化引用，让用户知道回答使用了哪个文件中的哪些片段。
    source_citations = format_source_citations(
        arguments.document_path,
        chunk_count=len(relevant_chunks),
    )

    # 把引用放在模型回答之后，作为回答依据的独立区域。
    # 开头先换行，避免引用标题紧贴在回答最后一行后面。
    print(f"\n{source_citations}")


# Python 直接运行这个文件时，特殊变量 __name__
# 的值会是 "__main__"，因此会执行 main()。
#
# 如果其他文件只是导入 rag_query，
# 下面的 main() 不会运行，也就不会意外调用 DeepSeek。
if __name__ == "__main__":
    main()
