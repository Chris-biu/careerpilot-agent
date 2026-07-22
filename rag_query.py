import re

# 复用文档读取和文本切分功能。
# 这些功能已经在 rag_ingest.py 中实现，
# 这里不重复编写。
from rag_ingest import read_text_file, split_text

# 复用 DeepSeek 模型调用功能。
# ask_llm() 会负责把 Prompt 发送给大模型。
from llm_client import ask_llm


# 第一版演示固定读取这个 PDF 文件。
DOCUMENT_PATH = "sample.pdf"

# 第一版演示固定使用这个问题。
QUESTION = "What does the document say about Python and RAG?"


def extract_keywords(text: str) -> set[str]:
    """从英文文本中提取小写关键词，供第一版检索使用。"""
    # 统一转成小写，避免 Python 与 python 被当成不同的词。
    normalized_text = text.lower()

    # [a-z0-9]+ 表示提取连续的英文字符或数字。
    # 第一版只服务当前英文测试资料；中文语义检索会在后续 Embedding 阶段解决。
    return set(re.findall(r"[a-z0-9]+", normalized_text))


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
    """运行固定 PDF 和固定问题的最小 RAG 演示。"""

    # 读取 DOCUMENT_PATH 指定的 sample.pdf，
    # 返回 PDF 中提取出来的完整文本。
    document_text = read_text_file(DOCUMENT_PATH)

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

    # 使用固定问题 QUESTION 对所有文本片段进行检索。
    # 默认返回关键词重合数量最高的两个片段。
    relevant_chunks = retrieve_relevant_chunks(
        QUESTION,
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

    # 把“用户问题”和“检索到的资料片段”
    # 组合成最终发送给大模型的 Prompt。
    prompt = build_rag_prompt(
        QUESTION,
        relevant_chunks,
    )

    # 把 Prompt 发送给 DeepSeek，
    # 并使用 answer 保存模型返回的回答文本。
    answer = ask_llm(prompt)

    # 先打印标题，再输出模型回答。
    # 开头的 \n 会先换行，让终端输出更容易阅读。
    print("\n模型回答：")
    print(answer)


# Python 直接运行这个文件时，特殊变量 __name__
# 的值会是 "__main__"，因此会执行 main()。
#
# 如果其他文件只是导入 rag_query，
# 下面的 main() 不会运行，也就不会意外调用 DeepSeek。
if __name__ == "__main__":
    main()
