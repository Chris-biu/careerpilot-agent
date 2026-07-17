from pathlib import Path
import fitz

def read_text_file(file_path: str) -> str:
    """读取 TXT、Markdown 或 PDF 文件的文本内容。"""
    path = Path(file_path)
    file_suffix = path.suffix.lower()

    if file_suffix == ".pdf":
        return read_pdf_file(file_path)

    if file_suffix not in {".txt", ".md"}:
        raise ValueError("目前只支持 .txt、.md 和 .pdf 文件")

    return path.read_text(encoding="utf-8")

def read_pdf_file(file_path: str) -> str:
    """提取 PDF 每一页的文本内容。"""
    with fitz.open(file_path) as document:
        return "\n".join(page.get_text() for page in document)

def split_text(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 10,
) -> list[str]:
    """按固定长度切分文本，并保留少量重叠内容。"""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    step = chunk_size - chunk_overlap

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), step)
    ]
if __name__ == "__main__":
    content = read_text_file("sample.pdf")
    chunks = split_text(content, chunk_size=50)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- 第 {index} 段 ---")
        print(chunk)
