# 第一周学习笔记：Prompt、Embedding 与 RAG

日期：2026-07-23

## 1. 本周目标与完成情况

本周目标是理解 LLM 应用的基本组成，并完成一个可以从本地文档检索资料、调用大模型回答问题的最小 RAG Demo。

目前已经完成：

- 使用 `llm_client.py` 封装 DeepSeek 的 OpenAI-compatible API 调用。
- 使用 `rag_ingest.py` 读取 TXT、Markdown 和 PDF 文档。
- 按固定字符长度切分文档，并给相邻片段保留少量重叠内容。
- 使用 `rag_query.py` 根据问题中的英文关键词检索相关片段。
- 从终端传入文档路径和用户问题，不再把输入写死在程序中。
- 将检索结果放入 Prompt，调用 DeepSeek 生成回答。
- 在模型回答后显示来源文件和检索片段编号。
- 使用离线自动化测试验证检索、Prompt、命令行参数和引用输出。

尚未完成：

- 还没有生成 Embedding，也没有接入向量数据库。
- 还不能根据中文语义或同义词进行检索。
- 引用只能定位到文件和检索片段，暂时不能定位到 PDF 页码或原文段落。

## 2. Prompt

### 2.1 Prompt 是什么

Prompt 是发送给大模型的输入。它不仅包含用户的问题，还可以包含任务说明、背景资料、输出格式和限制条件。

可以把 Prompt 理解成给模型的一份“任务单”：写得越明确，模型越容易知道应该依据什么回答、不能做什么、结果应当是什么形式。

### 2.2 当前项目中的 Prompt 结构

`rag_query.py` 中的 `build_rag_prompt()` 生成了四部分内容：

1. **角色和任务**：要求模型成为一个有帮助的助手。
2. **回答约束**：只能使用提供的 Context；资料没有答案时必须明确说明。
3. **Context**：程序从本地文档中检索出的相关片段。
4. **Question**：用户在终端中输入的问题。

当前 Prompt 的核心形式是：

```text
You are a helpful assistant.
Answer the question using only the provided context.
If the context does not contain the answer, say that the context does not provide it.

Context:
<检索到的文档片段>

Question:
<用户问题>
```

### 2.3 为什么要限制模型只能使用 Context

大模型可能根据训练数据补充信息，也可能生成听起来合理但没有依据的内容。要求模型只依据 Context 回答，可以降低幻觉风险，并让回答更容易核查。

不过 Prompt 约束不是绝对保证。后续仍需要引用溯源、结构化输出和评估模块共同检查回答质量。

## 3. Embedding

### 3.1 Embedding 是什么

Embedding 是把文本转换成一组数字，也就是向量。语义相近的文本，其向量在向量空间中的距离通常也更近。

例如，“Python 开发经验”和“熟悉 Python 编程”虽然没有完全相同的字面表达，但 Embedding 模型可能把它们转换成相近的向量，因此可以通过语义找到彼此。

### 3.2 Embedding 在 RAG 中的作用

典型的向量检索流程是：

1. 把每个文档片段转换成 Embedding。
2. 把用户问题也转换成 Embedding。
3. 计算问题向量与文档向量的相似度。
4. 选出相似度最高的若干文档片段。

### 3.3 当前项目为什么还不算向量检索

当前 `retrieve_relevant_chunks()` 使用的是**英文关键词重合数量**：问题和片段拥有越多相同单词，片段得分越高。

这种方式容易理解，也适合验证最小 RAG 流程，但存在明显限制：

- 不理解同义词。
- 不理解句子的真正语义。
- 当前正则表达式主要提取英文单词和数字，不适合中文语义检索。

因此，Embedding 是后续升级检索质量的重要步骤，但不能把当前关键词检索描述成 Embedding 检索。

## 4. RAG

### 4.1 RAG 是什么

RAG 是 Retrieval-Augmented Generation，中文通常称为“检索增强生成”。

它的核心思想是：先从外部资料中检索与问题相关的内容，再把这些内容交给大模型生成回答。模型不只依赖训练时记住的知识，还能使用当前提供的本地资料。

### 4.2 当前 Demo 的完整流程

```text
用户输入文档路径和问题
        ↓
读取 TXT / Markdown / PDF
        ↓
按固定字符长度切分文本
        ↓
计算问题与片段的关键词重合数量
        ↓
选择得分最高的两个片段
        ↓
把片段和问题组合成 Prompt
        ↓
调用 DeepSeek 生成回答
        ↓
显示答案、来源文件和检索片段编号
```

这对应 RAG 的三个核心阶段：

- **Retrieval（检索）**：`retrieve_relevant_chunks()` 找出相关片段。
- **Augmentation（增强）**：`build_rag_prompt()` 把片段加入 Prompt。
- **Generation（生成）**：`ask_llm()` 调用 DeepSeek 生成最终答案。

### 4.3 文档为什么要切分

如果把整份长文档一次性交给模型，会增加 token 消耗，也会混入大量与问题无关的信息。切分后，程序可以只选择少量相关片段。

当前切分使用：

- `chunk_size=200`：每个片段最多约 200 个字符。
- `chunk_overlap=20`：相邻片段重复保留 20 个字符，降低句子刚好被边界切断造成的信息损失。

这种固定字符切分简单，但可能切断标题、段落或完整句子。后续可以升级为按标题、段落和 token 长度混合切分。

### 4.4 引用来源的当前含义

程序现在会在模型回答后显示：

```text
引用来源：
- sample.pdf（检索片段 1）
- sample.pdf（检索片段 2）
```

这证明回答使用了哪个文件，并能与终端中打印的检索片段对应起来。

当前编号表示“检索结果中的顺序”，还不是 PDF 页码，也不是原始文档中的精确位置。要实现更精确的引用，需要在读取和切分时同时保存页码、段落号等元数据。

## 5. 如何运行当前 Demo

激活虚拟环境后，可以运行：

```powershell
python .\rag_query.py .\sample.pdf "What technologies are mentioned?"
```

两个位置参数分别是：

1. `.\sample.pdf`：要检索的本地文档路径。
2. `"What technologies are mentioned?"`：希望系统回答的问题。

离线测试命令：

```powershell
python -m unittest discover -s tests
```

离线测试不会调用 DeepSeek，也不会消耗 API 额度。

## 6. 第一周总结

我已经完成了一个最小可运行 RAG 闭环：文档读取、文本切分、关键词检索、Prompt 增强、DeepSeek 生成、引用展示和离线测试。

我需要特别记住：当前项目实现的是关键词检索，不是 Embedding 向量检索；引用能够标记来源文件和检索片段，但还没有精确的页码与段落元数据。这些限制不是需要隐藏的问题，而是后续迭代的明确方向。
