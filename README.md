# CareerPilot Agent



CareerPilot 是一个面向 AI / AI Agent 求职场景的岗位能力分析与项目升级 Agent。用户输入岗位 JD、个人简历及已有项目材料后，系统会解析岗位所需能力，并优先评估用户现有项目是否适合通过真实功能改造来补齐相关技术栈。



对于技术要求清晰的 JD，系统直接提取必需技能、加分项和经验要求；对于“精通 AI”“熟悉前后端”等模糊 JD，系统结合带来源的同类岗位材料进行能力补全，并严格区分“原文明确要求”和“基于市场证据的推断”。



系统最终输出项目适配排序、可执行的改造方案、验收标准与能力证据清单；只有在没有合适项目可改造时，才生成新的最小作品项目。CareerPilot 的目标不是包装简历，而是帮助求职者把已有经历升级为可运行、可验证、可在面试中解释的岗位能力证据。

## 当前功能

- 调用 DeepSeek 的 OpenAI-compatible API。
- 读取 TXT、Markdown、PDF 和 DOCX 文档。
- 按固定字符长度切分文档，并保留少量重叠内容。
- 根据英文关键词重合数量检索相关片段。
- 从终端传入文档路径和问题。
- 使用检索片段生成回答，并显示来源文件和片段编号。
- 使用离线自动化测试验证核心行为。

## 环境准备

本项目使用 Python 虚拟环境。在 PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 配置 DeepSeek

在项目根目录创建 `.env` 文件：

```dotenv
DEEPSEEK_API_KEY=你的_API_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

`.env` 已被 `.gitignore` 排除，不要把 API Key 提交到 GitHub。

## 运行 Demo

读取 PDF：

```powershell
python .\rag_query.py .\sample.pdf "What technologies are mentioned?"
```

读取 DOCX 简历时，如果路径包含中文或空格，请使用引号：

```powershell
python .\rag_query.py "C:\path\to\resume.docx" "What skills are mentioned?"
```

程序会在终端显示检索片段、模型回答和引用来源。

## 运行测试

```powershell
python -m unittest discover -s tests
```

自动化测试不会调用 DeepSeek，也不会消耗 API 额度。

## 隐私说明

- DOCX 由 MarkItDown 在本地解析，不需要上传到外部转换网站。
- 自动格式转换不等于脱敏。
- 真实简历只用于本地测试，不应提交到公开 GitHub 仓库。
- 公开测试数据必须删除姓名、电话、邮箱、详细地址等个人信息。

## 当前限制

- 当前检索基于英文关键词重合，不是 Embedding 向量检索。
- 引用只能定位到文件和检索片段，还不能定位到 PDF 页码或原始段落。
- 当前 Demo 一次只读取一个文档。

