# Tool Calling Dataset Processing Pipeline

一个用于工具调用数据集格式转换、质量评估和响应填充的完整处理管道。

## 项目简介

本项目提供了一套完整的工具来处理和评估工具调用数据集，主要功能包括：

- **格式转换**：将各种格式的工具调用数据集转换为统一的MCP（Model Context Protocol）格式
- **质量评估**：使用Gemini模型对转换后的数据集进行多维度质量评分
- **响应填充**：为转换后的数据生成占位符响应

## 目录结构

```
tool_calling/
├── converters/              # 转换器模块
│   ├── mcp_converter.py    # MCP格式转换器
│   └── uigen_converter.py  # UIGEN格式转换器
├── evaluators/              # 评估器模块
│   ├── quality_evaluator.py # 质量评估器
│   └── response_filler.py   # 响应填充器
├── config/                  # 配置文件
│   └── eval_guidelines.json # 评估指南
├── data/                    # 数据目录
│   ├── 1_raw/              # 原始输入数据
│   ├── 2_converted/        # 转换后数据
│   ├── 3_evaluated/        # 评估结果
│   └── 4_filled/           # 填充结果
├── prompts/                 # 提示词模板
│   └── conversion_guide.txt # 转换指南
├── .gitignore
└── README.md
```

## 安装依赖

```bash
pip install jsonschema tqdm orjson google-generativeai pyarrow requests datasets
```

## 使用方法

### 1. 格式转换

#### MCP格式转换器
将各种格式的数据集转换为MCP格式：

```bash
python converters/mcp_converter.py \
    --input data/1_raw/your_dataset.json \
    --output data/2_converted/ \
    --refine --score --model gemini-1.5-flash
```

#### UIGEN格式转换器
将UIGEN格式的对话数据转换为评估格式：

```bash
python converters/uigen_converter.py \
    --input data/1_raw/Tool-Calling-Dataset-UIGEN-X.jsonl
```

### 2. 质量评估

使用Gemini模型对数据集进行质量评估：

```bash
python evaluators/quality_evaluator.py \
    --sample data/1_raw/APIGen.json \
    --guideline config/eval_guidelines.json \
    --outdir data/3_evaluated/ \
    --model gemini-2.0-flash-exp
```

### 3. 响应填充

为转换后的数据生成响应：

```bash
python evaluators/response_filler.py \
    --input data/2_converted/Tool-Calling-Dataset-UIGEN-X.jsonl \
    --output data/4_filled/filled_pairs_gemini.jsonl \
    --model gemini-2.0-flash
```

## 配置说明

### API密钥配置

设置Google API密钥（以下方式任选其一）：

1. 环境变量：
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

2. 在代码中直接设置（不推荐用于生产环境）

### 评估指南

`config/eval_guidelines.json` 包含详细的评估标准，包括：
- 工具Schema完整性评分
- 工具链可行性评估
- 并行/串行/混合调用类型分析

## 示例用法

### 完整处理流程

1. **准备原始数据**：将数据集放入 `data/1_raw/` 目录

2. **格式转换**：
```bash
python converters/mcp_converter.py --input data/1_raw/APIGen.json --output data/2_converted/
```

3. **质量评估**：
```bash
python evaluators/quality_evaluator.py --sample data/1_raw/APIGen.json --outdir data/3_evaluated/
```

4. **响应填充**：
```bash
python evaluators/response_filler.py --input data/2_converted/APIGen.json --output data/4_filled/
```

### 批量处理

```bash
# 处理所有原始数据
for file in data/1_raw/*.json; do
    echo "Processing $file"
    python converters/mcp_converter.py --input "$file" --output data/2_converted/
    python evaluators/quality_evaluator.py --sample "$file" --outdir data/3_evaluated/
done
```

## 注意事项

1. **API限制**：使用Gemini API时请注意速率限制和配额
2. **大文件处理**：对于大型数据集，建议使用流式处理模式
3. **内存使用**：处理大文件时注意内存使用情况
4. **错误处理**：所有脚本都包含错误处理，失败时会输出详细错误信息
5. **数据备份**：建议在处理前备份重要数据

## 输出格式

### 评估结果格式

```json
{
  "InputSchemaScore": 95,
  "OutputSchemaScore": 10,
  "InputOutputSchemaDesc": "Schema完整性描述",
  "ChainPotential": 90,
  "ChainTypeRatio": {
    "parallel": 60,
    "sequential": 40,
    "hybrid": 0
  },
  "Coverage": 95,
  "DependencyAlignment": 90,
  "Redundancy": 90,
  "ExecutionFlow": 90,
  "reason": "整体质量评估说明"
}
```

### 转换后格式

```json
{
  "conversations": [
    { "from": "human", "value": "用户问题" },
    { "from": "function_call", "value": "{\"name\": \"tool_name\", \"arguments\": {...}}" },
    { "from": "observation", "value": "[{\"type\": \"tool_result\", \"name\": \"tool_name\", \"content\": \"...\"}]" },
    { "from": "gpt", "value": "助手回复" }
  ]
}
```

## 故障排除

1. **API密钥错误**：确保正确设置了GOOGLE_API_KEY环境变量
2. **文件路径错误**：检查输入文件路径是否正确
3. **依赖缺失**：运行 `pip install -r requirements.txt` 安装所有依赖
4. **内存不足**：对于大文件，考虑使用流式处理或增加系统内存

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证。
