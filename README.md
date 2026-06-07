# llm-tools

Lightweight utilities for working with LLMs and AI APIs.

## Features

- Simple chat client wrapper (OpenAI-compatible)
- Prompt formatting helpers
- Message history truncation

## Install

```
pip install -r requirements.txt
```

## Usage

```python
from llm_client import LLMClient
from prompt_utils import system_prompt

client = LLMClient(model="gpt-4o-mini")

messages = [
    system_prompt("a helpful assistant"),
    {"role": "user", "content": "Explain embeddings in one paragraph."}
]

print(client.chat(messages))
```

## License

MIT