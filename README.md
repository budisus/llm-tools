# llm-tools

Collection of utilities for working with LLMs and AI APIs.

```bash
pip install llm-tools
```

## Features

- Unified client for OpenAI and Anthropic APIs
- Prompt engineering utilities
- Token estimation and truncation
- Response parsing helpers

## Usage

```python
from llm_tools import get_client

# OpenAI
client = get_client("openai", model="gpt-4")
resp = client.chat([{"role": "user", "content": "Hello!"}])
print(resp.content)
```

## Prompt Utils

```python
from llm_tools.prompt_utils import system_prompt, few_shot_prompt

# System prompt
msgs = [system_prompt("You are a helpful assistant.", ["Be concise"])]

# Few-shot
msgs = few_shot_prompt(
    task="Translate to French",
    examples=[{"input": "Hello", "output": "Bonjour"}],
    query="Goodbye"
)
```

## License

MIT © 2026