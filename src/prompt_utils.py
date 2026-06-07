"""
Prompt utilities for LLM interactions.
"""


def system_prompt(role: str, context: str = "") -> dict:
    content = f"You are {role}."
    if context:
        content += f" {context}"
    return {"role": "system", "content": content}


def truncate_messages(messages: list, max_tokens: int = 4000) -> list:
    """Keep only recent messages within token budget."""
    result = []
    total = 0
    for msg in reversed(messages):
        tokens = len(msg["content"].split()) * 1.3
        if total + tokens > max_tokens:
            break
        result.insert(0, msg)
        total += tokens
    return result


def format_history(history: list) -> str:
    return "\n".join(
        f"[{m['role'].upper()}]: {m['content']}" for m in history
    )