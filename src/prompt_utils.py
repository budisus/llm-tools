"""
Prompt Engineering Utilities for LLMs.
"""

from typing import List, Dict, Any, Optional, Callable
import json
import re


def system_prompt(role: str, constraints: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Create a system prompt with role and optional constraints.
    
    Args:
        role: The AI's role/persona
        constraints: List of behavioral constraints
    
    Returns:
        Message dict ready for LLM
    """
    content = role
    if constraints:
        content += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
    return {"role": "system", "content": content}


def few_shot_prompt(
    task: str,
    examples: List[Dict[str, str]],
    query: str
) -> List[Dict[str, str]]:
    """
    Create a few-shot prompting message list.
    
    Args:
        task: Description of the task
        examples: List of {"input": "...", "output": "..."} examples
        query: The actual query to get a response for
    
    Returns:
        List of message dicts
    """
    messages = [
        {"role": "system", "content": f"Task: {task}\n\nFollow the example format exactly."}
    ]
    
    for ex in examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    
    messages.append({"role": "user", "content": query})
    return messages


def chain_of_thought(
    question: str,
    include_answer: bool = True
) -> List[Dict[str, str]]:
    """
    Create a chain-of-thought prompting prompt.
    
    Args:
        question: The question to reason through
        include_answer: Whether to ask for final answer at end
    
    Returns:
        Message list with CoT prompting
    """
    cot_template = """Think through this step by step:

Question: {question}

Take deep breath and work through this systematically. Show your reasoning between <thinking> tags, then provide your answer in <answer> tags."""

    if include_answer:
        cot_template += "\n\nFormat your response with <thinking> for reasoning and <answer> for the final answer."
    
    return [{"role": "user", "content": cot_template.format(question=question)}]


def extract_between(text: str, start: str, end: str) -> Optional[str]:
    """Extract content between two delimiters."""
    pattern = re.escape(start) + r"(.*?)" + re.escape(end)
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_xml_tags(text: str, tag: str) -> List[str]:
    """Extract all instances of <tag>content</tag> from text."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return re.findall(pattern, text, re.DOTALL)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text that may contain other content."""
    # Try to find JSON between code blocks
    json_match = re.search(r'```(?:json)?\s*({\s*.*})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def count_tokens_estimate(text: str, model: str = "gpt-4") -> int:
    """
    Rough token estimate using word/char ratios.
    Not exact, but useful for budgeting.
    
    Approximate ratios:
    - GPT-4: ~2.5 chars/token (English)
    - Claude: ~3.5 chars/token
    """
    if "claude" in model.lower():
        return len(text) // 3
    return len(text) // 2


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit."""
    ratio = 3 if "claude" in model.lower() else 2
    max_chars = max_tokens * ratio
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


class PromptTemplate:
    """Template-based prompt builder with variable substitution."""
    
    def __init__(self, template: str):
        self.template = template
        self.variables = set(re.findall(r'\{(\w+)\}', template))
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        missing = self.variables - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        
        return self.template.format(**kwargs)
    
    def __repr__(self) -> str:
        return f"PromptTemplate({len(self.variables)} variables)"


# Pre-built templates
CODE_REVIEW_TEMPLATE = PromptTemplate(
    """You are a senior code reviewer. Review the following code and provide feedback.

Language: {language}
Code:
```{language}
{code}
```

Focus on:
1. Bug potential and edge cases
2. Performance concerns
3. Security vulnerabilities
4. Code readability and style

Provide your review in this format:
<strengths>
- List of positive aspects
</strengths>
<issues>
- List of issues found with severity (HIGH/MEDIUM/LOW)
</issues>
<recommendations>
- Specific suggestions for improvement
</recommendations>"""
)

SUMMARIZE_TEMPLATE = PromptTemplate(
    """Summarize the following text in {style} style.

Text:
{text}

Requirements:
- Maximum {max_length} words
- Key points only
- Use bullet points for multiple items"""
)


def create_react_prompt(question: str, tools: List[str]) -> List[Dict[str, str]]:
    """
    Create a ReAct (Reasoning + Acting) style prompt with tools.
    
    Args:
        question: The question to answer
        tools: List of available tool descriptions
    
    Returns:
        Message list with ReAct system prompt
    """
    tools_text = "\n".join(f"- {t}" for t in tools)
    
    return [
        {"role": "system", "content": f"""You are a helpful assistant that uses tools to answer questions.

Available tools:
{tools_text}

Follow this format exactly:
Question: the question to answer
Thought: your reasoning about what to do
Action: the tool to use (if needed)
Action Input: the input for the tool
Observation: the result from the tool
... (repeat as needed)
Thought: I now know the final answer
Answer: your final response"""},
        {"role": "user", "content": f"Question: {question}"}
    ]