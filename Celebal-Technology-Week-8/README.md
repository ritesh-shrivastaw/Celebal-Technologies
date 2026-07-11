# Smart Agent

A short quiz and small agent pipeline project built in Python. It combines:

- a lightweight agent-style pipeline with intent classification and tool routing
- a mini quiz mode with scoring and feedback
- a few simple assistant tools for calculation, keyword extraction, unit conversion, and summarisation

## Run the project

From the project folder, run:

```bash
python smart_agent.py
```

When you start it, you can choose between:
1. Agent chat mode
2. Short quiz mode

## Example inputs

- `Calculate 20 + 5`
- `Extract keywords from Artificial Intelligence is transforming industries`
- `Convert 100 km to miles`
- `Summarise: Machine learning is a type of AI. It allows computers to learn from data.`
- `What is machine learning?`

## Files

- `smart_agent.py` — main assistant implementation and quiz flow
- `test_smart_agent.py` — regression tests for the agent behavior and quiz mode
