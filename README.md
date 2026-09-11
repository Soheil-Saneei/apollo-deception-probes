# Epistemic Deference

An interpretability experiment testing whether an LLM behaves differently when a user explicitly defers to its judgment.

The initial experiment pairs each question with neutral and deferential user framing, sends both variants to Qwen3.5-9B through Ollama, and stores the responses as JSONL. The included scorer reports simple descriptive measures. These are scaffolding metrics, not validated measures of epistemic behavior.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/)
- The Qwen3.5-9B model available locally as `qwen3.5:9b`

## Run

```sh
ollama pull qwen3.5:9b
python3 generate.py
python3 score.py outputs/run-YYYYMMDD-HHMMSS.jsonl
```

Use `python3 generate.py --help` to select another model, prompt file, output directory, or Ollama endpoint.

## Layout

- `prompts.json` contains paired experimental framing.
- `generate.py` runs the paired generations.
- `score.py` summarizes generated responses.
- `outputs/` contains generated data and is otherwise ignored by Git.

Before drawing conclusions, expand the prompt set, randomize condition order, repeat across seeds, pre-register outcome measures, and use a scorer blind to condition.
