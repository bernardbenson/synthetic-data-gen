# Synthetic Data Generation for TDAMM Classification

This project generates synthetic training data for Time-Domain and Multi-Messenger Astronomy (TDAMM) multi-label classification tasks.

## Two Approaches

### 1. Template-Based Generation (`generate_synthetic_data.py`)
- Uses predefined templates with realistic astronomical terminology
- Fast and consistent generation
- No API costs
- Good for large-scale data generation

### 2. LLM-Based Generation (`generate_synthetic_data_llm.py`)
- Uses OpenAI GPT models for natural language generation
- More diverse and sophisticated content
- Requires OpenAI API key
- Better for high-quality, varied training data

## Setup

### Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### For LLM Approach - OpenAI API Key
1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Copy `.env.example` to `.env` and add your key:
   ```bash
   cp .env.example .env
   # Edit .env with your API key
   ```
3. Or set environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Template-Based Generation
```bash
# Default: 200 samples, up to 3 labels each
python3 generate_synthetic_data.py

# Custom parameters
python3 generate_synthetic_data.py --max-labels 5 --num-samples 100

# Single-label classification
python3 generate_synthetic_data.py --max-labels 1 --num-samples 500
```

### LLM-Based Generation
```bash
# Default: 200 samples, 2-3 labels each (Non-TDAMM: 1 label, TDAMM: 2-3 labels)
uv run python generate_synthetic_data_llm.py

# Custom label range (2-5 labels for TDAMM classes)
uv run python generate_synthetic_data_llm.py --min-labels 2 --max-labels 5 --num-samples 50

# Use specific model
uv run python generate_synthetic_data_llm.py --model gpt-4 --num-samples 100

# Generate specific number of samples per class (multi-label mode)
uv run python generate_synthetic_data_llm.py --samples-per-class 5

# Combine model and per-class generation
uv run python generate_synthetic_data_llm.py --model gpt-4o --samples-per-class 10 --min-labels 3 --max-labels 4

# Control concurrency for API rate limits
uv run python generate_synthetic_data_llm.py --max-concurrent 5 --num-samples 100
```

## Classification Guidelines

Both approaches follow the same classification rules based on `data/default_prompt.txt`:

### TDAMM Labels (2-5 per sample)
- **Electromagnetic**: X-rays, Gamma rays, Optical, Radio, etc.
- **Gravitational Waves**: Binary mergers, continuous waves, etc.
- **Cosmic Messengers**: Cosmic rays, Neutrinos
- **Astronomical Objects**: Black holes, Neutron stars, etc.
- **Transient Phenomena**: Supernovae, Gamma-ray bursts, etc.

**Note**: All TDAMM samples must have 2-5 labels (configurable via `--min-labels` and `--max-labels`)

### Non-TDAMM (Always single label)
- Pure heliophysics/solar science
- Earth and planetary science
- Biographical content (non-astrophysics)
- Technical/administrative content
- Earth-based atmospheric science
- News announcements

**Note**: Non-TDAMM samples are always single-label and mutually exclusive with TDAMM classes

## Key Features

- **Multi-label classification**: TDAMM samples have 2-5 labels, Non-TDAMM always has 1 label
- **Configurable label range**: Use `--min-labels` and `--max-labels` to control TDAMM label count
- **5-label maximum**: Prevents model confusion while ensuring substantial content coverage
- **>30% content rule**: Each label gets substantial coverage in the generated text
- **Non-TDAMM exclusivity**: Non-TDAMM is mutually exclusive with TDAMM labels
- **Word count targeting**: ~1,178 words to match validation data
- **Async processing**: Fast concurrent generation with configurable rate limiting
- **Timestamped outputs**: Unique filenames for each run

## Output Format

```json
[
  {
    "link": "https://synthetic-data-gen.example.com/sample_1",
    "full_text": "Detailed scientific content...",
    "labels": ["X-rays", "Black Holes", "Neutron Stars", "Gravitational Waves"]
  }
]
```

## Cost Estimation (LLM Approach)

The script provides automatic cost estimation based on the selected model:

- **gpt-4o-mini**: ~$0.002 per 1K tokens (recommended for cost-effective generation)
- **gpt-4o**: ~$0.01 per 1K tokens  
- **gpt-4**: ~$0.03 per 1K tokens (highest quality)
- **gpt-4-turbo**: ~$0.01 per 1K tokens
- **gpt-3.5-turbo**: ~$0.002 per 1K tokens

### Sample Cost Examples:
- **200 samples with gpt-4o-mini**: ~$2-5
- **40 classes × 5 samples each with gpt-4o-mini**: ~$4-8
- **100 samples with gpt-4**: ~$15-25

Use `--samples-per-class`, `--model`, and `--max-concurrent` to control costs and rate limits effectively.

## Command Line Options

### LLM-Based Generation Options
- `--min-labels`: Minimum labels per TDAMM sample (default: 2, range: 2-5)
- `--max-labels`: Maximum labels per TDAMM sample (default: 3, range: 2-5)  
- `--num-samples`: Total number of samples to generate (default: 200)
- `--samples-per-class`: Generate N samples per class (overrides --num-samples)
- `--model`: OpenAI model to use (default: gpt-4o-mini)
- `--max-concurrent`: Maximum concurrent API requests (default: 10)
- `--output-dir`: Output directory (default: data)

## Files Generated

- `data/synthetic_training_data_YYYYMMDD_HHMMSS.json` (Template-based)
- `data/synthetic_training_data_llm_YYYYMMDD_HHMMSS.json` (LLM-based)