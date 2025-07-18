# Synthetic Data Generation for TDAMM Classification

This project generates synthetic training data for Time-Domain and Multi-Messenger Astronomy (TDAMM) multi-label classification tasks using OpenAI GPT models. It features intelligent label distribution balancing, asynchronous batch processing, and comprehensive cost estimation.

## Setup

### Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### OpenAI API Key
1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add API Key to `.env` and add your key:
3. Or set environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage
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

The generation follows classification rules based on `data/default_prompt.txt`:

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

### Core Functionality
- **Multi-label classification**: TDAMM samples have 2-5 labels, Non-TDAMM always has 1 label
- **Intelligent label distribution**: Automatic balancing ensures all classes get adequate representation
- **Configurable label range**: Use `--min-labels` and `--max-labels` to control TDAMM label count (2-5)
- **Two generation modes**: Total samples mode or samples-per-class mode
- **>30% content rule**: Each label gets substantial coverage in the generated text (enforced by prompts)
- **Word count targeting**: ~1,178 words to match validation data
- **Non-TDAMM exclusivity**: Non-TDAMM is mutually exclusive with TDAMM labels

### Performance & Reliability
- **Asynchronous batch processing**: Concurrent API calls with configurable batch sizes
- **Intelligent rate limiting**: Exponential backoff with jitter for API rate limits
- **Retry logic**: Up to 3 retries with progressive delays for failed requests
- **Error handling**: Graceful failure handling that continues processing other samples
- **Progress tracking**: Real-time progress updates for batch processing

### Content Generation
- **Specialized prompts**: Separate prompt engineering for TDAMM vs Non-TDAMM content
- **Realistic scientific content**: Uses actual astronomical terminology, instruments, and measurements
- **Label-specific requirements**: Each label type has specific content requirements (observational data, energy ranges, etc.)
- **Balanced selection algorithm**: Weighted sampling favors underrepresented classes
- **Quality validation**: Word count warnings for samples below target length

### Cost Management
- **Automatic cost estimation**: Real-time cost calculations based on model pricing
- **Model flexibility**: Support for all major OpenAI models (GPT-4, GPT-4o, GPT-4-turbo, GPT-3.5-turbo, GPT-4o-mini)
- **Batch size control**: Adjustable concurrency to manage rate limits and costs
- **Detailed statistics**: Post-generation analysis of label distribution and word counts

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

## Advanced Features

### Intelligent Label Distribution
The system automatically calculates target distributions to ensure balanced representation:
- **Non-TDAMM allocation**: Gets proportional share (1/total_classes of samples)
- **TDAMM distribution**: Remaining samples distributed evenly among TDAMM classes
- **Multi-label accounting**: Accounts for average 2.5 labels per TDAMM sample
- **Weighted selection**: Algorithm favors underrepresented classes during generation

### Balanced Selection Algorithm
For each sample generation, the system:
1. Calculates current vs target ratios for all classes
2. Prioritizes classes with largest deficits
3. Uses weighted random sampling favoring underrepresented classes
4. Updates tracking counters to maintain balance throughout generation

### Content Quality Controls
- **Label coverage validation**: Each label must comprise >30% of substantial content
- **Word count targeting**: Aims for ~1,178 words (median of validation data)
- **Scientific accuracy**: Uses realistic astronomical measurements and terminology
- **Instrument specificity**: References actual telescopes (Chandra, LIGO, IceCube, etc.)

### Error Handling & Resilience
- **Rate limit management**: Exponential backoff (2^attempt + random jitter)
- **Retry logic**: Up to 3 attempts with progressive delays
- **Graceful degradation**: Failed samples don't stop batch processing
- **Exception isolation**: Individual sample failures don't affect others

### Batch Processing Architecture
- **Configurable batch sizes**: Default 20 samples per batch (adjustable)
- **Concurrent execution**: Async/await pattern for parallel API calls
- **Progress tracking**: Real-time batch completion updates
- **Memory efficiency**: Processes samples in chunks to manage memory usage

## Cost Estimation

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

Use `--samples-per-class`, `--model`, and `--batch-size` to control costs and rate limits effectively.

## Command Line Options

### Generation Control
- `--min-labels`: Minimum labels per TDAMM sample (default: 2, range: 2-5)
- `--max-labels`: Maximum labels per TDAMM sample (default: 3, range: 2-5)  
- `--num-samples`: Total number of samples to generate (default: 200)
- `--samples-per-class`: Generate N samples per class (overrides --num-samples)

### Model & Performance
- `--model`: OpenAI model to use (default: gpt-4o-mini)
  - Supported: gpt-4, gpt-4o, gpt-4-turbo, gpt-3.5-turbo, gpt-4o-mini
- `--batch-size`: Samples per batch for concurrent processing (default: 20)

### Output
- `--output-dir`: Output directory (default: data)

## Technical Implementation Details

### Prompt Engineering
The system uses specialized prompts for different content types:

#### TDAMM Content Prompts
- **Label-specific requirements**: Each label type has detailed generation guidelines
- **Quantitative emphasis**: Requires specific measurements, energy ranges, flux values
- **Instrument integration**: References real telescopes and detectors (Chandra, XMM-Newton, Fermi, LIGO, Virgo, KAGRA, IceCube, Pierre Auger)
- **Content distribution**: Automatically calculates word allocation per label (1200 words ÷ number of labels)
- **Scientific accuracy**: Enforces proper astronomical terminology and realistic observations

#### Non-TDAMM Content Prompts  
- **Exclusion criteria**: Explicitly avoids astronomical content that could cause false positives
- **Focus areas**: Pure heliophysics, Earth science, biographical content, technical/administrative notices
- **Length requirements**: 1200+ words with clear non-astronomical focus

### Generation Modes

#### Total Samples Mode (`--num-samples`)
- Generates specified total number of samples
- Automatically balances label distribution across all samples
- Uses intelligent selection algorithm to maintain class balance
- Default: 200 samples

#### Samples Per Class Mode (`--samples-per-class`)
- Generates N samples for each class
- Ensures minimum representation for all classes
- For TDAMM classes: creates multi-label samples while guaranteeing each class appears
- Total samples = N × number of classes

### Performance Optimizations
- **Asynchronous processing**: Uses Python's asyncio for concurrent API calls
- **Batch management**: Processes samples in configurable batches (default: 20)
- **Rate limit handling**: Exponential backoff with jitter (2^attempt + random(0,1) seconds)
- **Memory efficiency**: Streams results rather than holding all in memory
- **Progress tracking**: Real-time updates on batch completion status

### Quality Assurance
- **Word count validation**: Warns when samples fall below 1,178 word target
- **Label coverage enforcement**: Prompts ensure >30% content coverage per label
- **Error isolation**: Individual sample failures don't affect batch processing
- **Statistical reporting**: Post-generation analysis of label distribution and word counts

### Cost Management Features
- **Real-time cost estimation**: Calculates costs based on token usage and model pricing
- **Model selection**: Supports all major OpenAI models with different cost/quality tradeoffs
- **Batch size control**: Adjustable concurrency to manage API rate limits
- **Usage statistics**: Detailed breakdown of token consumption and estimated costs

## Files Generated

- `data/synthetic_training_data_llm_YYYYMMDD_HHMMSS.json`

## Output Statistics

After generation, the script provides comprehensive statistics:

### Label Distribution Analysis
- Count of each label in generated samples
- Top 10 most frequent labels
- Balance assessment across all classes

### Word Count Statistics
- Average, median, min, and max word counts
- Comparison to target length (1,178 words)
- Warnings for samples below target threshold

### Cost Analysis
- Total estimated tokens consumed
- Cost breakdown by model pricing
- Recommendations for cost optimization

### Performance Metrics
- Total generation time
- Samples per second processing rate
- Batch processing efficiency
- Success rate (completed vs failed samples)