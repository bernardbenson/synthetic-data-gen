#!/usr/bin/env python3
"""
Generate synthetic multi-label astronomical data using OpenAI LLM.

This script generates synthetic training data for TDAMM (Time-Domain and Multi-Messenger Astronomy)
classification using OpenAI's GPT models while following the same classification guidelines
as the template-based approach.
"""

import json
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time
import asyncio

try:
    from openai import AsyncOpenAI
except ImportError:
    print("Error: OpenAI library not found. Install with: pip install openai")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, will still work with environment variables
    pass

def load_classes(classes_file: str) -> Dict[str, str]:
    """Load class definitions from JSON file."""
    with open(classes_file, 'r') as f:
        return json.load(f)

def load_prompt(prompt_file: str) -> str:
    """Load the classification prompt from file."""
    with open(prompt_file, 'r') as f:
        return f.read().strip()

def load_sample_data(data_file: str) -> List[Dict[str, Any]]:
    """Load sample data to understand structure and content."""
    with open(data_file, 'r') as f:
        return json.load(f)

def create_non_tdamm_prompt() -> str:
    """Create a prompt for generating Non-TDAMM content."""
    return """Generate content that is NOT about time-domain and multi-messenger astronomy (Non-TDAMM).

CRITICAL: This content must be clearly NON-TDAMM to avoid false positives. 

Based on classification guidelines, create content that falls into one of these categories:
- Pure heliophysics/solar science WITHOUT any extragalactic or stellar astrophysics context
  * Solar wind measurements, coronal mass ejections, sunspot analysis
  * Solar magnetic field studies, solar atmosphere research
  * Heliospheric physics, solar particle events
- Personal biographical content for non-astrophysics persons
  * Professors in chemistry, biology, computer science, engineering
  * Academic career highlights, teaching awards, research in non-astronomy fields
- Website maintenance notices, broken pages, or pure technical/administrative content
  * Server maintenance announcements, broken link notifications
  * Database updates, website redesign notices
  * IT infrastructure reports, system outages
- Content exclusively about Earth-based phenomena or atmospheric science
  * Weather patterns, climate studies, atmospheric composition
  * Meteorology, Earth's magnetic field, ionospheric research
- Brief news items or announcements without detailed scientific analysis
  * Personnel changes, workshop announcements, funding news
  * University administrative updates, policy changes

Requirements:
- 1200+ words
- Should NOT contain ANY substantial discussion of:
  * Black holes, neutron stars, or other compact objects
  * Gravitational waves or multi-messenger astronomy
  * Gamma-ray bursts, supernovae, or other transient phenomena
  * Extragalactic or stellar astrophysics
  * Time-domain astronomy or transient events
  * Multi-wavelength astronomy beyond Earth/solar system
  
Write as if it's from a university website, press release, or administrative notice.
Focus on making it clearly non-astronomical or purely solar/Earth-based science."""

def create_tdamm_prompt(selected_classes: List[str]) -> str:
    """Create a prompt for generating TDAMM content with specific labels."""
    
    classes_str = ", ".join(selected_classes)
    
    return f"""Generate scientific content about time-domain and multi-messenger astronomy (TDAMM) that would be classified with these labels: {classes_str}

CRITICAL REQUIREMENTS (based on classification guidelines):
- Each label must comprise >30% of the substantial content (not just brief mentions)
- Require at least 2-3 sentences of dedicated discussion for each label
- Include specific observational data, research findings, or detailed scientific analysis
- False negatives are better than false positives - be substantial and specific

CONTENT STRUCTURE - Ensure each label gets dedicated coverage:
For {len(selected_classes)} labels, dedicate approximately {1200 // len(selected_classes)} words per label.

For each assigned label, provide SUBSTANTIAL discussion including:
- Electromagnetic spectrum labels (X-rays, Gamma rays, Optical, Radio, etc.): 
  * Actual observational data with specific flux measurements, detector descriptions, energy ranges
  * Telescope/instrument names (Chandra, XMM-Newton, Fermi, etc.)
  * Specific photon energies, spectral indices, luminosities
- Gravitational wave labels: 
  * Waveform analysis details, detector data from LIGO/Virgo/KAGRA
  * Parameter estimation (masses, spins, distances)
  * Strain measurements, frequency evolution
- Cosmic ray/neutrino labels: 
  * Particle physics data, energy measurements in TeV/PeV ranges
  * Arrival directions, cosmic ray composition
  * Detector specifics (IceCube, Pierre Auger, etc.)
- Astronomical object labels (Black Holes, Neutron Stars, etc.): 
  * Formation mechanisms, physical properties, mass measurements
  * Specific observations and discoveries
  * Orbital parameters for binaries
- Transient phenomenon labels (Supernovae, Gamma-ray Bursts, etc.): 
  * Detailed descriptions, light curves, spectral evolution
  * Time scales, peak luminosities, decay rates
  * Multi-wavelength observations

Requirements:
- 1200+ words total
- Scientific accuracy with realistic measurements and citations
- Substantial dedicated sections for each label: {classes_str}
- Write as if from a research paper, GCN circular, or technical report
- Use proper astronomical terminology and specific instruments/telescopes
- Include quantitative measurements and observational details
- Each topic must have enough content to warrant >30% classification

Focus on ensuring each of these topics gets substantial, detailed coverage: {classes_str}"""

async def generate_text_with_openai(prompt: str, model: str = "gpt-4o-mini", max_retries: int = 3, client: AsyncOpenAI = None) -> str:
    """Generate text using OpenAI's API with retry logic (async version)."""
    
    if client is None:
        raise ValueError("OpenAI client must be provided")
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert astrophysicist specializing in time-domain and multi-messenger astronomy. Generate realistic, scientifically accurate content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.8,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit. Waiting {wait_time:.1f} seconds before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            else:
                if attempt < max_retries - 1:
                    print(f"Error generating text (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(1)
                else:
                    raise

async def generate_single_sample(
    class_name: str,
    sample_id: int,
    classes: Dict[str, str],
    model: str,
    client: AsyncOpenAI
) -> Dict[str, Any]:
    """Generate a single synthetic sample (async)."""
    try:
        if class_name == "Non-TDAMM":
            selected_class_names = ["Non-TDAMM"]
            generation_prompt = create_non_tdamm_prompt()
        else:
            selected_class_names = [class_name]
            generation_prompt = create_tdamm_prompt(selected_class_names)
        
        print(f"Generating sample {sample_id} for class '{class_name}'")
        
        # Generate synthetic text using OpenAI
        synthetic_text = await generate_text_with_openai(generation_prompt, model, client=client)
        
        # Ensure minimum word count
        word_count = len(synthetic_text.split())
        if word_count < 1178:
            print(f"  Warning: Sample {sample_id} is {word_count} words, below target of 1178")
        
        # Create sample in the same format as original data
        sample = {
            "link": f"https://synthetic-data-gen.example.com/sample_{sample_id}",
            "full_text": synthetic_text,
            "labels": selected_class_names
        }
        
        return sample
        
    except Exception as e:
        print(f"Error generating sample {sample_id} for class '{class_name}': {e}")
        return None

async def generate_mixed_sample(
    sample_id: int,
    selected_class_names: List[str],
    model: str,
    client: AsyncOpenAI
) -> Dict[str, Any]:
    """Generate a single mixed-label synthetic sample (async)."""
    try:
        if "Non-TDAMM" in selected_class_names:
            generation_prompt = create_non_tdamm_prompt()
        else:
            generation_prompt = create_tdamm_prompt(selected_class_names)
        
        print(f"Generating sample {sample_id} with labels: {selected_class_names}")
        
        # Generate synthetic text using OpenAI
        synthetic_text = await generate_text_with_openai(generation_prompt, model, client=client)
        
        # Ensure minimum word count
        word_count = len(synthetic_text.split())
        if word_count < 1178:
            print(f"  Warning: Sample {sample_id} is {word_count} words, below target of 1178")
        
        # Create sample in the same format as original data
        sample = {
            "link": f"https://synthetic-data-gen.example.com/sample_{sample_id}",
            "full_text": synthetic_text,
            "labels": selected_class_names
        }
        
        return sample
        
    except Exception as e:
        print(f"Error generating sample {sample_id}: {e}")
        return None

async def process_batch(tasks: List, batch_num: int, total_batches: int) -> List[Dict[str, Any]]:
    """Process a batch of tasks concurrently."""
    print(f"Processing batch {batch_num}/{total_batches} ({len(tasks)} samples)")
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results and exceptions
    batch_results = [
        sample for sample in results 
        if sample is not None and not isinstance(sample, Exception)
    ]
    
    print(f"Completed batch {batch_num}/{total_batches}: {len(batch_results)} successful samples")
    return batch_results

def calculate_target_distributions(classes: Dict[str, str], total_samples: int) -> Dict[str, int]:
    """Calculate target label counts for balanced distribution."""
    all_class_names = list(classes.values())
    tdamm_classes = [c for c in all_class_names if c != "Non-TDAMM"]
    
    # For Non-TDAMM: proportional to total class count (1/total_classes of samples)
    non_tdamm_target = max(1, total_samples // len(all_class_names))
    
    # For TDAMM classes: distribute remaining samples evenly
    remaining_samples = total_samples - non_tdamm_target
    # Account for multi-label: each TDAMM sample will have 2-3 labels on average
    avg_labels_per_sample = 2.5
    tdamm_label_instances = remaining_samples * avg_labels_per_sample
    tdamm_target_per_class = max(1, int(tdamm_label_instances // len(tdamm_classes)))
    
    target_dist = {}
    for class_name in all_class_names:
        if class_name == "Non-TDAMM":
            target_dist[class_name] = non_tdamm_target
        else:
            target_dist[class_name] = tdamm_target_per_class
    
    return target_dist

def select_balanced_labels(
    classes: Dict[str, str], 
    current_counts: Dict[str, int], 
    target_counts: Dict[str, int],
    min_labels: int = 2,
    max_labels: int = 3
) -> List[str]:
    """Select labels to maintain balanced distribution."""
    all_class_names = list(classes.values())
    tdamm_classes = [c for c in all_class_names if c != "Non-TDAMM"]
    
    # Check if we should generate Non-TDAMM
    non_tdamm_ratio = current_counts.get("Non-TDAMM", 0) / max(1, target_counts.get("Non-TDAMM", 1))
    total_tdamm_count = sum(current_counts.get(c, 0) for c in tdamm_classes)
    total_tdamm_target = sum(target_counts.get(c, 0) for c in tdamm_classes)
    tdamm_ratio = total_tdamm_count / max(1, total_tdamm_target)
    
    # Generate Non-TDAMM if it's significantly behind
    if non_tdamm_ratio < 0.8 and (tdamm_ratio > non_tdamm_ratio + 0.2):
        return ["Non-TDAMM"]
    
    # Otherwise generate TDAMM with balanced selection
    # Calculate how far behind each TDAMM class is
    class_priorities = []
    for class_name in tdamm_classes:
        current = current_counts.get(class_name, 0)
        target = target_counts.get(class_name, 1)
        deficit = max(0, target - current)
        ratio = current / max(1, target)
        class_priorities.append((deficit, 1 - ratio, class_name))
    
    # Sort by deficit first, then by ratio (most behind first)
    class_priorities.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Select labels favoring classes that are behind
    num_labels = random.randint(min_labels, max_labels)
    selected_classes = []
    
    # Use weighted selection favoring classes that are behind
    available_classes = tdamm_classes.copy()
    weights = []
    
    for class_name in available_classes:
        current = current_counts.get(class_name, 0)
        target = target_counts.get(class_name, 1)
        # Weight inversely proportional to how close to target we are
        weight = max(0.1, target - current + 1)
        weights.append(weight)
    
    # Select classes with weighted random sampling
    for _ in range(num_labels):
        if not available_classes:
            break
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight <= 0:
            selected_class = random.choice(available_classes)
        else:
            rand_val = random.uniform(0, total_weight)
            cumulative = 0
            selected_class = available_classes[-1]  # fallback
            
            for i, weight in enumerate(weights):
                cumulative += weight
                if rand_val <= cumulative:
                    selected_class = available_classes[i]
                    break
        
        selected_classes.append(selected_class)
        # Remove selected class to avoid duplicates
        idx = available_classes.index(selected_class)
        available_classes.pop(idx)
        weights.pop(idx)
    
    return selected_classes

async def generate_synthetic_samples(
    classes: Dict[str, str],
    prompt: str,
    sample_data: List[Dict[str, Any]],
    num_samples: int = 100,
    max_labels_per_sample: int = 3,
    min_labels_per_sample: int = 2,
    samples_per_class: int = None,
    model: str = "gpt-4o-mini",
    client: AsyncOpenAI = None,
    batch_size: int = 20
) -> List[Dict[str, Any]]:
    """Generate synthetic astronomical text samples using OpenAI (batch processing version)."""
    
    if client is None:
        raise ValueError("AsyncOpenAI client must be provided")
    
    synthetic_samples = []
    class_keys = list(classes.keys())
    
    # Handle samples per class logic - now generates multi-label samples with balanced distribution
    if samples_per_class is not None:
        print(f"Generating {samples_per_class} multi-label samples per class mode (balanced batch processing)")
        all_class_names = list(classes.values())
        total_samples = samples_per_class * len(classes)
        
        # Calculate target distributions for balanced generation
        target_counts = calculate_target_distributions(classes, total_samples)
        current_counts = {class_name: 0 for class_name in classes.values()}
        
        # Adjust targets for samples-per-class mode: ensure each class gets at least samples_per_class
        for class_name in all_class_names:
            target_counts[class_name] = max(target_counts[class_name], samples_per_class)
        
        print("Target label distributions for balanced samples-per-class generation:")
        for class_name, target in target_counts.items():
            print(f"  {class_name}: {target}")
        
        # Create tasks for all samples using balanced selection
        all_tasks = []
        sample_id = 1
        for primary_class in all_class_names:
            for i in range(samples_per_class):
                if primary_class == "Non-TDAMM":
                    # Non-TDAMM is always single label
                    selected_class_names = ["Non-TDAMM"]
                else:
                    # For TDAMM classes, use balanced selection but ensure primary class is included
                    selected_class_names = select_balanced_labels(
                        classes, current_counts, target_counts, 
                        min_labels_per_sample, max_labels_per_sample
                    )
                    
                    # Ensure primary class is included if it's not Non-TDAMM
                    if primary_class not in selected_class_names and primary_class != "Non-TDAMM":
                        if len(selected_class_names) >= max_labels_per_sample:
                            # Replace one random class with primary class
                            selected_class_names[random.randint(0, len(selected_class_names)-1)] = primary_class
                        else:
                            # Add primary class
                            selected_class_names.append(primary_class)
                
                # Update current counts
                for class_name in selected_class_names:
                    current_counts[class_name] += 1
                
                task = generate_mixed_sample(
                    sample_id=sample_id,
                    selected_class_names=selected_class_names,
                    model=model,
                    client=client
                )
                all_tasks.append(task)
                sample_id += 1
        
        # Process tasks in batches
        total_batches = (len(all_tasks) + batch_size - 1) // batch_size
        print(f"Processing {len(all_tasks)} samples in {total_batches} batches of {batch_size}")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(all_tasks))
            batch_tasks = all_tasks[start_idx:end_idx]
            
            batch_results = await process_batch(batch_tasks, batch_num + 1, total_batches)
            synthetic_samples.extend(batch_results)
        
        print(f"Completed {len(synthetic_samples)} multi-label samples successfully")
    
    else:
        # Balanced distribution sampling logic (batch processing)
        print(f"Generating {num_samples} samples with {min_labels_per_sample}-{max_labels_per_sample} labels each (balanced distribution)")
        
        # Calculate target distributions for balanced generation
        target_counts = calculate_target_distributions(classes, num_samples)
        current_counts = {class_name: 0 for class_name in classes.values()}
        
        print("Target label distributions for balanced generation:")
        for class_name, target in target_counts.items():
            print(f"  {class_name}: {target}")
        
        all_tasks = []
        for i in range(num_samples):
            # Use balanced selection to choose labels
            selected_class_names = select_balanced_labels(
                classes, current_counts, target_counts, 
                min_labels_per_sample, max_labels_per_sample
            )
            
            # Update current counts to track progress
            for class_name in selected_class_names:
                current_counts[class_name] += 1
            
            task = generate_mixed_sample(
                sample_id=i+1,
                selected_class_names=selected_class_names,
                model=model,
                client=client
            )
            all_tasks.append(task)
        
        # Process tasks in batches
        total_batches = (len(all_tasks) + batch_size - 1) // batch_size
        print(f"Processing {num_samples} samples in {total_batches} batches of {batch_size}")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(all_tasks))
            batch_tasks = all_tasks[start_idx:end_idx]
            
            batch_results = await process_batch(batch_tasks, batch_num + 1, total_batches)
            synthetic_samples.extend(batch_results)
        
        print(f"Completed {len(synthetic_samples)} samples successfully")
    
    return synthetic_samples

async def main():
    """Main execution function (async version)."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-label astronomical training data using OpenAI LLM (async version)"
    )
    parser.add_argument(
        "--max-labels", 
        type=int, 
        default=3, 
        help="Maximum number of labels per sample (default: 3, range: 2-5)"
    )
    parser.add_argument(
        "--min-labels", 
        type=int, 
        default=2, 
        help="Minimum number of labels per sample (default: 2, range: 2-5)"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=200, 
        help="Number of synthetic samples to generate (default: 200)"
    )
    parser.add_argument(
        "--samples-per-class", 
        type=int, 
        help="Number of samples to generate per class (overrides --num-samples)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data", 
        help="Output directory for generated data (default: data)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o-mini", 
        help="OpenAI model to use (default: gpt-4o-mini, options: gpt-4, gpt-4-turbo, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=20, 
        help="Number of samples to process in each batch (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_labels < 2:
        print("Error: --min-labels must be at least 2")
        return
    if args.max_labels > 5:
        print("Error: --max-labels cannot exceed 5 (to avoid confusing the model with too many classes)")
        print("Based on your classification guidelines, each label should comprise >30% of substantial content")
        return
    if args.min_labels > args.max_labels:
        print("Error: --min-labels cannot be greater than --max-labels")
        return
    if args.num_samples < 1:
        print("Error: --num-samples must be at least 1")
        return
    if args.batch_size < 1:
        print("Error: --batch-size must be at least 1")
        return
    
    # Setup OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable in .env file")
        return
    
    client = AsyncOpenAI(api_key=api_key)
    
    # File paths
    classes_file = "data/classes.txt"
    prompt_file = "data/default_prompt.txt"
    sample_data_file = "data/stratified_split_val.json"
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_dir}/synthetic_training_data_llm_{timestamp}.json"
    
    # Load existing data
    print("Loading existing data...")
    classes = load_classes(classes_file)
    prompt = load_prompt(prompt_file)
    sample_data = load_sample_data(sample_data_file)
    
    print(f"Loaded {len(classes)} classes")
    print(f"Loaded {len(sample_data)} sample records")
    print(f"Using OpenAI model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    
    # Validate max_labels doesn't exceed available classes (but respect the 5 label limit)
    # For multi-class scenario, we need to ensure we have enough classes for the minimum labels
    available_classes = len(classes)
    max_allowed = min(5, available_classes)
    
    if args.max_labels > max_allowed:
        print(f"Warning: --max-labels ({args.max_labels}) exceeds limit")
        print(f"Setting max_labels to {max_allowed} (respecting 5-label limit and available classes)")
        args.max_labels = max_allowed
    
    if args.min_labels > available_classes:
        print(f"Error: --min-labels ({args.min_labels}) exceeds available classes ({available_classes})")
        return
    
    # Special handling for Non-TDAMM: if it exists, we need enough TDAMM classes for multi-label
    has_non_tdamm = any(class_name == "Non-TDAMM" for class_name in classes.values())
    tdamm_classes = len(classes) - (1 if has_non_tdamm else 0)
    
    if has_non_tdamm and args.min_labels > tdamm_classes:
        print(f"Error: With Non-TDAMM class present, need at least {args.min_labels} TDAMM classes for multi-label samples")
        print(f"Available TDAMM classes: {tdamm_classes}")
        print(f"Note: Non-TDAMM samples will have 1 label, TDAMM samples need {args.min_labels}-{args.max_labels} labels")
        return
    
    # Determine generation mode and validate parameters
    if args.samples_per_class is not None:
        if args.samples_per_class < 1:
            print("Error: --samples-per-class must be at least 1")
            return
        total_samples = args.samples_per_class * len(classes)
        print(f"Generating {args.samples_per_class} samples per class ({len(classes)} classes = {total_samples} total samples)")
        print(f"Note: Non-TDAMM will have single label, TDAMM classes will have {args.min_labels}-{args.max_labels} labels per sample")
    else:
        total_samples = args.num_samples
        print(f"Generating {args.num_samples} synthetic samples (Non-TDAMM: 1 label, TDAMM: {args.min_labels}-{args.max_labels} labels)...")
    
    print("Using batch processing for efficient generation...")
    
    start_time = time.time()
    synthetic_samples = await generate_synthetic_samples(
        classes=classes,
        prompt=prompt,
        sample_data=sample_data,
        num_samples=args.num_samples,
        max_labels_per_sample=args.max_labels,
        min_labels_per_sample=args.min_labels,
        samples_per_class=args.samples_per_class,
        model=args.model,
        client=client,
        batch_size=args.batch_size
    )
    end_time = time.time()
    
    # Save synthetic data
    print(f"Saving {len(synthetic_samples)} synthetic samples to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(synthetic_samples, f, indent=2)
    
    print("Synthetic data generation complete!")
    print(f"Total time: {end_time - start_time:.1f} seconds")
    
    # Print some statistics
    all_labels = []
    word_counts = []
    for sample in synthetic_samples:
        all_labels.extend(sample['labels'])
        word_counts.append(len(sample['full_text'].split()))
    
    from collections import Counter
    label_counts = Counter(all_labels)
    print(f"\nLabel distribution in generated data:")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count}")
    
    # Print word count statistics
    import statistics
    if word_counts:
        print(f"\nWord count statistics:")
        print(f"  Average: {statistics.mean(word_counts):.1f} words")
        print(f"  Median: {statistics.median(word_counts):.1f} words")
        print(f"  Min: {min(word_counts)} words")
        print(f"  Max: {max(word_counts)} words")
        print(f"  Target (median of validation data): 1178 words")
    
    # Print cost estimate (approximate)
    total_tokens = sum(len(sample['full_text'].split()) for sample in synthetic_samples) * 1.3  # rough estimate
    
    # Cost per 1K tokens (approximate as of 2024)
    cost_per_1k = {
        "gpt-4o-mini": 0.002,
        "gpt-4o": 0.01,
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.002
    }
    
    model_cost = cost_per_1k.get(args.model, 0.002)  # default to gpt-4o-mini pricing
    estimated_cost = (total_tokens / 1000) * model_cost
    print(f"\nEstimated API cost: ${estimated_cost:.2f} (using {args.model} pricing)")

if __name__ == "__main__":
    asyncio.run(main())