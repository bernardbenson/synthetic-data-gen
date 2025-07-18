#!/usr/bin/env python3
"""
Label Distribution Analyzer

Analyzes label distributions from synthetic training data JSON files.
"""

import json
import argparse
from collections import Counter
from pathlib import Path


def analyze_label_distributions(json_file_path):
    """
    Analyze label distributions from a JSON file containing training data.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        dict: Label distribution statistics
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect all labels
    all_labels = []
    for item in data:
        if 'labels' in item:
            all_labels.extend(item['labels'])
    
    # Count label frequencies
    label_counts = Counter(all_labels)
    total_labels = len(all_labels)
    total_entries = len(data)
    
    # Calculate statistics
    stats = {
        'total_entries': total_entries,
        'total_labels': total_labels,
        'unique_labels': len(label_counts),
        'avg_labels_per_entry': total_labels / total_entries if total_entries > 0 else 0,
        'label_distribution': dict(label_counts),
        'label_percentages': {label: (count / total_labels * 100) 
                            for label, count in label_counts.items()},
        'most_common_labels': label_counts.most_common(),
    }
    
    return stats


def print_analysis(stats):
    """Print formatted analysis results."""
    print("=" * 60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Total entries: {stats['total_entries']:,}")
    print(f"Total labels: {stats['total_labels']:,}")
    print(f"Unique labels: {stats['unique_labels']}")
    print(f"Average labels per entry: {stats['avg_labels_per_entry']:.2f}")
    print()
    
    print("LABEL FREQUENCIES:")
    print("-" * 40)
    for label, count in stats['most_common_labels']:
        percentage = stats['label_percentages'][label]
        print(f"{label:<25} {count:>5} ({percentage:>5.1f}%)")
    print()
    
    print("LABEL DISTRIBUTION SUMMARY:")
    print("-" * 40)
    sorted_labels = sorted(stats['label_distribution'].items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_labels:
        print(f"{label}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Analyze label distributions from JSON training data')
    parser.add_argument('json_file', help='Path to the JSON file to analyze')
    parser.add_argument('--output', '-o', help='Output file to save analysis results (optional)')
    
    args = parser.parse_args()
    
    # Validate input file
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File '{args.json_file}' not found.")
        return 1
    
    try:
        # Analyze the data
        stats = analyze_label_distributions(args.json_file)
        
        # Print results
        print_analysis(stats)
        
        # Save to output file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed analysis saved to: {args.output}")
    
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())