"""
Split MidiCaps dataset into train/validation/test sets.

This script combines the existing training and test MidiCaps files
and creates new splits: 80% train, 10% validation, 10% test.
"""

import json
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_JSONL = DATA_DIR / "training" / "clamp3_train_mtf.jsonl"
TEST_JSONL = DATA_DIR / "test" / "midicaps_clamp3.jsonl"

OUTPUT_DIR = DATA_DIR / "midicaps_splits"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_OUT = OUTPUT_DIR / "midicaps_train.jsonl"
VAL_OUT = OUTPUT_DIR / "midicaps_val.jsonl"
TEST_OUT = OUTPUT_DIR / "midicaps_test.jsonl"

# Split sizes
VAL_SIZE = 1000   # Fixed validation size
TEST_SIZE = 1000  # Fixed test size
# Remaining samples go to training

def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    """Save data to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    print("Loading MidiCaps data...")
    train_data = load_jsonl(TRAIN_JSONL)
    test_data = load_jsonl(TEST_JSONL)
    
    print(f"Loaded {len(train_data):,} training samples")
    print(f"Loaded {len(test_data):,} test samples")
    
    # Combine all data
    all_data = train_data + test_data
    total = len(all_data)
    print(f"\nTotal samples: {total:,}")
    
    # Shuffle
    random.shuffle(all_data)
    
    # Split: 1000 val, 1000 test, rest for training
    val_split = all_data[:VAL_SIZE]
    test_split = all_data[VAL_SIZE:VAL_SIZE + TEST_SIZE]
    train_split = all_data[VAL_SIZE + TEST_SIZE:]
    
    print(f"\nSplit sizes:")
    print(f"  Train:      {len(train_split):,} ({len(train_split)/total*100:.1f}%)")
    print(f"  Validation: {len(val_split):,} ({len(val_split)/total*100:.1f}%)")
    print(f"  Test:       {len(test_split):,} ({len(test_split)/total*100:.1f}%)")
    
    # Save splits
    print(f"\nSaving splits to {OUTPUT_DIR}...")
    save_jsonl(train_split, TRAIN_OUT)
    save_jsonl(val_split, VAL_OUT)
    save_jsonl(test_split, TEST_OUT)
    
    print("\n✓ Done!")
    print(f"  Train: {TRAIN_OUT}")
    print(f"  Val:   {VAL_OUT}")
    print(f"  Test:  {TEST_OUT}")

if __name__ == "__main__":
    main()
