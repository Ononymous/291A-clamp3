"""
Generate CLaMP3 training JSONL files from parsed PDMX metadata.
Creates separate JSONL files for ABC and MTF training.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CLaMP3 training JSONL files from PDMX metadata"
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="data/processed",
        help="Directory containing parsed metadata JSON files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory containing abc/ and mtf/ folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/training",
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--verify_files",
        action="store_true",
        help="Verify that referenced files actually exist (slower but safer)"
    )
    return parser.parse_args()


def verify_file_exists(base_dir, relative_path):
    """Check if a file exists given base directory and relative path."""
    if not relative_path:
        return False
    full_path = os.path.join(base_dir, relative_path)
    return os.path.exists(full_path)


def convert_metadata_to_jsonl_entry(metadata, file_format, data_dir, verify_files=False):
    """
    Convert parsed metadata to CLaMP3 JSONL format.
    
    Args:
        metadata: Dictionary containing parsed metadata
        file_format: 'abc' or 'mtf'
        data_dir: Base data directory
        verify_files: Whether to verify file existence
    
    Returns:
        Dictionary in CLaMP3 JSONL format, or None if file doesn't exist
    """
    
    # Determine file path based on format
    if file_format == 'abc':
        # For ABC: convert mxl path to abc path
        if not metadata.get('mxl_path'):
            return None
        
        # Replace 'mxl/' with 'abc/' and change extension
        abc_path = metadata['mxl_path'].replace('mxl/', 'abc/').replace('.mxl', '.abc')
        
        # Verify file exists if requested
        if verify_files and not verify_file_exists(data_dir, abc_path):
            return None
        
        filepath = abc_path
        
    elif file_format == 'mtf':
        # For MTF: convert mid path to mtf path
        if not metadata.get('mid_path'):
            return None
        
        # Replace 'mid/' with 'mtf/' and change extension
        mtf_path = metadata['mid_path'].replace('mid/', 'mtf/')
        # Handle both .mid and .midi extensions
        if mtf_path.endswith('.mid'):
            mtf_path = mtf_path[:-4] + '.mtf'
        elif mtf_path.endswith('.midi'):
            mtf_path = mtf_path[:-5] + '.mtf'
        
        # Verify file exists if requested
        if verify_files and not verify_file_exists(data_dir, mtf_path):
            return None
        
        filepath = mtf_path
    else:
        return None
    
    # Create JSONL entry following CLaMP3 format
    entry = {
        "id": metadata.get('score_id', ''),
        "filepaths": [filepath],
        "analysis": metadata.get('text_description', ''),
        "language": "en",  # Default to English
        "translations": {}  # No translations for now
    }
    
    return entry


def generate_jsonl_files(metadata_dir, data_dir, output_dir, verify_files=False):
    """Generate JSONL training files for ABC and MTF formats."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parsed metadata
    print("Loading parsed metadata...")
    
    abc_metadata_file = os.path.join(metadata_dir, 'abc_entries.json')
    mtf_metadata_file = os.path.join(metadata_dir, 'mtf_entries.json')
    
    if not os.path.exists(abc_metadata_file):
        print(f"Error: ABC metadata not found: {abc_metadata_file}")
        print("Please run parse_pdmx_csv.py first!")
        return False
    
    if not os.path.exists(mtf_metadata_file):
        print(f"Error: MTF metadata not found: {mtf_metadata_file}")
        print("Please run parse_pdmx_csv.py first!")
        return False
    
    with open(abc_metadata_file, 'r', encoding='utf-8') as f:
        abc_metadata_list = json.load(f)
    
    with open(mtf_metadata_file, 'r', encoding='utf-8') as f:
        mtf_metadata_list = json.load(f)
    
    print(f"Loaded {len(abc_metadata_list)} ABC entries")
    print(f"Loaded {len(mtf_metadata_list)} MTF entries")
    
    # Generate ABC JSONL
    print("\nGenerating ABC training JSONL...")
    abc_jsonl_path = os.path.join(output_dir, 'clamp3_train_abc.jsonl')
    abc_count = 0
    abc_skipped = 0
    
    with open(abc_jsonl_path, 'w', encoding='utf-8') as f:
        for metadata in tqdm(abc_metadata_list, desc="Processing ABC entries"):
            entry = convert_metadata_to_jsonl_entry(
                metadata, 'abc', data_dir, verify_files
            )
            if entry:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                abc_count += 1
            else:
                abc_skipped += 1
    
    print(f"  ✓ Written {abc_count} entries to {abc_jsonl_path}")
    if abc_skipped > 0:
        print(f"  ⚠ Skipped {abc_skipped} entries (missing files or metadata)")
    
    # Generate MTF JSONL
    print("\nGenerating MTF training JSONL...")
    mtf_jsonl_path = os.path.join(output_dir, 'clamp3_train_mtf.jsonl')
    mtf_count = 0
    mtf_skipped = 0
    
    with open(mtf_jsonl_path, 'w', encoding='utf-8') as f:
        for metadata in tqdm(mtf_metadata_list, desc="Processing MTF entries"):
            entry = convert_metadata_to_jsonl_entry(
                metadata, 'mtf', data_dir, verify_files
            )
            if entry:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                mtf_count += 1
            else:
                mtf_skipped += 1
    
    print(f"  ✓ Written {mtf_count} entries to {mtf_jsonl_path}")
    if mtf_skipped > 0:
        print(f"  ⚠ Skipped {mtf_skipped} entries (missing files or metadata)")
    
    # Generate summary statistics
    stats = {
        'abc': {
            'total_entries': len(abc_metadata_list),
            'written': abc_count,
            'skipped': abc_skipped,
            'jsonl_path': abc_jsonl_path
        },
        'mtf': {
            'total_entries': len(mtf_metadata_list),
            'written': mtf_count,
            'skipped': mtf_skipped,
            'jsonl_path': mtf_jsonl_path
        }
    }
    
    stats_path = os.path.join(output_dir, 'jsonl_generation_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("JSONL Generation Summary")
    print("="*60)
    print(f"ABC Training JSONL: {abc_count} entries")
    print(f"MTF Training JSONL: {mtf_count} entries")
    print(f"\nFiles saved to: {output_dir}")
    print("="*60)
    
    return True


if __name__ == '__main__':
    args = parse_args()
    
    metadata_dir = os.path.abspath(args.metadata_dir)
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print("CLaMP3 JSONL Generator")
    print("="*60)
    print(f"Metadata directory: {metadata_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Verify files: {args.verify_files}")
    print("="*60)
    
    success = generate_jsonl_files(
        metadata_dir, 
        data_dir, 
        output_dir,
        args.verify_files
    )
    
    if success:
        print("\n✓ JSONL generation completed successfully!")
    else:
        print("\n✗ JSONL generation failed!")
