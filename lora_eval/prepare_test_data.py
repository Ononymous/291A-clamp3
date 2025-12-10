#!/usr/bin/env python3
"""
Prepare Test Data for CLaMP3 Evaluation

Data Sources:
1. WikiMT-X: JSONL file with ABC leadsheets → Extract ABC directly (standard format works)
2. Lakh MIDI: Folder structure with MIDI files → Convert to MTF

Workflow:
- WikiMT-X: Extract 'leadsheet' field from JSONL → Save as ABC files
- Lakh: Sample MIDI files → Convert to MTF using preprocessing/midi/batch_midi2mtf.py

Note: Interleaving is NOT required - M3Patchilizer handles standard ABC notation directly
"""

import os
import sys
import json
import subprocess
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def prepare_wikimt_abc(num_samples=100):
    """
    Extract ABC leadsheets from WikiMT-X JSONL file.
    
    Args:
        num_samples: Number of samples to extract
    
    Returns:
        (output_dir, count) tuple
    """
    print("\n" + "="*80)
    print("PREPARING WikiMT-X ABC DATASET")
    print("="*80)
    
    # Get project root (go up one level from lora_eval)
    base_dir = Path(__file__).parent.parent
    jsonl_file = base_dir / 'data/test/wikimt-x/wikimt-x-public.jsonl'
    output_abc_dir = base_dir / 'data/test/wikimt_abc'
    
    if not jsonl_file.exists():
        print(f"✗ WikiMT-X JSONL file not found: {jsonl_file}")
        print("  Expected at: data/test/wikimt-x/wikimt-x-public.jsonl")
        return create_minimal_abc_dataset(output_abc_dir)
    
    output_abc_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading WikiMT-X JSONL file: {jsonl_file}")
    
    metadata = {}
    saved_count = 0
    
    # Read JSONL and extract leadsheets
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc="Extracting leadsheets", total=1000)):
            if saved_count >= num_samples:
                break
            
            try:
                data = json.loads(line.strip())
                
                # Extract ABC leadsheet
                leadsheet = data.get('leadsheet', '')
                if not leadsheet:
                    continue
                
                # Save ABC file
                file_id = f"wikimt_{idx:04d}"
                abc_file = output_abc_dir / f"{file_id}.abc"
                
                with open(abc_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(leadsheet)
                
                # Create metadata with annotations
                metadata[file_id] = {
                    'file': f"{file_id}.abc",
                    'title': data.get('title', ''),
                    'artists': data.get('artists', []),
                    'genre': data.get('genre', ''),
                    'description': data.get('description', ''),
                    'background': data.get('background', ''),
                    'analysis': data.get('analysis', ''),
                    'scene': data.get('scene', ''),
                    'format': 'abc',
                    'source': 'wikimt-x'
                }
                
                saved_count += 1
                
            except Exception as e:
                print(f"Warning: Failed to process line {idx}: {e}")
                continue
    
    print(f"✓ Extracted {saved_count} ABC files to: {output_abc_dir}")
    
    # Save metadata
    metadata_file = output_abc_dir / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved metadata with rich annotations")
    print(f"✓ Location: {output_abc_dir}")
    
    return output_abc_dir, saved_count


def create_minimal_abc_dataset(output_dir):
    """Fallback: Create minimal ABC samples."""
    print("\nCreating minimal ABC dataset as fallback...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = {
        'minimal_001': 'X:1\nM:4/4\nL:1/4\nK:C\nCDEF|GABc|',
        'minimal_002': 'X:2\nM:3/4\nL:1/4\nK:G\nGAB|cde|',
    }
    
    metadata = {}
    for file_id, abc_content in samples.items():
        filepath = output_dir / f'{file_id}.abc'
        with open(filepath, 'w') as f:
            f.write(abc_content)
        metadata[file_id] = {
            'file': f'{file_id}.abc',
            'description': 'Minimal sample',
            'format': 'abc',
            'source': 'minimal_sample'
        }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created {len(samples)} minimal ABC samples")
    return output_dir, len(samples)


def prepare_lakh_mtf(midi_base_dir, num_samples=100):
    """
    Convert MIDI files from lmd_matched to MTF format.
    
    Args:
        midi_base_dir: Base directory containing MIDI folder structure (A/B/C/...)
        num_samples: Number of MIDI files to convert
    
    Returns:
        (output_dir, count) tuple
    """
    print("\n" + "="*80)
    print("PREPARING LAKH MTF DATASET")
    print("="*80)
    
    # Get project root (go up one level from lora_eval)
    base_dir = Path(__file__).parent.parent
    midi_dir = Path(midi_base_dir)
    output_dir = base_dir / 'data/test/lakh_mtf'
    temp_midi_dir = base_dir / 'data/test/temp_midi_subset'
    
    if not midi_dir.exists():
        print(f"✗ MIDI directory not found: {midi_dir}")
        return create_minimal_mtf_dataset(output_dir)
    
    # Scan for MIDI files in folder structure
    print(f"Scanning for MIDI files in: {midi_dir}")
    print("  (This may take a moment for large datasets...)")
    
    midi_files = []
    for pattern in ['**/*.mid', '**/*.midi']:
        midi_files.extend(list(midi_dir.glob(pattern)))
    
    if not midi_files:
        print(f"✗ No MIDI files found in: {midi_dir}")
        return create_minimal_mtf_dataset(output_dir)
    
    print(f"✓ Found {len(midi_files)} MIDI files")
    
    # Select random subset
    random.seed(42)  # For reproducibility
    if len(midi_files) > num_samples:
        selected_files = random.sample(midi_files, num_samples)
        print(f"  Selected {num_samples} random files for conversion")
    else:
        selected_files = midi_files
        print(f"  Using all {len(selected_files)} files")
    
    # Create temp directory and copy selected MIDIs
    temp_midi_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCopying {len(selected_files)} MIDI files to temporary directory...")
    file_mapping = {}  # Maps temp filename to original path
    
    for idx, midi_file in enumerate(tqdm(selected_files, desc="Copying MIDI files")):
        dest_name = f"lakh_{idx:04d}.mid"
        dest = temp_midi_dir / dest_name
        shutil.copy(midi_file, dest)
        file_mapping[dest_name] = str(midi_file.relative_to(midi_dir))
    
    # Convert MIDI to MTF using preprocessing script
    print(f"\nConverting MIDI to MTF format...")
    midi2mtf_script = base_dir / 'preprocessing/midi/batch_midi2mtf.py'
    
    if not midi2mtf_script.exists():
        print(f"✗ Conversion script not found: {midi2mtf_script}")
        shutil.rmtree(temp_midi_dir, ignore_errors=True)
        return create_minimal_mtf_dataset(output_dir)
    
    try:
        cmd = [
            sys.executable,
            str(midi2mtf_script),
            str(temp_midi_dir),
            str(output_dir),
            '--m3_compatible',
            '--max_messages', '400'  # Limit to 400 messages to fit in 512 patch limit
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=str(base_dir / 'preprocessing'),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"⚠ Conversion had issues:")
            print(result.stderr[:500])
        
        # Count converted files
        mtf_files = list(output_dir.glob('**/*.mtf'))
        print(f"✓ Converted {len(mtf_files)} MIDI files to MTF format")
        
        # Create metadata (minimal annotations for Lakh dataset)
        metadata = {}
        for mtf_file in mtf_files:
            file_id = mtf_file.stem
            original_path = file_mapping.get(f"{file_id}.mid", "unknown")
            
            metadata[file_id] = {
                'file': mtf_file.name,
                'format': 'mtf',
                'source': 'lakh_midi_dataset',
                'original_midi_path': original_path,
                'description': f'MIDI performance from Lakh dataset: {original_path}'
            }
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Cleanup
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_midi_dir, ignore_errors=True)
        
        print(f"✓ Saved {len(mtf_files)} MTF files with metadata")
        print(f"✓ Location: {output_dir}")
        
        return output_dir, len(mtf_files)
        
    except subprocess.TimeoutExpired:
        print(f"✗ Conversion timed out after 10 minutes")
        shutil.rmtree(temp_midi_dir, ignore_errors=True)
        return create_minimal_mtf_dataset(output_dir)
    except Exception as e:
        print(f"✗ Conversion error: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_midi_dir, ignore_errors=True)
        return create_minimal_mtf_dataset(output_dir)


def create_minimal_mtf_dataset(output_dir):
    """Fallback: Create minimal MTF samples."""
    print("\nCreating minimal MTF dataset as fallback...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = {
        'minimal_001': 'ticks_per_beat 480\n0 note_on channel=0 pitch=60 velocity=100\n480 note_off channel=0 pitch=60 velocity=0\n',
        'minimal_002': 'ticks_per_beat 480\n0 note_on channel=0 pitch=64 velocity=100\n480 note_off channel=0 pitch=64 velocity=0\n',
    }
    
    metadata = {}
    for file_id, mtf_content in samples.items():
        filepath = output_dir / f'{file_id}.mtf'
        with open(filepath, 'w') as f:
            f.write(mtf_content)
        metadata[file_id] = {
            'file': f'{file_id}.mtf',
            'description': 'Minimal MTF sample',
            'format': 'mtf',
            'source': 'minimal_sample'
        }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created {len(samples)} minimal MTF samples")
    return output_dir, len(samples)


def main():
    """Prepare evaluation datasets."""
    
    parser = argparse.ArgumentParser(description='Prepare CLaMP3 evaluation datasets')
    parser.add_argument('--num_abc', type=int, default=100,
                       help='Number of ABC samples from WikiMT-X (default: 100)')
    parser.add_argument('--num_mtf', type=int, default=100,
                       help='Number of MTF samples from Lakh MIDI (default: 100)')
    parser.add_argument('--midi_dir', type=str,
                       default='/Users/kaitlynt/291A-clamp3/data/test/lmd_matched',
                       help='Directory containing MIDI files with folder structure')
    parser.add_argument('--skip_abc', action='store_true',
                       help='Skip ABC dataset preparation')
    parser.add_argument('--skip_mtf', action='store_true',
                       help='Skip MTF dataset preparation')
    args = parser.parse_args()
    
    print("\n" + "#"*80)
    print("# CLaMP3 EVALUATION DATA PREPARATION")
    print("# WikiMT-X (ABC Leadsheets) + Lakh MIDI Dataset (MTF)")
    print("#"*80)
    
    abc_count = 0
    mtf_count = 0
    abc_dir = None
    mtf_dir = None
    
    # Prepare ABC dataset from WikiMT-X JSONL
    if not args.skip_abc:
        try:
            abc_dir, abc_count = prepare_wikimt_abc(num_samples=args.num_abc)
        except Exception as e:
            print(f"\n✗ ABC preparation failed: {e}")
            import traceback
            traceback.print_exc()
            base_dir = Path(__file__).parent.parent
            abc_dir = base_dir / 'data/test/wikimt_abc'
            abc_dir, abc_count = create_minimal_abc_dataset(abc_dir)
    else:
        print("\n⊗ Skipping ABC dataset preparation")
        base_dir = Path(__file__).parent.parent
        abc_dir = base_dir / 'data/test/wikimt_abc'
    
    # Prepare MTF dataset from Lakh MIDI
    if not args.skip_mtf:
        try:
            mtf_dir, mtf_count = prepare_lakh_mtf(
                midi_base_dir=args.midi_dir,
                num_samples=args.num_mtf
            )
        except Exception as e:
            print(f"\n✗ MTF preparation failed: {e}")
            import traceback
            traceback.print_exc()
            base_dir = Path(__file__).parent.parent
            mtf_dir = base_dir / 'data/test/lakh_mtf'
            mtf_dir, mtf_count = create_minimal_mtf_dataset(mtf_dir)
    else:
        print("\n⊗ Skipping MTF dataset preparation")
        base_dir = Path(__file__).parent.parent
        mtf_dir = base_dir / 'data/test/lakh_mtf'
    
    # Summary
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE")
    print("="*80)
    
    print(f"\n📊 ABC Dataset (WikiMT-X):")
    print(f"  Location: {abc_dir}")
    print(f"  Files: {abc_count} ABC files")
    print(f"  Format: Standard ABC notation")
    print(f"  Annotations: Rich (title, description, analysis, etc.)")
    print(f"  Source: data/test/wikimt-x/wikimt-x-public.jsonl")
    
    print(f"\n🎹 MTF Dataset (Lakh MIDI):")
    print(f"  Location: {mtf_dir}")
    print(f"  Files: {mtf_count} MTF files")
    print(f"  Format: MTF (M3-compatible MIDI text format)")
    print(f"  Annotations: Minimal (file path descriptions)")
    print(f"  Source: {args.midi_dir}")
    
    print(f"\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Test pipeline integrity:")
    print("   python test_pipeline.py")
    print("\n2. Evaluate baseline model:")
    print("   python evaluate_baseline.py")
    print("\n3. Evaluate LoRA adapters:")
    print("   python evaluate_lora.py")
    print("\n4. Compare results:")
    print("   python compare_results.py")
    print("\nOr run complete evaluation:")
    print("   bash run_evaluation.sh")
    print()


if __name__ == '__main__':
    main()
