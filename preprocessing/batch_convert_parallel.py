"""
Parallel batch converter for PDMX dataset using multiprocessing.
Converts MusicXML to ABC and MIDI to MTF formats using all CPU cores.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import mido


def msg_to_str(msg):
    """Convert MIDI message to string format."""
    str_msg = ""
    for key, value in msg.dict().items():
        str_msg += " " + str(value)
    return str_msg.strip().encode('unicode_escape').decode('utf-8')


def convert_midi_to_mtf_content(midi_path):
    """Convert a single MIDI file to MTF text content."""
    mid = mido.MidiFile(str(midi_path))
    msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]
    
    # Traverse the MIDI file
    for msg in mid.merged_track:
        # Skip metadata for M3 compatibility
        if msg.is_meta:
            if msg.type in ["text", "copyright", "track_name", "instrument_name", 
                           "lyrics", "marker", "cue_marker", "device_name"]:
                continue
        str_msg = msg_to_str(msg)
        msg_list.append(str_msg)
    
    return "\n".join(msg_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel batch convert PDMX dataset")
    parser.add_argument("--skip_abc", action="store_true", help="Skip ABC conversion")
    parser.add_argument("--skip_mtf", action="store_true", help="Skip MTF conversion")
    parser.add_argument("--start_from", type=int, default=0, help="Start from index")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: all cores)")
    return parser.parse_args()


def convert_single_file(entry_data):
    """Process a single entry - convert both MusicXML and MIDI."""
    idx, entry, skip_abc, skip_mtf = entry_data
    
    mxl_path = Path("data") / entry['mxl_path']
    mid_path = Path("data") / entry['mid_path']
    
    # Determine output paths
    abc_output = Path("data/abc") / entry['mxl_path'].replace('.mxl', '.abc').replace('mxl/', '')
    mtf_output = Path("data/mtf") / entry['mid_path'].replace('.mid', '.mtf').replace('mid/', '')
    
    results = {
        'idx': idx,
        'abc_status': 'skipped',
        'mtf_status': 'skipped',
        'abc_error': None,
        'mtf_error': None
    }
    
    # Convert to ABC
    if not skip_abc:
        if abc_output.exists():
            results['abc_status'] = 'exists'
        elif not mxl_path.exists():
            results['abc_status'] = 'failed'
            results['abc_error'] = 'Source file not found'
        else:
            try:
                abc_output.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "preprocessing/abc/utils/xml2abc.py",
                    "-d", "8",
                    "-x", str(mxl_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.stdout:
                    with open(abc_output, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                    results['abc_status'] = 'success'
                else:
                    results['abc_status'] = 'failed'
                    results['abc_error'] = result.stderr[:100] if result.stderr else 'No output'
                    
            except subprocess.TimeoutExpired:
                results['abc_status'] = 'failed'
                results['abc_error'] = 'Timeout (>60s)'
            except Exception as e:
                results['abc_status'] = 'failed'
                results['abc_error'] = str(e)[:100]
    
    # Convert to MTF
    if not skip_mtf:
        if mtf_output.exists():
            results['mtf_status'] = 'exists'
        elif not mid_path.exists():
            results['mtf_status'] = 'failed'
            results['mtf_error'] = 'Source file not found'
        else:
            try:
                mtf_output.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert MIDI to MTF directly
                mtf_content = convert_midi_to_mtf_content(mid_path)
                
                if mtf_content:
                    with open(mtf_output, 'w', encoding='utf-8') as f:
                        f.write(mtf_content)
                    results['mtf_status'] = 'success'
                else:
                    results['mtf_status'] = 'failed'
                    results['mtf_error'] = 'No output generated'
                    
            except Exception as e:
                results['mtf_status'] = 'failed'
                results['mtf_error'] = str(e)[:100]
    
    return results


def main():
    args = parse_args()
    
    # Determine number of workers
    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"Using {num_workers} parallel workers")
    
    # Load entries
    print("Loading metadata...")
    with open('data/processed/abc_entries.json', 'r', encoding='utf-8') as f:
        entries = json.load(f)
    
    # Apply start and limit
    if args.limit:
        entries = entries[args.start_from:args.start_from + args.limit]
    else:
        entries = entries[args.start_from:]
    
    print(f"Processing {len(entries)} entries (starting from index {args.start_from})")
    
    # Create log directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Prepare data for parallel processing
    entry_data = [(args.start_from + i, entry, args.skip_abc, args.skip_mtf) 
                  for i, entry in enumerate(entries)]
    
    # Statistics
    stats = {
        'abc_success': 0,
        'abc_failed': 0,
        'abc_exists': 0,
        'mtf_success': 0,
        'mtf_failed': 0,
        'mtf_exists': 0
    }
    
    # Open error logs
    abc_error_log = open(logs_dir / 'abc_conversion_errors.txt', 'a', encoding='utf-8')
    mtf_error_log = open(logs_dir / 'mtf_conversion_errors.txt', 'a', encoding='utf-8')
    
    # Process in parallel with progress bar
    print("\nStarting parallel conversion...")
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(convert_single_file, entry_data), 
                          total=len(entry_data), 
                          desc="Converting files",
                          unit="files"):
            
            # Update statistics
            if result['abc_status'] == 'success':
                stats['abc_success'] += 1
            elif result['abc_status'] == 'failed':
                stats['abc_failed'] += 1
                if result['abc_error']:
                    abc_error_log.write(f"{result['idx']}\t{result['abc_error']}\n")
            elif result['abc_status'] == 'exists':
                stats['abc_exists'] += 1
            
            if result['mtf_status'] == 'success':
                stats['mtf_success'] += 1
            elif result['mtf_status'] == 'failed':
                stats['mtf_failed'] += 1
                if result['mtf_error']:
                    mtf_error_log.write(f"{result['idx']}\t{result['mtf_error']}\n")
            elif result['mtf_status'] == 'exists':
                stats['mtf_exists'] += 1
            
            # Flush logs periodically
            if (stats['abc_success'] + stats['abc_failed']) % 100 == 0:
                abc_error_log.flush()
                mtf_error_log.flush()
    
    # Close log files
    abc_error_log.close()
    mtf_error_log.close()
    
    # Final statistics
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"ABC Conversion:")
    print(f"  [+] Success:        {stats['abc_success']}")
    print(f"  [-] Failed:         {stats['abc_failed']}")
    print(f"  [=] Already exists: {stats['abc_exists']}")
    print(f"\nMTF Conversion:")
    print(f"  [+] Success:        {stats['mtf_success']}")
    print(f"  [-] Failed:         {stats['mtf_failed']}")
    print(f"  [=] Already exists: {stats['mtf_exists']}")
    print(f"\nError logs: {logs_dir}")
    print(f"Output directories: data/abc, data/mtf")
    
    # Save statistics
    with open(logs_dir / 'conversion_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {logs_dir / 'conversion_stats.json'}")


if __name__ == "__main__":
    main()
