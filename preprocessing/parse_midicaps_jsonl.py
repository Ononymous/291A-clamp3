#!/usr/bin/env python3
"""
Parse MidiCaps train.json and convert to CLaMP3-compatible JSONL format.
Preserves rich semantic information similar to WikiMT structure.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MidiCaps train.json to CLaMP3 JSONL format")
    parser.add_argument(
        "--input_json",
        type=str,
        default="data/test/midicaps_full/train.json",
        help="Path to MidiCaps train.json file"
    )
    parser.add_argument(
        "--mtf_dir",
        type=str,
        default="data/test/midicaps_mtf",
        help="Directory containing converted MTF files"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="data/test/midicaps_clamp3.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--verify_files",
        action="store_true",
        help="Verify that MTF files exist before adding to JSONL"
    )
    return parser.parse_args()


def format_genre_mood(genre_list, genre_prob, mood_list, mood_prob):
    """Format genre and mood information into readable text."""
    parts = []
    
    # Add top genres with probabilities
    if genre_list and len(genre_list) > 0:
        top_genres = [f"{genre_list[i]}" for i in range(min(3, len(genre_list)))]
        parts.append(f"Genre: {', '.join(top_genres)}")
    
    # Add top moods
    if mood_list and len(mood_list) > 0:
        top_moods = [f"{mood_list[i]}" for i in range(min(5, len(mood_list)))]
        parts.append(f"Mood: {', '.join(top_moods)}")
    
    return ". ".join(parts) + "." if parts else ""


def format_musical_details(entry):
    """Format key, time signature, tempo, and duration into readable text."""
    parts = []
    
    if entry.get("key"):
        parts.append(f"Key: {entry['key']}")
    
    if entry.get("time_signature"):
        parts.append(f"Time signature: {entry['time_signature']}")
    
    if entry.get("tempo") and entry.get("tempo_word"):
        parts.append(f"Tempo: {entry['tempo']} BPM ({entry['tempo_word']})")
    
    if entry.get("duration") and entry.get("duration_word"):
        parts.append(f"Duration: {entry['duration']}s ({entry['duration_word']})")
    
    return ". ".join(parts) + "." if parts else ""


def format_chord_progression(entry):
    """Format chord information into readable text."""
    if entry.get("chord_summary") and len(entry["chord_summary"]) > 0:
        chord_list = entry["chord_summary"]
        return f"Main chord progression: {' - '.join(chord_list)}."
    return ""


def format_instruments(entry):
    """Format instrument information into readable text."""
    if entry.get("instrument_summary") and len(entry["instrument_summary"]) > 0:
        instruments = entry["instrument_summary"]
        return f"Instruments: {', '.join(instruments)}."
    return ""


def create_description(entry):
    """
    Create a rich, semantic description similar to WikiMT format.
    Combines caption with additional musical metadata.
    """
    # Start with the main caption
    description_parts = []
    
    if entry.get("caption"):
        description_parts.append(entry["caption"])
    
    # Add genre and mood information
    genre_mood = format_genre_mood(
        entry.get("genre", []),
        entry.get("genre_prob", []),
        entry.get("mood", []),
        entry.get("mood_prob", [])
    )
    if genre_mood:
        description_parts.append(genre_mood)
    
    # Add musical details (key, tempo, time signature)
    musical_details = format_musical_details(entry)
    if musical_details:
        description_parts.append(musical_details)
    
    # Add chord progression
    chord_info = format_chord_progression(entry)
    if chord_info:
        description_parts.append(chord_info)
    
    # Add instruments
    instrument_info = format_instruments(entry)
    if instrument_info:
        description_parts.append(instrument_info)
    
    # Combine all parts
    full_description = " ".join(description_parts)
    
    return full_description.strip()


def convert_midicaps_path_to_mtf(original_path, mtf_dir):
    """
    Convert MidiCaps location path to corresponding MTF file path.
    
    Example:
        Input: lmd_full/1/1a0751ad20e2f82957410a7510a1b13e.mid
        Output: data/test/midicaps_mtf/1/1a0751ad20e2f82957410a7510a1b13e.mtf
    """
    # Extract filename without extension
    path_obj = Path(original_path)
    filename_stem = path_obj.stem
    
    # Get the subfolder (0-9, a-f)
    # MidiCaps uses lmd_full/X/hash.mid format
    parts = path_obj.parts
    if len(parts) >= 2:
        subfolder = parts[1]  # e.g., "1" from lmd_full/1/...
    else:
        # Fallback: use first character of filename
        subfolder = filename_stem[0]
    
    # Construct MTF path
    mtf_path = os.path.join(mtf_dir, subfolder, f"{filename_stem}.mtf")
    
    return mtf_path


def main():
    args = parse_args()
    
    input_json = args.input_json
    mtf_dir = args.mtf_dir
    output_jsonl = args.output_jsonl
    verify_files = args.verify_files
    
    print(f"Reading MidiCaps metadata from: {input_json}")
    
    # Read JSONL file (each line is a JSON object)
    entries = []
    with open(input_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                entries.append(entry)
    
    print(f"Loaded {len(entries)} entries from MidiCaps")
    
    # Convert to CLaMP3 format
    output_entries = []
    skipped_count = 0
    
    for entry in tqdm(entries, desc="Processing entries"):
        # Get MTF file path
        mtf_path = convert_midicaps_path_to_mtf(entry["location"], mtf_dir)
        
        # Verify file exists if requested
        if verify_files and not os.path.exists(mtf_path):
            skipped_count += 1
            continue
        
        # Create rich description
        description = create_description(entry)
        
        # Create CLaMP3-compatible entry
        clamp3_entry = {
            "music": mtf_path,
            "description": description,
            "metadata": {
                "original_location": entry["location"],
                "genre": entry.get("genre", []),
                "mood": entry.get("mood", []),
                "key": entry.get("key", ""),
                "time_signature": entry.get("time_signature", ""),
                "tempo": entry.get("tempo", 0),
                "duration": entry.get("duration", 0),
                "test_set": entry.get("test_set", False)
            }
        }
        
        output_entries.append(clamp3_entry)
    
    # Write output JSONL
    print(f"\nWriting {len(output_entries)} entries to: {output_jsonl}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Successfully created {output_jsonl}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} entries (MTF files not found)")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total entries: {len(output_entries)}")
    test_entries = sum(1 for e in output_entries if e["metadata"].get("test_set", False))
    print(f"  Test set entries: {test_entries}")
    print(f"  Train set entries: {len(output_entries) - test_entries}")


if __name__ == "__main__":
    main()
