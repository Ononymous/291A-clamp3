"""
Parse PDMX.csv file to extract metadata for CLaMP3 training.
This script reads the PDMX dataset CSV and creates structured data
for generating training JSONL files.
"""

import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse PDMX.csv to extract metadata for CLaMP3 training"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/PDMX.csv",
        help="Path to PDMX.csv file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save parsed metadata"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data",
        help="Base directory containing the data files"
    )
    return parser.parse_args()


def clean_path(path_str):
    """Clean and normalize file paths from CSV."""
    if not path_str or path_str == "NA":
        return None
    # Remove leading './' if present
    path_str = path_str.strip()
    if path_str.startswith('./'):
        path_str = path_str[2:]
    return path_str


def extract_metadata_fields(row):
    """Extract relevant metadata fields from a CSV row."""
    metadata = {}
    
    # Basic identification
    metadata['score_id'] = row.get('path', '').split('/')[-1].replace('.json', '')
    
    # Title and naming
    metadata['title'] = row.get('title', '') or row.get('song_name', '') or row.get('file_score_title', '')
    metadata['subtitle'] = row.get('subtitle', '')
    metadata['song_name'] = row.get('song_name', '')
    
    # Creator information
    metadata['composer'] = row.get('composer_name', '')
    metadata['artist'] = row.get('artist_name', '')
    metadata['publisher'] = row.get('publisher', '')
    
    # Classification
    metadata['genres'] = row.get('genres', '')
    metadata['tags'] = row.get('tags', '')
    metadata['groups'] = row.get('groups', '')
    
    # Musical characteristics
    metadata['complexity'] = row.get('complexity', '')
    metadata['n_tracks'] = row.get('n_tracks', '')
    metadata['instruments'] = row.get('tracks', '')
    
    # Quality indicators
    metadata['rating'] = row.get('rating', '')
    metadata['n_ratings'] = row.get('n_ratings', '')
    metadata['is_rated'] = row.get('is_rated', '')
    
    # License information
    metadata['license'] = row.get('license', '')
    metadata['is_public_domain'] = row.get('is_public_domain', '')
    metadata['is_original'] = row.get('is_original', '')
    
    # File paths
    metadata['path'] = clean_path(row.get('path', ''))
    metadata['metadata_path'] = clean_path(row.get('metadata', ''))
    metadata['mxl_path'] = clean_path(row.get('mxl', ''))
    metadata['mid_path'] = clean_path(row.get('mid', ''))
    
    # Subset membership (for filtering if needed)
    metadata['subset_all'] = row.get('subset:all', '')
    metadata['subset_rated'] = row.get('subset:rated', '')
    metadata['subset_deduplicated'] = row.get('subset:deduplicated', '')
    metadata['subset_rated_deduplicated'] = row.get('subset:rated_deduplicated', '')
    metadata['subset_no_license_conflict'] = row.get('subset:no_license_conflict', '')
    metadata['subset_all_valid'] = row.get('subset:all_valid', '')
    
    return metadata


def create_text_description(metadata):
    """Create a text description from metadata for contrastive learning."""
    parts = []
    
    # Title
    if metadata['title']:
        parts.append(f"Title: {metadata['title']}")
    
    # Composer/Artist
    if metadata['composer']:
        parts.append(f"Composer: {metadata['composer']}")
    elif metadata['artist']:
        parts.append(f"Artist: {metadata['artist']}")
    
    # Genre
    if metadata['genres'] and metadata['genres'] != 'NA':
        parts.append(f"Genre: {metadata['genres']}")
    
    # Instruments/Tracks
    if metadata['instruments'] and metadata['instruments'] != 'NA':
        parts.append(f"Instruments: {metadata['instruments']}")
    
    # Tags
    if metadata['tags'] and metadata['tags'] != 'NA':
        parts.append(f"Tags: {metadata['tags']}")
    
    # Complexity
    if metadata['complexity']:
        try:
            complexity_level = int(float(metadata['complexity']))
            if complexity_level == 0:
                parts.append("Difficulty: Simple")
            elif complexity_level == 1:
                parts.append("Difficulty: Intermediate")
            elif complexity_level >= 2:
                parts.append("Difficulty: Advanced")
        except (ValueError, TypeError):
            pass
    
    return ". ".join(parts) if parts else metadata['title'] or "Untitled"


def parse_pdmx_csv(csv_path, base_dir, output_dir):
    """Parse PDMX.csv and extract metadata."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for parsed data
    all_entries = []
    abc_entries = []
    mtf_entries = []
    
    # Statistics
    stats = {
        'total_rows': 0,
        'with_mxl': 0,
        'with_mid': 0,
        'with_both': 0,
        'with_metadata': 0,
        'errors': 0
    }
    
    print(f"Reading CSV file: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Use csv.DictReader for easier column access
            reader = csv.DictReader(f)
            
            for row in reader:
                stats['total_rows'] += 1
                
                try:
                    metadata = extract_metadata_fields(row)
                    
                    # Check file existence
                    has_mxl = metadata['mxl_path'] is not None
                    has_mid = metadata['mid_path'] is not None
                    has_metadata_file = metadata['metadata_path'] is not None
                    
                    if has_mxl:
                        stats['with_mxl'] += 1
                    if has_mid:
                        stats['with_mid'] += 1
                    if has_mxl and has_mid:
                        stats['with_both'] += 1
                    if has_metadata_file:
                        stats['with_metadata'] += 1
                    
                    # Create text description
                    text_description = create_text_description(metadata)
                    metadata['text_description'] = text_description
                    
                    all_entries.append(metadata)
                    
                    # Add to format-specific lists if files exist
                    if has_mxl:
                        abc_entries.append(metadata)
                    if has_mid:
                        mtf_entries.append(metadata)
                    
                except Exception as e:
                    stats['errors'] += 1
                    print(f"Error processing row {stats['total_rows']}: {e}")
                    continue
                
                # Progress update every 10000 rows
                if stats['total_rows'] % 10000 == 0:
                    print(f"Processed {stats['total_rows']} rows...")
    
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Save parsed data
    print(f"\nSaving parsed metadata...")
    
    with open(os.path.join(output_dir, 'all_entries.json'), 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, 'abc_entries.json'), 'w', encoding='utf-8') as f:
        json.dump(abc_entries, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, 'mtf_entries.json'), 'w', encoding='utf-8') as f:
        json.dump(mtf_entries, f, indent=2, ensure_ascii=False)
    
    # Save statistics
    with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PDMX CSV Parsing Summary:")
    print("="*60)
    print(f"Total rows processed: {stats['total_rows']}")
    print(f"Entries with MusicXML: {stats['with_mxl']}")
    print(f"Entries with MIDI: {stats['with_mid']}")
    print(f"Entries with both: {stats['with_both']}")
    print(f"Entries with metadata JSON: {stats['with_metadata']}")
    print(f"Errors encountered: {stats['errors']}")
    print("="*60)
    print(f"\nParsed data saved to: {output_dir}")
    print(f"  - all_entries.json: {len(all_entries)} entries")
    print(f"  - abc_entries.json: {len(abc_entries)} entries")
    print(f"  - mtf_entries.json: {len(mtf_entries)} entries")
    
    return {
        'all_entries': all_entries,
        'abc_entries': abc_entries,
        'mtf_entries': mtf_entries,
        'stats': stats
    }


if __name__ == '__main__':
    args = parse_args()
    
    csv_path = os.path.abspath(args.csv_path)
    output_dir = os.path.abspath(args.output_dir)
    base_dir = os.path.abspath(args.base_dir)
    
    print("PDMX CSV Parser")
    print("="*60)
    print(f"CSV file: {csv_path}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    result = parse_pdmx_csv(csv_path, base_dir, output_dir)
    
    if result:
        print("\n[SUCCESS] Parsing completed successfully!")
    else:
        print("\n[ERROR] Parsing failed!")
