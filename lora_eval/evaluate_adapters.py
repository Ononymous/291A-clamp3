#!/usr/bin/env python3
"""
Combined evaluation script for LoRA adapters.
Evaluates on MidiCaps (MIDI/MTF) and WikiMT (ABC) test sets.
"""

import os
import sys
import torch
import numpy as np
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from config import *
from utils import *
from transformers import BertConfig, AutoTokenizer
from peft import PeftModel

# Set random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def load_test_data(jsonl_path, data_format, max_samples=None):
    """Load test data from JSONL file."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                entry = json.loads(line)
                entry['index'] = i
                entry['format'] = data_format
                data.append(entry)
    return data


def extract_features(model, tokenizer, patchilizer, entries, device, data_format, text_field='analysis'):
    """Extract text and music features from entries.
    
    Args:
        text_field: For wikimt data_format, specify which field to use: 'background', 'analysis', 'description', or 'scene'
    """
    text_features = []
    music_features = []
    valid_indices = []
    
    print(f"Extracting features for {data_format} data (text field: {text_field})...")
    for entry in tqdm(entries, desc="Processing"):
        # Extract text features
        if data_format == 'midicaps':
            description = entry.get('description', '')
        elif data_format == 'wikimt':
            description = entry.get(text_field, '')
        else:
            continue
            
        if not description:
            continue
            
        inputs = tokenizer(description, return_tensors="pt", max_length=MAX_TEXT_LENGTH, 
                          truncation=True, padding='max_length')
        text_inputs = inputs['input_ids'].to(device)
        text_masks = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            text_feat = model.get_text_features(text_inputs, text_masks, get_global=True)
        
        # Extract music features
        if data_format == 'midicaps':
            # MidiCaps: uses 'music' field pointing to MTF files
            music_path = entry.get('music', '')
            if not music_path:
                continue
            
            # Construct full path
            repo_root = os.path.dirname(os.path.dirname(__file__))
            full_path = os.path.join(repo_root, music_path)
            
            if not os.path.exists(full_path):
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    music_content = f.read()
            except:
                continue
                
        elif data_format == 'wikimt':
            # Both WikiMT and WikiMT-X use 'leadsheet' field with ABC notation directly embedded
            music_content = entry.get('leadsheet', '')
            if not music_content:
                continue
        else:
            continue
        
        # Encode music
        try:
            encoded = patchilizer.encode(music_content, add_special_patches=True, truncate=True)
            encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
            masks = torch.ones(encoded_tensor.size(1)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                music_feat = model.get_symbolic_features(symbolic_inputs=encoded_tensor,
                                                        symbolic_masks=masks,
                                                        get_global=True)
        except:
            continue
        
        text_features.append(text_feat.cpu().numpy().flatten())
        music_features.append(music_feat.cpu().numpy().flatten())
        valid_indices.append(entry['index'])
    
    # Clear CUDA cache after all features extracted
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return np.array(text_features), np.array(music_features), valid_indices


def compute_retrieval_metrics(query_features, reference_features, k_values=[1, 5, 10]):
    """Compute retrieval metrics (MRR, Hit@K)."""
    # Normalize
    query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
    ref_norm = reference_features / (np.linalg.norm(reference_features, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    similarities = np.dot(query_norm, ref_norm.T)
    
    mrr = 0
    hits = {k: 0 for k in k_values}
    n_queries = len(query_features)
    
    for i in range(n_queries):
        sim_scores = similarities[i]
        ranked_indices = np.argsort(-sim_scores)
        rank_position = np.where(ranked_indices == i)[0][0] + 1
        
        mrr += 1.0 / rank_position
        
        for k in k_values:
            if i in ranked_indices[:k]:
                hits[k] += 1
    
    mrr /= n_queries
    hits = {k: (v / n_queries) * 100 for k, v in hits.items()}
    
    return mrr, hits


def evaluate_model(model, tokenizer, patchilizer, test_data, device, model_name, data_format, text_field='analysis'):
    """Evaluate a model on test data."""
    field_str = f" ({text_field})" if data_format == 'wikimt' else ""
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} on {data_format.upper()}{field_str}")
    print(f"{'='*80}")
    
    text_features, music_features, valid_indices = extract_features(
        model, tokenizer, patchilizer, test_data, device, data_format, text_field
    )
    
    print(f"Valid samples: {len(valid_indices)}")
    
    # Text-to-Music
    print("\nComputing Text-to-Music metrics...")
    t2m_mrr, t2m_hits = compute_retrieval_metrics(text_features, music_features)
    
    # Music-to-Text
    print("Computing Music-to-Text metrics...")
    m2t_mrr, m2t_hits = compute_retrieval_metrics(music_features, text_features)
    
    results = {
        'model': model_name,
        'dataset': data_format,
        'num_samples': len(valid_indices),
        'text_to_music': {
            'MRR': float(t2m_mrr),
            'Hit@1': float(t2m_hits[1]),
            'Hit@5': float(t2m_hits[5]),
            'Hit@10': float(t2m_hits[10])
        },
        'music_to_text': {
            'MRR': float(m2t_mrr),
            'Hit@1': float(m2t_hits[1]),
            'Hit@5': float(m2t_hits[5]),
            'Hit@10': float(m2t_hits[10])
        }
    }
    
    return results


def print_comparison(dataset_name, baseline_results, lora_results):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"\n{dataset_name.upper()}")
    print("-" * 80)
    
    # Text-to-Music table
    print("\nText-to-Music Retrieval:")
    print(f"{'Metric':<10} {'Baseline':<15} {'LoRA':<15} {'Change':<15} {'%Change':<10}")
    for metric in ['MRR', 'Hit@1', 'Hit@5', 'Hit@10']:
        baseline_val = baseline_results['text_to_music'][metric]
        lora_val = lora_results['text_to_music'][metric]
        diff = lora_val - baseline_val
        pct = (diff / baseline_val * 100) if baseline_val > 0 else 0
        print(f"{metric:<10} {baseline_val:<15.4f} {lora_val:<15.4f} {diff:<+15.4f} {pct:>8.2f}%")
    
    # Music-to-Text table
    print("\nMusic-to-Text Retrieval:")
    print(f"{'Metric':<10} {'Baseline':<15} {'LoRA':<15} {'Change':<15} {'%Change':<10}")
    for metric in ['MRR', 'Hit@1', 'Hit@5', 'Hit@10']:
        baseline_val = baseline_results['music_to_text'][metric]
        lora_val = lora_results['music_to_text'][metric]
        diff = lora_val - baseline_val
        pct = (diff / baseline_val * 100) if baseline_val > 0 else 0
        print(f"{metric:<10} {baseline_val:<15.4f} {lora_val:<15.4f} {diff:<+15.4f} {pct:>8.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA adapters on test datasets')
    parser.add_argument('--eval_midicaps', action='store_true', help='Evaluate on MidiCaps')
    parser.add_argument('--eval_wikimt', action='store_true', help='Evaluate on WikiMT-X')
    parser.add_argument('--wikimt_text_field', type=str, default='analysis', 
                       choices=['background', 'analysis', 'description', 'scene', 'all'],
                       help='WikiMT-X text field to use (or "all" to evaluate each separately)')
    parser.add_argument('--midicaps_adapter', type=str, default=None, help='Path to MidiCaps adapter (default: LORA_MTF_ADAPTER_PATH)')
    parser.add_argument('--wikimt_adapter', type=str, default=None, help='Path to WikiMT adapter (default: LORA_ABC_ADAPTER_PATH)')
    args = parser.parse_args()
    
    # If no flags specified, evaluate both
    if not args.eval_midicaps and not args.eval_wikimt:
        args.eval_midicaps = True
        args.eval_wikimt = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Setup configs
    audio_config = BertConfig(
        vocab_size=1, hidden_size=AUDIO_HIDDEN_SIZE,
        num_hidden_layers=AUDIO_NUM_LAYERS,
        num_attention_heads=AUDIO_HIDDEN_SIZE//64,
        intermediate_size=AUDIO_HIDDEN_SIZE*4,
        max_position_embeddings=MAX_AUDIO_LENGTH
    )
    symbolic_config = BertConfig(
        vocab_size=1, hidden_size=M3_HIDDEN_SIZE,
        num_hidden_layers=PATCH_NUM_LAYERS,
        num_attention_heads=M3_HIDDEN_SIZE//64,
        intermediate_size=M3_HIDDEN_SIZE*4,
        max_position_embeddings=PATCH_LENGTH
    )
    
    # Load tokenizer and patchilizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    patchilizer = M3Patchilizer()
    
    all_results = {}
    
    # ========== EVALUATE MIDICAPS ==========
    if args.eval_midicaps:
        print("\n" + "="*80)
        print("EVALUATING ON MIDICAPS")
        print("="*80)
        
        # Load test data
        midicaps_test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'midicaps_splits', 'midicaps_test.jsonl')
        print(f"\nLoading MidiCaps test data from {midicaps_test_path}")
        midicaps_data = load_test_data(midicaps_test_path, 'midicaps')
        print(f"Loaded {len(midicaps_data)} test samples")
        
        # Load base model
        print("\nLoading base CLaMP3 model...")
        base_model = CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=TEXT_MODEL_NAME,
            hidden_size=CLAMP3_HIDDEN_SIZE,
            load_m3=CLAMP3_LOAD_M3
        )
        
        checkpoint = torch.load(CLAMP3_WEIGHTS_PATH, map_location='cpu', weights_only=True)
        base_model.load_state_dict(checkpoint['model'])
        base_model = base_model.to(device)
        base_model.eval()
        
        # Evaluate baseline
        baseline_midicaps = evaluate_model(base_model, tokenizer, patchilizer, midicaps_data, device, "Baseline", "midicaps")
        
        # Load LoRA adapter
        adapter_path = args.midicaps_adapter if args.midicaps_adapter else LORA_MTF_ADAPTER_PATH
        print(f"\nLoading MTF LoRA adapter from {adapter_path}")
        base_model.symbolic_model.base = PeftModel.from_pretrained(
            base_model.symbolic_model.base,
            adapter_path
        )
        base_model.symbolic_model.base = base_model.symbolic_model.base.to(device)
        base_model.eval()
        
        # Evaluate LoRA
        lora_midicaps = evaluate_model(base_model, tokenizer, patchilizer, midicaps_data, device, "LoRA", "midicaps")
        
        # Print comparison
        print_comparison("MidiCaps", baseline_midicaps, lora_midicaps)
        
        # Save results
        all_results['midicaps'] = {
            'baseline': baseline_midicaps,
            'lora': lora_midicaps
        }
        
        # Clean up for next dataset
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========== EVALUATE WIKIMT ==========
    if args.eval_wikimt:
        print("\n" + "="*80)
        print(f"EVALUATING ON WIKIMT-X")
        print("="*80)
        
        # Load WikiMT-X test data
        wikimt_test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test', 'wikimt-x-public.jsonl')
            
        if not os.path.exists(wikimt_test_path):
            print(f"\n⚠ WikiMT-X test file not found: {wikimt_test_path}")
            print(f"Skipping WikiMT-X evaluation. Please ensure wikimt-x-public.jsonl is available.")
            args.eval_wikimt = False
        else:
            print(f"\nLoading WikiMT-X test data from {wikimt_test_path}")
            wikimt_data = load_test_data(wikimt_test_path, 'wikimt')
            print(f"Loaded {len(wikimt_data)} test samples")
    
    if args.eval_wikimt:
        # Determine which text fields to evaluate
        text_fields = ['background', 'analysis', 'description', 'scene'] if args.wikimt_text_field == 'all' else [args.wikimt_text_field]
        
        # Evaluate each text field
        for field_idx, text_field in enumerate(text_fields):
            if len(text_fields) > 1:
                print(f"\n{'='*80}")
                print(f"FIELD {field_idx + 1}/{len(text_fields)}: {text_field.upper()}")
                print(f"{'='*80}")
            
            # Load base model (fresh instance for each field)
            print("\nLoading base CLaMP3 model...")
            base_model = CLaMP3Model(
                audio_config=audio_config,
                symbolic_config=symbolic_config,
                text_model_name=TEXT_MODEL_NAME,
                hidden_size=CLAMP3_HIDDEN_SIZE,
                load_m3=CLAMP3_LOAD_M3
            )
            
            checkpoint = torch.load(CLAMP3_WEIGHTS_PATH, map_location='cpu', weights_only=True)
            base_model.load_state_dict(checkpoint['model'])
            base_model = base_model.to(device)
            base_model.eval()
            
            # Evaluate baseline
            baseline_wikimt = evaluate_model(base_model, tokenizer, patchilizer, wikimt_data, device,
                                            "Baseline", "wikimt", text_field)            # Load LoRA adapter
            adapter_path = args.wikimt_adapter if args.wikimt_adapter else LORA_ABC_ADAPTER_PATH
            print(f"\nLoading ABC LoRA adapter from {adapter_path}")
            base_model.symbolic_model.base = PeftModel.from_pretrained(
                base_model.symbolic_model.base,
                adapter_path
            )
            base_model.symbolic_model.base = base_model.symbolic_model.base.to(device)
            base_model.eval()
            
            # Evaluate LoRA
            lora_wikimt = evaluate_model(base_model, tokenizer, patchilizer, wikimt_data, device,
                                        "LoRA", "wikimt", text_field)            # Store results with field name
            field_key = f'wikimt_{text_field}' if len(text_fields) > 1 else 'wikimt'
            all_results[field_key] = {
                'baseline': baseline_wikimt,
                'lora': lora_wikimt
            }
            
            # Print comparison
            comparison_title = f"WikiMT ({text_field})" if len(text_fields) > 1 else "WikiMT"
            print_comparison(comparison_title, baseline_wikimt, lora_wikimt)
            
            # Clean up model for next field
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save combined results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': all_results
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
