#!/usr/bin/env python3
"""
Evaluate LoRA adapter on MidiCaps test split.
Compares baseline vs LoRA-adapted model on the held-out 1000 test samples.
"""

import os
import sys
import torch
import numpy as np
import json
import random
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

# Test data path
MIDICAPS_TEST_JSONL = os.path.join(os.path.dirname(__file__), '..', 'data', 'midicaps_splits', 'midicaps_test.jsonl')


def load_test_data(jsonl_path, max_samples=None):
    """Load MidiCaps test data."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                entry = json.loads(line)
                # Add index for evaluation
                entry['index'] = i
                data.append(entry)
    return data


def extract_features(model, tokenizer, patchilizer, entries, device):
    """Extract text and music features from entries."""
    text_features = []
    music_features = []
    valid_indices = []
    
    print("Extracting features...")
    for entry in tqdm(entries, desc="Processing"):
        # Extract text features
        description = entry.get('description', '')
        if not description:
            continue
            
        inputs = tokenizer(description, return_tensors="pt", max_length=MAX_TEXT_LENGTH, 
                          truncation=True, padding='max_length')
        text_inputs = inputs['input_ids'].to(device)
        text_masks = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            text_feat = model.get_text_features(text_inputs, text_masks, get_global=True)
        
        # Extract music features
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
        
        # Encode music
        encoded = patchilizer.encode(music_content, add_special_patches=True, truncate=True)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
        masks = torch.ones(encoded_tensor.size(1)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            music_feat = model.get_symbolic_features(symbolic_inputs=encoded_tensor,
                                                    symbolic_masks=masks,
                                                    get_global=True)
        
        text_features.append(text_feat.cpu().numpy().flatten())
        music_features.append(music_feat.cpu().numpy().flatten())
        valid_indices.append(entry['index'])
    
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


def evaluate_model(model, tokenizer, patchilizer, test_data, device, model_name):
    """Evaluate a model on test data."""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}")
    print(f"{'='*80}")
    
    text_features, music_features, valid_indices = extract_features(
        model, tokenizer, patchilizer, test_data, device
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
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS - {model_name}")
    print(f"{'='*80}")
    print(f"Samples: {len(valid_indices)}")
    print(f"\nText-to-Music:")
    print(f"  MRR:    {t2m_mrr:.4f}")
    print(f"  Hit@1:  {t2m_hits[1]:.2f}%")
    print(f"  Hit@5:  {t2m_hits[5]:.2f}%")
    print(f"  Hit@10: {t2m_hits[10]:.2f}%")
    print(f"\nMusic-to-Text:")
    print(f"  MRR:    {m2t_mrr:.4f}")
    print(f"  Hit@1:  {m2t_hits[1]:.2f}%")
    print(f"  Hit@5:  {m2t_hits[5]:.2f}%")
    print(f"  Hit@10: {m2t_hits[10]:.2f}%")
    
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load test data
    print(f"\nLoading test data from {MIDICAPS_TEST_JSONL}")
    test_data = load_test_data(MIDICAPS_TEST_JSONL)
    print(f"Loaded {len(test_data)} test samples")
    
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
    baseline_results = evaluate_model(base_model, tokenizer, patchilizer, test_data, device, "Baseline")
    
    # Load LoRA adapter
    print(f"\nLoading LoRA adapter from {LORA_MTF_ADAPTER_PATH}")
    base_model.symbolic_model.base = PeftModel.from_pretrained(
        base_model.symbolic_model.base,
        LORA_MTF_ADAPTER_PATH
    )
    base_model.symbolic_model.base = base_model.symbolic_model.base.to(device)
    base_model.eval()
    
    # Evaluate LoRA
    lora_results = evaluate_model(base_model, tokenizer, patchilizer, test_data, device, "LoRA")
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    for direction in ['text_to_music', 'music_to_text']:
        direction_name = "Text-to-Music" if direction == "text_to_music" else "Music-to-Text"
        print(f"\n{direction_name}:")
        for metric in ['MRR', 'Hit@1', 'Hit@5', 'Hit@10']:
            baseline_val = baseline_results[direction][metric]
            lora_val = lora_results[direction][metric]
            diff = lora_val - baseline_val
            pct = (diff / baseline_val * 100) if baseline_val > 0 else 0
            
            sign = "+" if diff > 0 else ""
            print(f"  {metric:6s}: {baseline_val:7.4f} → {lora_val:7.4f}  ({sign}{diff:+7.4f}, {sign}{pct:+6.2f}%)")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test_samples': len(test_data),
        'baseline': baseline_results,
        'lora': lora_results
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'midicaps_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
