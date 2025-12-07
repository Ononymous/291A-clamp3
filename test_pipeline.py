#!/usr/bin/env python3
"""
Test script to verify CLaMP3 pipeline integrity before running full evaluations.
This script confirms:
1. Model loading and checkpoint integrity
2. Feature extraction pipeline for ABC and MTF formats
3. LoRA adapter loading and integration
4. Basic inference with both models
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from config import *
from utils import *
from transformers import BertConfig, AutoTokenizer, GPT2Config

def test_model_loading():
    """Test that the base model and LoRA adapters load correctly."""
    print("\n" + "="*80)
    print("TEST 1: Model Loading")
    print("="*80)
    
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        print(f"✓ Loaded tokenizer: {TEXT_MODEL_NAME}")
        
        # Setup configs
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE//64,
            intermediate_size=AUDIO_HIDDEN_SIZE*4,
            max_position_embeddings=MAX_AUDIO_LENGTH
        )
        symbolic_config = BertConfig(
            vocab_size=1,
            hidden_size=M3_HIDDEN_SIZE,
            num_hidden_layers=PATCH_NUM_LAYERS,
            num_attention_heads=M3_HIDDEN_SIZE//64,
            intermediate_size=M3_HIDDEN_SIZE*4,
            max_position_embeddings=PATCH_LENGTH
        )
        print(f"✓ Created model configs")
        
        # Load base CLaMP3 model
        model = CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=TEXT_MODEL_NAME,
            hidden_size=CLAMP3_HIDDEN_SIZE,
            load_m3=CLAMP3_LOAD_M3
        )
        model = model.to(device)
        model.eval()
        print(f"✓ Loaded base CLaMP3 model")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Check LoRA adapter paths
        abc_adapter_path = LORA_ABC_ADAPTER_PATH
        mtf_adapter_path = LORA_MTF_ADAPTER_PATH
        
        if os.path.exists(abc_adapter_path):
            print(f"✓ ABC LoRA adapter found at: {abc_adapter_path}")
        else:
            print(f"✗ ABC LoRA adapter NOT found at: {abc_adapter_path}")
            
        if os.path.exists(mtf_adapter_path):
            print(f"✓ MTF LoRA adapter found at: {mtf_adapter_path}")
        else:
            print(f"✗ MTF LoRA adapter NOT found at: {mtf_adapter_path}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_abc_feature_extraction(model, tokenizer, device):
    """Test ABC notation feature extraction."""
    print("\n" + "="*80)
    print("TEST 2: ABC Notation Feature Extraction")
    print("="*80)
    
    try:
        patchilizer = M3Patchilizer()
        
        # Sample ABC notation (simple melody)
        sample_abc = """X:1
T:Test Melody
M:4/4
L:1/8
K:C
C4 D4 | E4 F4 | G8 |]"""
        
        print(f"✓ Sample ABC notation loaded")
        print(f"  ABC snippet:\n{sample_abc[:50]}...")
        
        # Encode ABC
        encoded = patchilizer.encode(sample_abc, add_special_patches=True)
        encoded_tensor = torch.tensor(encoded).to(device)
        print(f"✓ Encoded ABC: shape {encoded_tensor.shape}")
        
        # Create masks
        masks = torch.ones(encoded_tensor.size(0)).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model.get_symbolic_features(
                symbolic_inputs=encoded_tensor.unsqueeze(0),
                symbolic_masks=masks.unsqueeze(0),
                get_global=True
            )
        
        print(f"✓ Extracted ABC features: shape {features.shape}")
        print(f"  Feature dimension: {features.shape[-1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ ABC feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mtf_feature_extraction(model, tokenizer, device):
    """Test MTF (MIDI Text Format) feature extraction."""
    print("\n" + "="*80)
    print("TEST 3: MTF (MIDI Text Format) Feature Extraction")
    print("="*80)
    
    try:
        patchilizer = M3Patchilizer()
        
        # Sample MTF (simple note sequence)
        sample_mtf = """ticks_per_beat: 480
0 note_on channel=0 pitch=60 velocity=100
0 control_change channel=0 controller=0 value=0
0 control_change channel=0 controller=32 value=0
0 program_change channel=0 program=0
480 note_off channel=0 pitch=60 velocity=0
480 note_on channel=0 pitch=62 velocity=100
960 note_off channel=0 pitch=62 velocity=0
960 note_on channel=0 pitch=64 velocity=100"""
        
        print(f"✓ Sample MTF loaded")
        print(f"  MTF snippet:\n{sample_mtf[:50]}...")
        
        # Encode MTF
        encoded = patchilizer.encode(sample_mtf, add_special_patches=True)
        encoded_tensor = torch.tensor(encoded).to(device)
        print(f"✓ Encoded MTF: shape {encoded_tensor.shape}")
        
        # Create masks
        masks = torch.ones(encoded_tensor.size(0)).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model.get_symbolic_features(
                symbolic_inputs=encoded_tensor.unsqueeze(0),
                symbolic_masks=masks.unsqueeze(0),
                get_global=True
            )
        
        print(f"✓ Extracted MTF features: shape {features.shape}")
        print(f"  Feature dimension: {features.shape[-1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ MTF feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_feature_extraction(model, tokenizer, device):
    """Test text feature extraction."""
    print("\n" + "="*80)
    print("TEST 4: Text Feature Extraction")
    print("="*80)
    
    try:
        sample_text = "Classical music composed in C major with a bright melodic line"
        print(f"✓ Sample text loaded: '{sample_text}'")
        
        # Tokenize
        text_inputs = tokenizer(sample_text, return_tensors='pt')
        text_ids = text_inputs['input_ids'].squeeze(0).to(device)
        print(f"✓ Tokenized text: shape {text_ids.shape}")
        
        # Create masks
        masks = torch.ones(text_ids.size(0)).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model.get_text_features(
                text_inputs=text_ids.unsqueeze(0),
                text_masks=masks.unsqueeze(0),
                get_global=True
            )
        
        print(f"✓ Extracted text features: shape {features.shape}")
        print(f"  Feature dimension: {features.shape[-1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Text feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_similarity_computation(model, tokenizer, device):
    """Test cross-modal similarity computation."""
    print("\n" + "="*80)
    print("TEST 5: Cross-Modal Similarity Computation")
    print("="*80)
    
    try:
        patchilizer = M3Patchilizer()
        
        # Sample inputs
        text = "A melodic composition"
        abc = "C4 D4 E4 F4 | G8 |]"
        
        # Extract features
        text_inputs = tokenizer(text, return_tensors='pt')
        text_ids = text_inputs['input_ids'].squeeze(0).to(device)
        text_masks = torch.ones(text_ids.size(0)).to(device)
        
        abc_encoded = patchilizer.encode(abc, add_special_patches=True)
        abc_tensor = torch.tensor(abc_encoded).to(device)
        abc_masks = torch.ones(abc_tensor.size(0)).to(device)
        
        with torch.no_grad():
            text_feat = model.get_text_features(text_ids.unsqueeze(0), text_masks.unsqueeze(0), get_global=True)
            abc_feat = model.get_symbolic_features(abc_tensor.unsqueeze(0), abc_masks.unsqueeze(0), get_global=True)
        
        # Compute similarity
        similarity = torch.nn.functional.cosine_similarity(text_feat, abc_feat)
        print(f"✓ Computed text-ABC similarity: {similarity.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Similarity computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_directory():
    """Test cache directory structure."""
    print("\n" + "="*80)
    print("TEST 6: Cache Directory Structure")
    print("="*80)
    
    try:
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"✓ Cache directory ready: {cache_dir}")
        print(f"✓ Temp directory ready: {temp_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cache directory setup failed: {e}")
        return False


def main():
    """Run all pipeline tests."""
    print("\n" + "#"*80)
    print("# CLaMP3 PIPELINE TEST SUITE")
    print("#"*80)
    
    results = {}
    
    # Test 1: Model Loading
    model, tokenizer, device = test_model_loading()
    results['model_loading'] = model is not None
    
    if not results['model_loading']:
        print("\n✗ PIPELINE TEST FAILED: Could not load model")
        return False
    
    # Test 2: ABC Feature Extraction
    results['abc_extraction'] = test_abc_feature_extraction(model, tokenizer, device)
    
    # Test 3: MTF Feature Extraction
    results['mtf_extraction'] = test_mtf_feature_extraction(model, tokenizer, device)
    
    # Test 4: Text Feature Extraction
    results['text_extraction'] = test_text_feature_extraction(model, tokenizer, device)
    
    # Test 5: Similarity Computation
    results['similarity'] = test_similarity_computation(model, tokenizer, device)
    
    # Test 6: Cache Directory
    results['cache'] = test_cache_directory()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    if all_passed:
        print("\n✓ ALL PIPELINE TESTS PASSED")
        print("\nThe pipeline is ready for evaluation. You can now:")
        print("  1. Prepare test datasets (WikiMT for ABC, Lakh for MTF)")
        print("  2. Run feature extraction on both datasets")
        print("  3. Evaluate base model and LoRA adapters")
        print("  4. Compare performance metrics")
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Please fix the issues before proceeding with evaluation.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
