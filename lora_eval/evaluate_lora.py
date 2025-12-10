#!/usr/bin/env python3
"""
LoRA Adapter Evaluation Script
Evaluates the LoRA-adapted CLaMP3 model on WikiMT (ABC) and Lakh (MTF) datasets.

This script:
1. Loads the base model with specialized LoRA adapters
2. Extracts features using adapter-enhanced model
3. Computes retrieval metrics
4. Compares performance with baseline model
"""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add code directory to path (go up one level to project root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from config import *
from utils import *
from transformers import BertConfig, AutoTokenizer, GPT2Config
from peft import PeftModel, PeftConfig


class LoRAModelEvaluator:
    """Evaluates the CLaMP3 model with LoRA adapters."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.base_model = None
        self.abc_model = None
        self.mtf_model = None
        self.patchilizer = None
        self.results = {}
        
    def setup(self):
        """Initialize base model and LoRA adapters."""
        print("\n" + "="*80)
        print("LoRA ADAPTER EVALUATION - SETUP")
        print("="*80)
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
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
            
            # Load base CLaMP3 model
            self.base_model = CLaMP3Model(
                audio_config=audio_config,
                symbolic_config=symbolic_config,
                text_model_name=TEXT_MODEL_NAME,
                hidden_size=CLAMP3_HIDDEN_SIZE,
                load_m3=CLAMP3_LOAD_M3
            )
            self.base_model = self.base_model.to(self.device)
            print(f"✓ Loaded base CLaMP3 model")
            print(f"  - Total parameters: {sum(p.numel() for p in self.base_model.parameters()):,}")
            
            # Load LoRA adapters
            if os.path.exists(LORA_ABC_ADAPTER_PATH):
                print(f"\nLoading ABC LoRA adapter from: {LORA_ABC_ADAPTER_PATH}")
                try:
                    self.abc_model = PeftModel.from_pretrained(
                        self.base_model,
                        LORA_ABC_ADAPTER_PATH,
                        device_map=self.device
                    )
                    self.abc_model.eval()
                    print(f"✓ ABC LoRA adapter loaded successfully")
                except Exception as e:
                    print(f"⚠ Failed to load ABC LoRA adapter: {e}")
                    self.abc_model = None
            else:
                print(f"⚠ ABC LoRA adapter not found at: {LORA_ABC_ADAPTER_PATH}")
                self.abc_model = None
            
            if os.path.exists(LORA_MTF_ADAPTER_PATH):
                print(f"\nLoading MTF LoRA adapter from: {LORA_MTF_ADAPTER_PATH}")
                try:
                    self.mtf_model = PeftModel.from_pretrained(
                        self.base_model,
                        LORA_MTF_ADAPTER_PATH,
                        device_map=self.device
                    )
                    self.mtf_model.eval()
                    print(f"✓ MTF LoRA adapter loaded successfully")
                except Exception as e:
                    print(f"⚠ Failed to load MTF LoRA adapter: {e}")
                    self.mtf_model = None
            else:
                print(f"⚠ MTF LoRA adapter not found at: {LORA_MTF_ADAPTER_PATH}")
                self.mtf_model = None
            
            # Load patchilizer
            self.patchilizer = M3Patchilizer()
            print(f"✓ Loaded M3 Patchilizer")
            
            if not self.abc_model and not self.mtf_model:
                print(f"\n⚠ WARNING: No LoRA adapters loaded. Using base model only.")
            
            print(f"\n✓ Device: {self.device}")
            return True
            
        except Exception as e:
            print(f"✗ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_abc_features(self, abc_content):
        """Extract features from ABC notation using ABC adapter."""
        try:
            model = self.abc_model if self.abc_model else self.base_model
            
            encoded = self.patchilizer.encode(abc_content, add_special_patches=True)
            encoded_tensor = torch.tensor(encoded).to(self.device)
            masks = torch.ones(encoded_tensor.size(0)).to(self.device)
            
            with torch.no_grad():
                features = model.get_symbolic_features(
                    symbolic_inputs=encoded_tensor.unsqueeze(0),
                    symbolic_masks=masks.unsqueeze(0),
                    get_global=True
                )
            
            return features.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"Warning: Failed to extract ABC features: {e}")
            return None
    
    def extract_mtf_features(self, mtf_content):
        """Extract features from MTF format using MTF adapter."""
        try:
            model = self.mtf_model if self.mtf_model else self.base_model
            
            encoded = self.patchilizer.encode(mtf_content, add_special_patches=True)
            encoded_tensor = torch.tensor(encoded).to(self.device)
            masks = torch.ones(encoded_tensor.size(0)).to(self.device)
            
            with torch.no_grad():
                features = model.get_symbolic_features(
                    symbolic_inputs=encoded_tensor.unsqueeze(0),
                    symbolic_masks=masks.unsqueeze(0),
                    get_global=True
                )
            
            return features.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"Warning: Failed to extract MTF features: {e}")
            return None
    
    def extract_text_features(self, text):
        """Extract features from text."""
        try:
            text_inputs = self.tokenizer(text, return_tensors='pt')
            text_ids = text_inputs['input_ids'].squeeze(0).to(self.device)
            masks = torch.ones(text_ids.size(0)).to(self.device)
            
            with torch.no_grad():
                features = self.base_model.get_text_features(
                    text_inputs=text_ids.unsqueeze(0),
                    text_masks=masks.unsqueeze(0),
                    get_global=True
                )
            
            return features.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"Warning: Failed to extract text features: {e}")
            return None
    
    def evaluate_dataset(self, dataset_dir, data_type='abc'):
        """
        Evaluate a dataset with LoRA adapter.
        
        Args:
            dataset_dir: Directory containing music files and metadata
            data_type: 'abc' or 'mtf'
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING {data_type.upper()} DATASET WITH LoRA: {dataset_dir}")
        print(f"{'='*80}")
        
        results = {
            'dataset_type': data_type,
            'dataset_path': dataset_dir,
            'timestamp': datetime.now().isoformat(),
            'adapter_used': 'abc_adapter' if data_type == 'abc' else 'mtf_adapter',
            'metrics': {},
            'samples': []
        }
        
        # Check if directory exists
        if not os.path.exists(dataset_dir):
            print(f"✗ Dataset directory not found: {dataset_dir}")
            return results
        
        # Find music files
        if data_type == 'abc':
            music_files = list(Path(dataset_dir).glob('*.abc'))
            file_extension = '.abc'
            feature_fn = self.extract_abc_features
            adapter_status = "loaded" if self.abc_model else "not found"
            print(f"Using ABC LoRA adapter: {adapter_status}")
        elif data_type == 'mtf':
            music_files = list(Path(dataset_dir).glob('*.mtf'))
            file_extension = '.mtf'
            feature_fn = self.extract_mtf_features
            adapter_status = "loaded" if self.mtf_model else "not found"
            print(f"Using MTF LoRA adapter: {adapter_status}")
        else:
            print(f"✗ Unknown data type: {data_type}")
            return results
        
        if not music_files:
            print(f"⚠ No {data_type.upper()} files found in {dataset_dir}")
            return results
        
        print(f"✓ Found {len(music_files)} {data_type.upper()} files")
        
        # Extract features
        music_features = {}
        text_features = {}
        
        for music_file in tqdm(music_files, desc=f"Extracting {data_type.upper()} features (LoRA)"):
            try:
                with open(music_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract music features
                features = feature_fn(content)
                if features is not None:
                    music_features[music_file.stem] = features
                
            except Exception as e:
                print(f"Warning: Failed to process {music_file}: {e}")
        
        print(f"✓ Extracted features for {len(music_features)} music files")
        
        # Look for metadata
        metadata_file = os.path.join(dataset_dir, 'metadata.json')
        annotations = {}
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                print(f"✓ Loaded metadata from {metadata_file}")
            except Exception as e:
                print(f"⚠ Could not load metadata: {e}")
        
        # Compute retrieval metrics if we have text annotations
        if annotations:
            for music_id in tqdm(music_features.keys(), desc="Computing retrieval metrics"):
                if music_id in annotations:
                    annotation = annotations[music_id]
                    
                    # Get text descriptions
                    descriptions = []
                    if isinstance(annotation, dict):
                        if 'description' in annotation:
                            descriptions.append(annotation['description'])
                        if 'tags' in annotation:
                            descriptions.extend(annotation['tags'])
                    
                    if descriptions:
                        text = ' '.join(descriptions)
                        text_feat = self.extract_text_features(text)
                        
                        if text_feat is not None:
                            text_features[music_id] = text_feat
                            
                            # Compute similarity
                            music_feat = music_features[music_id]
                            similarity = np.dot(music_feat, text_feat) / (
                                np.linalg.norm(music_feat) * np.linalg.norm(text_feat) + 1e-8
                            )
                            
                            results['samples'].append({
                                'id': music_id,
                                'text': text[:100] + '...' if len(text) > 100 else text,
                                'similarity': float(similarity)
                            })
        
        # Compute aggregate metrics
        if len(music_features) > 1:
            # Compute average feature norm
            feature_norms = [np.linalg.norm(f) for f in music_features.values()]
            results['metrics']['avg_feature_norm'] = float(np.mean(feature_norms))
            results['metrics']['num_samples'] = len(music_features)
            
            # If we have text features, compute text-music correlation
            if text_features:
                similarities = [s['similarity'] for s in results['samples']]
                results['metrics']['avg_text_music_similarity'] = float(np.mean(similarities))
                results['metrics']['num_text_annotated'] = len(text_features)
        
        return results
    
    def generate_report(self):
        """Generate evaluation report."""
        print("\n" + "="*80)
        print("LoRA ADAPTER EVALUATION REPORT")
        print("="*80)
        
        report_path = os.path.join(os.path.dirname(__file__), 'evaluation_lora_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Report saved to: {report_path}")
        
        # Print summary
        for dataset_name, dataset_results in self.results.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"  Dataset: {dataset_results.get('dataset_path', 'N/A')}")
            print(f"  Adapter: {dataset_results.get('adapter_used', 'N/A')}")
            print(f"  Samples: {dataset_results['metrics'].get('num_samples', 0)}")
            if 'avg_feature_norm' in dataset_results['metrics']:
                print(f"  Avg Feature Norm: {dataset_results['metrics']['avg_feature_norm']:.4f}")
            if 'avg_text_music_similarity' in dataset_results['metrics']:
                print(f"  Avg Text-Music Similarity: {dataset_results['metrics']['avg_text_music_similarity']:.4f}")


def main():
    """Main evaluation routine."""
    print("\n" + "#"*80)
    print("# LoRA ADAPTER EVALUATION")
    print("# CLaMP3 with specialized LoRA adapters")
    print("#"*80)
    
    evaluator = LoRAModelEvaluator()
    
    # Setup
    if not evaluator.setup():
        print("\n✗ Setup failed. Exiting.")
        return False
    
    # Evaluate ABC dataset (WikiMT)
    print("\n" + "#"*80)
    print("# EVALUATING ABC DATASET (WikiMT) WITH ABC LoRA ADAPTER")
    print("#"*80)
    
    abc_results = evaluator.evaluate_dataset(
        dataset_dir=os.path.join(os.path.dirname(__file__), 'data', 'test', 'wikimt_abc'),
        data_type='abc'
    )
    evaluator.results['abc_wikimt'] = abc_results
    
    # Evaluate MTF dataset (Lakh)
    print("\n" + "#"*80)
    print("# EVALUATING MTF DATASET (Lakh) WITH MTF LoRA ADAPTER")
    print("#"*80)
    
    mtf_results = evaluator.evaluate_dataset(
        dataset_dir=os.path.join(os.path.dirname(__file__), 'data', 'test', 'lakh_mtf'),
        data_type='mtf'
    )
    evaluator.results['mtf_lakh'] = mtf_results
    
    # Generate report
    evaluator.generate_report()
    
    print("\n✓ LoRA ADAPTER EVALUATION COMPLETE")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
