#!/usr/bin/env python3
"""
Evaluation Results Comparison Script
Compares the performance of baseline and LoRA-adapted models on WikiMT and Lakh datasets.

This script:
1. Loads evaluation results from both models
2. Computes performance differences
3. Generates comparison visualizations and reports
4. Provides statistical analysis
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np


class EvaluationComparison:
    """Compares baseline and LoRA model evaluation results."""
    
    def __init__(self):
        self.baseline_results = {}
        self.lora_results = {}
        self.comparison_results = {}
        
    def load_results(self, baseline_path, lora_path):
        """Load evaluation results from JSON files."""
        print("\n" + "="*80)
        print("LOADING EVALUATION RESULTS")
        print("="*80)
        
        # Load baseline results
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, 'r') as f:
                    self.baseline_results = json.load(f)
                print(f"✓ Loaded baseline results: {baseline_path}")
            except Exception as e:
                print(f"✗ Failed to load baseline results: {e}")
        else:
            print(f"⚠ Baseline results not found: {baseline_path}")
        
        # Load LoRA results
        if os.path.exists(lora_path):
            try:
                with open(lora_path, 'r') as f:
                    self.lora_results = json.load(f)
                print(f"✓ Loaded LoRA results: {lora_path}")
            except Exception as e:
                print(f"✗ Failed to load LoRA results: {e}")
        else:
            print(f"⚠ LoRA results not found: {lora_path}")
        
        return len(self.baseline_results) > 0 or len(self.lora_results) > 0
    
    def compare_metrics(self, baseline_metrics: Dict, lora_metrics: Dict) -> Dict:
        """Compare metrics between baseline and LoRA models."""
        comparison = {
            'baseline': baseline_metrics,
            'lora': lora_metrics,
            'differences': {},
            'improvements': {}
        }
        
        # Compute differences
        for key in baseline_metrics:
            if key in lora_metrics and isinstance(baseline_metrics[key], (int, float)):
                diff = lora_metrics[key] - baseline_metrics[key]
                pct_change = (diff / baseline_metrics[key] * 100) if baseline_metrics[key] != 0 else 0
                
                comparison['differences'][key] = {
                    'absolute': round(diff, 6),
                    'percent': round(pct_change, 2),
                    'improved': diff > 0
                }
                
                if diff > 0:
                    comparison['improvements'][key] = round(pct_change, 2)
        
        return comparison
    
    def analyze_dataset_comparison(self, dataset_name: str) -> Dict:
        """Analyze comparison for a specific dataset."""
        print(f"\n{'='*80}")
        print(f"ANALYZING {dataset_name.upper()}")
        print(f"{'='*80}")
        
        baseline_data = self.baseline_results.get(dataset_name, {})
        lora_data = self.lora_results.get(dataset_name, {})
        
        comparison = {
            'dataset': dataset_name,
            'baseline': baseline_data,
            'lora': lora_data,
            'metric_comparison': {}
        }
        
        # Compare metrics
        if baseline_data.get('metrics') and lora_data.get('metrics'):
            comparison['metric_comparison'] = self.compare_metrics(
                baseline_data['metrics'],
                lora_data['metrics']
            )
            
            # Print comparison
            print(f"\nBaseline Metrics:")
            for key, value in baseline_data['metrics'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            print(f"\nLoRA Metrics:")
            for key, value in lora_data['metrics'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            print(f"\nDifferences:")
            for key, diff_info in comparison['metric_comparison']['differences'].items():
                symbol = "↑" if diff_info['improved'] else "↓"
                print(f"  {key}: {symbol} {diff_info['absolute']:+.6f} ({diff_info['percent']:+.2f}%)")
        
        # Compare sample results
        baseline_samples = baseline_data.get('samples', [])
        lora_samples = lora_data.get('samples', [])
        
        if baseline_samples and lora_samples:
            comparison['sample_comparison'] = {
                'baseline_count': len(baseline_samples),
                'lora_count': len(lora_samples),
                'samples': []
            }
            
            # Match samples by ID and compare
            lora_by_id = {s['id']: s for s in lora_samples}
            
            for baseline_sample in baseline_samples[:10]:  # Compare first 10 samples
                sample_id = baseline_sample['id']
                if sample_id in lora_by_id:
                    lora_sample = lora_by_id[sample_id]
                    sim_diff = lora_sample['similarity'] - baseline_sample['similarity']
                    
                    comparison['sample_comparison']['samples'].append({
                        'id': sample_id,
                        'baseline_similarity': round(baseline_sample['similarity'], 4),
                        'lora_similarity': round(lora_sample['similarity'], 4),
                        'difference': round(sim_diff, 4)
                    })
        
        return comparison
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "="*80)
        print("GENERATING COMPARISON REPORT")
        print("="*80)
        
        # Analyze each dataset
        all_comparisons = {
            'timestamp': datetime.now().isoformat(),
            'baseline_results_count': len(self.baseline_results),
            'lora_results_count': len(self.lora_results),
            'datasets': {}
        }
        
        # Compare ABC dataset
        if 'abc_wikimt' in self.baseline_results or 'abc_wikimt' in self.lora_results:
            all_comparisons['datasets']['abc_wikimt'] = self.analyze_dataset_comparison('abc_wikimt')
        
        # Compare MTF dataset
        if 'mtf_lakh' in self.baseline_results or 'mtf_lakh' in self.lora_results:
            all_comparisons['datasets']['mtf_lakh'] = self.analyze_dataset_comparison('mtf_lakh')
        
        self.comparison_results = all_comparisons
        
        # Save report
        report_path = os.path.join(os.path.dirname(__file__), 'evaluation_comparison_report.json')
        with open(report_path, 'w') as f:
            json.dump(all_comparisons, f, indent=2)
        
        print(f"\n✓ Comparison report saved to: {report_path}")
        
        return all_comparisons
    
    def generate_summary(self):
        """Generate a human-readable summary."""
        print("\n" + "#"*80)
        print("# EVALUATION SUMMARY")
        print("#"*80)
        
        summary_path = os.path.join(os.path.dirname(__file__), 'evaluation_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLAMP3 MODEL EVALUATION SUMMARY\n")
            f.write("Baseline vs LoRA Adapters\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # ABC Dataset
            abc_comparison = self.comparison_results.get('datasets', {}).get('abc_wikimt', {})
            f.write("="*80 + "\n")
            f.write("ABC DATASET (WikiMT)\n")
            f.write("="*80 + "\n\n")
            
            if abc_comparison:
                baseline_metrics = abc_comparison.get('metric_comparison', {}).get('baseline', {})
                lora_metrics = abc_comparison.get('metric_comparison', {}).get('lora', {})
                improvements = abc_comparison.get('metric_comparison', {}).get('improvements', {})
                
                f.write("Baseline Model Metrics:\n")
                for key, value in baseline_metrics.items():
                    if isinstance(value, float):
                        f.write(f"  - {key}: {value:.4f}\n")
                    else:
                        f.write(f"  - {key}: {value}\n")
                f.write("\n")
                
                f.write("LoRA Adapter Metrics:\n")
                for key, value in lora_metrics.items():
                    if isinstance(value, float):
                        f.write(f"  - {key}: {value:.4f}\n")
                    else:
                        f.write(f"  - {key}: {value}\n")
                f.write("\n")
                
                if improvements:
                    f.write("Performance Improvements:\n")
                    for key, pct in improvements.items():
                        f.write(f"  - {key}: +{pct}%\n")
                f.write("\n")
            else:
                f.write("No comparison data available.\n\n")
            
            # MTF Dataset
            mtf_comparison = self.comparison_results.get('datasets', {}).get('mtf_lakh', {})
            f.write("="*80 + "\n")
            f.write("MTF DATASET (Lakh)\n")
            f.write("="*80 + "\n\n")
            
            if mtf_comparison:
                baseline_metrics = mtf_comparison.get('metric_comparison', {}).get('baseline', {})
                lora_metrics = mtf_comparison.get('metric_comparison', {}).get('lora', {})
                improvements = mtf_comparison.get('metric_comparison', {}).get('improvements', {})
                
                f.write("Baseline Model Metrics:\n")
                for key, value in baseline_metrics.items():
                    if isinstance(value, float):
                        f.write(f"  - {key}: {value:.4f}\n")
                    else:
                        f.write(f"  - {key}: {value}\n")
                f.write("\n")
                
                f.write("LoRA Adapter Metrics:\n")
                for key, value in lora_metrics.items():
                    if isinstance(value, float):
                        f.write(f"  - {key}: {value:.4f}\n")
                    else:
                        f.write(f"  - {key}: {value}\n")
                f.write("\n")
                
                if improvements:
                    f.write("Performance Improvements:\n")
                    for key, pct in improvements.items():
                        f.write(f"  - {key}: +{pct}%\n")
                f.write("\n")
            else:
                f.write("No comparison data available.\n\n")
            
            f.write("="*80 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*80 + "\n\n")
            f.write("The LoRA adapters provide specialized fine-tuning for ABC and MTF formats.\n")
            f.write("This evaluation compares their performance against the baseline model.\n")
        
        print(f"✓ Summary saved to: {summary_path}")
        
        # Print to console
        with open(summary_path, 'r') as f:
            print(f.read())


def main():
    """Main comparison routine."""
    print("\n" + "#"*80)
    print("# EVALUATION COMPARISON")
    print("# Baseline vs LoRA Adapters")
    print("#"*80)
    
    base_dir = os.path.dirname(__file__)
    
    baseline_report = os.path.join(base_dir, 'evaluation_baseline_report.json')
    lora_report = os.path.join(base_dir, 'evaluation_lora_report.json')
    
    comparator = EvaluationComparison()
    
    # Load results
    if not comparator.load_results(baseline_report, lora_report):
        print("\n✗ No evaluation results found.")
        print(f"Expected files:")
        print(f"  - {baseline_report}")
        print(f"  - {lora_report}")
        return False
    
    # Generate comparison report
    comparator.generate_comparison_report()
    
    # Generate summary
    comparator.generate_summary()
    
    print("\n✓ EVALUATION COMPARISON COMPLETE")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
