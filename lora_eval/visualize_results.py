"""
Visualize evaluation results from evaluation_results.json
Creates separate graphs for MidiCaps and WikiMT-X
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path(__file__).parent / 'evaluation_results.json'
with open(results_path, 'r') as f:
    data = json.load(f)

results = data['results']

# ============================================================================
# GRAPH 1: MidiCaps
# ============================================================================

if 'midicaps' in results:
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    midicaps_baseline = results['midicaps']['baseline']
    midicaps_lora = results['midicaps']['lora']
    
    metrics = ['Hit@1', 'Hit@5', 'Hit@10']
    
    # Text-to-Music
    baseline_t2m = [midicaps_baseline['text_to_music'][m] for m in metrics]
    lora_t2m = [midicaps_lora['text_to_music'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_t2m, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, lora_t2m, width, label='LoRA', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Text-to-Music Retrieval', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 100)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Music-to-Text
    baseline_m2t = [midicaps_baseline['music_to_text'][m] for m in metrics]
    lora_m2t = [midicaps_lora['music_to_text'][m] for m in metrics]
    
    bars3 = ax2.bar(x - width/2, baseline_m2t, width, label='Baseline', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, lora_m2t, width, label='LoRA', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Music-to-Text Retrieval', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('MidiCaps (MTF) Hit Rates: Baseline vs LoRA', 
                 fontsize=16, fontweight='bold', y=1)
    plt.tight_layout()
    
    output_path1 = Path(__file__).parent / 'evaluation_midicaps_hits.png'
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"✓ MidiCaps hit rates graph saved to {output_path1}")

# ============================================================================
# GRAPH 2: WikiMT-X
# ============================================================================

# Extract text fields and metrics (WikiMT + MidiCaps)
text_fields = []
baseline_t2m_mrr = []
lora_t2m_mrr = []
baseline_m2t_mrr = []
lora_m2t_mrr = []

# Add MidiCaps first
if 'midicaps' in results:
    text_fields.append('MidiCaps')
    baseline_t2m_mrr.append(results['midicaps']['baseline']['text_to_music']['MRR'])
    lora_t2m_mrr.append(results['midicaps']['lora']['text_to_music']['MRR'])
    baseline_m2t_mrr.append(results['midicaps']['baseline']['music_to_text']['MRR'])
    lora_m2t_mrr.append(results['midicaps']['lora']['music_to_text']['MRR'])

# Add WikiMT-X fields
for key in ['wikimt_background', 'wikimt_analysis', 'wikimt_description', 'wikimt_scene']:
    field_name = key.replace('wikimt_', '').capitalize()
    text_fields.append(field_name)
    
    baseline = results[key]['baseline']
    lora = results[key]['lora']
    
    baseline_t2m_mrr.append(baseline['text_to_music']['MRR'])
    lora_t2m_mrr.append(lora['text_to_music']['MRR'])
    baseline_m2t_mrr.append(baseline['music_to_text']['MRR'])
    lora_m2t_mrr.append(lora['music_to_text']['MRR'])

# Create WikiMT-X figure with 2 subplots
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(text_fields))
width = 0.35

# Text-to-Music subplot
bars1 = ax1.bar(x - width/2, baseline_t2m_mrr, width, label='Baseline', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, lora_t2m_mrr, width, label='LoRA', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('MRR Score', fontsize=12, fontweight='bold')
ax1.set_title('Text-to-Music Retrieval', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(text_fields, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(lora_t2m_mrr) * 1.15)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Music-to-Text subplot
bars3 = ax2.bar(x - width/2, baseline_m2t_mrr, width, label='Baseline', color='#3498db', alpha=0.8)
bars4 = ax2.bar(x + width/2, lora_m2t_mrr, width, label='LoRA', color='#e74c3c', alpha=0.8)

ax2.set_ylabel('MRR Score', fontsize=12, fontweight='bold')
ax2.set_title('Music-to-Text Retrieval', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(text_fields, fontsize=11)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(lora_m2t_mrr) * 1.15)

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('MRR Comparison: MidiCaps (MTF) and WikiMT-X (ABC) - Baseline vs LoRA', 
             fontsize=16, fontweight='bold', y=0)
plt.tight_layout()

# Save WikiMT-X MRR figure
output_path2 = Path(__file__).parent / 'evaluation_wikimt_mrr.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✓ WikiMT-X MRR graph saved to {output_path2}")

# ============================================================================
# GRAPH 3: WikiMT-X Background Hit Rates
# ============================================================================

if 'wikimt_background' in results:
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    background_baseline = results['wikimt_background']['baseline']
    background_lora = results['wikimt_background']['lora']
    
    metrics = ['Hit@1', 'Hit@5', 'Hit@10']
    
    # Text-to-Music
    baseline_t2m = [background_baseline['text_to_music'][m] for m in metrics]
    lora_t2m = [background_lora['text_to_music'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_t2m, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, lora_t2m, width, label='LoRA', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Text-to-Music Retrieval', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(max(baseline_t2m), max(lora_t2m)) * 1.15)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Music-to-Text
    baseline_m2t = [background_baseline['music_to_text'][m] for m in metrics]
    lora_m2t = [background_lora['music_to_text'][m] for m in metrics]
    
    bars3 = ax2.bar(x - width/2, baseline_m2t, width, label='Baseline', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, lora_m2t, width, label='LoRA', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Music-to-Text Retrieval', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(max(baseline_m2t), max(lora_m2t)) * 1.15)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('WikiMT-X (ABC) Background Field Hit Rates: Baseline vs LoRA', 
                 fontsize=16, fontweight='bold', y=1)
    plt.tight_layout()
    
    output_path3 = Path(__file__).parent / 'evaluation_wikimt_background_hits.png'
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"✓ WikiMT-X background hit rates graph saved to {output_path3}")

print("\n✓ All graphs generated successfully!")
