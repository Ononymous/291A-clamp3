"""
Visualize evaluation results from evaluation_results.json
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

# Extract text fields and metrics
text_fields = []
baseline_t2m_mrr = []
lora_t2m_mrr = []
baseline_m2t_mrr = []
lora_m2t_mrr = []

for key in ['wikimt_background', 'wikimt_analysis', 'wikimt_description', 'wikimt_scene']:
    field_name = key.replace('wikimt_', '').capitalize()
    text_fields.append(field_name)
    
    baseline = results[key]['baseline']
    lora = results[key]['lora']
    
    baseline_t2m_mrr.append(baseline['text_to_music']['MRR'])
    lora_t2m_mrr.append(lora['text_to_music']['MRR'])
    baseline_m2t_mrr.append(baseline['music_to_text']['MRR'])
    lora_m2t_mrr.append(lora['music_to_text']['MRR'])

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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

plt.suptitle('WikiMT-X Evaluation: Baseline vs LoRA Across Text Fields', 
             fontsize=16, fontweight='bold', y=1)
plt.tight_layout()

# Save figure
output_path = Path(__file__).parent / 'evaluation_results.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Graph saved to {output_path}")

# Show plot
plt.show()
