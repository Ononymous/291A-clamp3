# LoRA Adapter Evaluation

This folder contains the evaluation script for trained LoRA adapters.

## evaluate_adapters.py

Combined evaluation script for both MidiCaps (MIDI/MTF) and WikiMT-X (ABC) test sets.

### Usage

**Evaluate both datasets (default):**
```bash
python lora_eval/evaluate_adapters.py
```

**Evaluate only MidiCaps:**
```bash
python lora_eval/evaluate_adapters.py --eval_midicaps
```

**Evaluate only WikiMT-X (default: analysis field):**
```bash
python lora_eval/evaluate_adapters.py --eval_wikimt
```

**Evaluate WikiMT-X with specific text field:**
```bash
python lora_eval/evaluate_adapters.py --eval_wikimt --wikimt_text_field background
python lora_eval/evaluate_adapters.py --eval_wikimt --wikimt_text_field description
python lora_eval/evaluate_adapters.py --eval_wikimt --wikimt_text_field scene
```

**Evaluate WikiMT-X with ALL text fields (background, analysis, description, scene):**
```bash
python lora_eval/evaluate_adapters.py --eval_wikimt --wikimt_text_field all
```

**Use custom adapter paths:**
```bash
python lora_eval/evaluate_adapters.py \
  --midicaps_adapter path/to/mtf_adapter \
  --wikimt_adapter path/to/abc_adapter
```

### WikiMT-X Text Fields

WikiMT-X dataset provides 4 different text perspectives for each song:
- **background**: Cultural and historical context
- **analysis**: Technical music analysis (default)
- **description**: General musical description
- **scene**: Abstract scenario depiction

### Datasets

- **MidiCaps**: 1000 test samples from `data/midicaps_splits/midicaps_test.jsonl`
  - Uses MTF (MIDI Text Format) and text descriptions
  - Evaluates MTF adapter (trained on 420K MIDI-text pairs)

- **WikiMT-X**: 1000 test samples from `data/test/wikimt-x/wikimt-x-public.jsonl`
  - Uses ABC notation (leadsheets) and analysis text
  - Evaluates ABC adapter (trained on PDMX sheet music dataset)

### Outputs

- Console: Comparison tables showing baseline vs LoRA performance
- File: `evaluation_results.json` with detailed metrics for both datasets

### Metrics

For both Text-to-Music and Music-to-Text retrieval:
- **MRR**: Mean Reciprocal Rank
- **Hit@1**: Percentage of correct matches in top-1
- **Hit@5**: Percentage of correct matches in top-5
- **Hit@10**: Percentage of correct matches in top-10
