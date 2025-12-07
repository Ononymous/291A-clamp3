# CLaMP3 Evaluation - Quick Start Guide

## Overview

This evaluation framework compares two model variants on real music datasets:

### Models
1. **Baseline CLaMP3** - Universal model for all symbolic music formats
2. **LoRA-Adapted CLaMP3** - Specialized adapters for ABC notation and MTF (MIDI)

### Datasets  
1. **WikiMT-X** - Sheet music in ABC notation with rich text annotations (from HuggingFace)
2. **Lakh MIDI Dataset** - MIDI performances converted to MTF format with minimal annotations

## Quick Start (3 Steps)

### Step 1: Prepare Real Datasets
```bash
# Download WikiMT-X and convert Lakh MIDI files
python prepare_test_data.py --num_abc 1000 --num_mtf 1000

# This will:
# - Download 1000 samples from WikiMT-X HuggingFace dataset
# - Convert 1000 random MIDI files from lmd_matched to MTF format
# - Create metadata.json for both datasets
```

**Arguments:**
- `--num_abc N` - Number of ABC samples from WikiMT-X (default: 100)
- `--num_mtf N` - Number of MTF samples from Lakh (default: 100)
- `--skip_abc` - Skip ABC dataset preparation
- `--skip_mtf` - Skip MTF dataset preparation

### Step 2: Run Complete Evaluation
```bash
# Automated pipeline (recommended)
bash run_evaluation.sh

# Or run step-by-step:
python test_pipeline.py       # Verify everything works
python evaluate_baseline.py   # Test baseline model
python evaluate_lora.py       # Test LoRA adapters
python compare_results.py     # Compare results
```

### Step 3: View Results
```bash
# Human-readable summary
cat evaluation_summary.txt

# Detailed JSON reports
cat evaluation_comparison_report.json | python -m json.tool
```

## What Gets Evaluated

### ABC Dataset (WikiMT-X)
- **Content**: Sheet music notation from classical, folk, and popular music
- **Annotations**: Rich text descriptions, titles, composers, genres
- **Format**: ABC notation (text-based sheet music)
- **Models Tested**:
  - Baseline: Universal symbolic encoder
  - LoRA: ABC-specialized adapter

**Key Metrics:**
- Text-music semantic similarity (how well descriptions match music)
- Feature quality (embedding norms, consistency)

### MTF Dataset (Lakh MIDI)
- **Content**: MIDI performances from real recordings
- **Annotations**: Minimal (generic descriptions only)
- **Format**: MTF (MIDI Text Format) - symbolic performance signals
- **Models Tested**:
  - Baseline: Universal symbolic encoder  
  - LoRA: MTF-specialized adapter

**Key Metrics:**
- Text-music alignment (limited by minimal annotations)
- Feature representation quality

## Expected Results

### Strong LoRA Performance
- **ABC Dataset**: 5-20% improvement in text-music similarity
  - LoRA specializes in sheet music patterns
  - Rich annotations enable good semantic alignment
  
- **MTF Dataset**: 0-10% improvement
  - Minimal annotations limit text-music evaluation
  - Focuses on feature quality metrics

### What Success Looks Like
```
ABC DATASET (WikiMT-X):
  Baseline text-music similarity:  0.42
  LoRA text-music similarity:      0.51
  Improvement: +21.4% ✓

MTF DATASET (Lakh):
  Baseline text-music similarity:  0.35
  LoRA text-music similarity:      0.37
  Improvement: +5.7% ✓
```

## File Structure

```
data/test/
├── wikimt_abc/              # WikiMT-X dataset
│   ├── wikimt_0000.abc     # ABC notation files
│   ├── wikimt_0001.abc
│   ├── ...
│   └── metadata.json       # Rich annotations
│
├── lakh_mtf/               # Lakh MIDI dataset  
│   ├── lakh_0000.mtf      # MTF format files
│   ├── lakh_0001.mtf
│   ├── ...
│   └── metadata.json       # Minimal annotations
│
└── lmd_matched/            # Original MIDI files
    └── A/B/C/.../file.mid  # Hierarchical structure
```

## Troubleshooting

### datasets library not found
```bash
pip install datasets
# or
python -m pip install datasets
```

### HuggingFace authentication required
```bash
# Login to HuggingFace (if dataset requires auth)
huggingface-cli login
# Enter your token when prompted
```

### MIDI conversion fails
The script will create minimal fallback datasets if conversion fails.
Check `logs/midi2mtf_error_log.txt` for details.

### No MIDI files found
Ensure `/Users/kaitlynt/291A-clamp3/data/test/lmd_matched/` contains MIDI files.
Current count: 116,189 MIDI files detected.

## Configuration

Edit `prepare_test_data.py` to customize:

```python
# Default sample counts
--num_abc 100   # WikiMT-X samples
--num_mtf 100   # Lakh MIDI samples

# Paths (auto-configured)
WikiMT-X: data/test/wikimt_abc/
Lakh MTF: data/test/lakh_mtf/
Source MIDI: data/test/lmd_matched/
```

## Time Estimates

| Task | Time | Notes |
|------|------|-------|
| Install datasets lib | 1-2 min | First time only |
| Download WikiMT-X (100) | 30-60 sec | Depends on connection |
| Convert MIDI to MTF (100) | 2-5 min | Parallel processing |
| Test pipeline | 30 sec | First run downloads model |
| Evaluate baseline | 5-10 min | 100+100 samples |
| Evaluate LoRA | 5-10 min | Same as baseline |
| Compare results | <1 sec | JSON processing |
| **Total** | **15-25 min** | For 200 total samples |

## Scaling to Full Datasets

### Full WikiMT-X (1000 samples)
```bash
python prepare_test_data.py --num_abc 1000 --skip_mtf
# Time: ~5 min download + 30-60 min evaluation
```

### Full Lakh Subset (1000 MIDI files)
```bash
python prepare_test_data.py --num_mtf 1000 --skip_abc
# Time: ~15 min conversion + 30-60 min evaluation
```

### Both at Scale
```bash
python prepare_test_data.py --num_abc 1000 --num_mtf 1000
bash run_evaluation.sh
# Total time: ~2 hours for complete pipeline
```

## Next Steps

1. **Start small**: Run with 10-20 samples to verify pipeline
   ```bash
   python prepare_test_data.py --num_abc 10 --num_mtf 10
   python test_pipeline.py
   ```

2. **Medium scale**: 100 samples for meaningful results
   ```bash
   python prepare_test_data.py --num_abc 100 --num_mtf 100
   bash run_evaluation.sh
   ```

3. **Full evaluation**: 1000+ samples for publication-ready metrics
   ```bash
   python prepare_test_data.py --num_abc 1000 --num_mtf 1000
   bash run_evaluation.sh
   ```

## Key Evaluation Scripts

All scripts are documented with `--help`:

```bash
python prepare_test_data.py --help
python test_pipeline.py --help  
python evaluate_baseline.py --help
python evaluate_lora.py --help
python compare_results.py --help
```

## Support

- Full documentation: See `README.md` in main directory
- Dataset details: See `preprocessing/README.md`
- Model info: See `code/README.md`
- LoRA training: See `LORA_TRAINING_GUIDE.md`

## Summary

This evaluation uses **real datasets** (WikiMT-X + Lakh MIDI) to measure whether **LoRA specialization** improves performance over the **universal baseline model** for ABC notation and MTF (MIDI) formats.

**Start evaluation:**
```bash
python prepare_test_data.py --num_abc 100 --num_mtf 100
bash run_evaluation.sh
cat evaluation_summary.txt
```
