# **CLaMP 3 with LoRA: Specialized Symbolic Music Encoder**

By Gen Tamada and Kaitlyn Tom

## **Overview**

This project extends [**CLaMP 3**](https://sanderwood.github.io/clamp3/) with [**LoRA (Low-Rank Adaptation)**](https://github.com/microsoft/LoRA) on their symbolic music encoder to enable specialized learning for **sheet music (ABC notation)** and **MIDI** formats. By fine-tuning only adapter layers (0.05% of parameters), we achieve efficient specialization to symbolic modalities while maintaining the pre-trained knowledge of the base model.

## **Project Objectives**

- Improve symbolic music (ABC & MIDI) retrieval performance over base CLaMP 3
- Demonstrate efficient fine-tuning via LoRA on the symbolic encoder (221K trainable params out of 457M total trainable parameters)
- Train on large-scale datasets: **PDMX** (sheet music) and **MidiCaps** (420K MIDI-text pairs)
- Evaluate on specialized test sets (MidiCaps test, WikiMT)  

## **What Can This Project Do?**

 **Cross-Modal Music Retrieval**: All the original functionalities of CLaMP 3  
 **Efficient Fine-Tuning**: Adapt base CLaMP 3 with minimal parameters using LoRA  
 **Specialized Evaluation**: Performance metrics on publically-available symbolic music datasets

## **Installation**

### **Prerequisites**
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- PyTorch 2.0+

### **Setup Environment (venv)**

**1. Create a virtual environment:**

```bash
python -m venv clamp3-lora
source clamp3-lora/bin/activate
python -m pip install --upgrade pip
```

**2. Install Dependencies and PyTorch with CUDA:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## **Data Preparation**

This project uses **PDMX** dataset for training and **Lakh MIDI** + **WikiMT** for evaluation.

### **Step 1: Parse PDMX Metadata**

Extract metadata from the PDMX CSV file:

```bash
python preprocessing/parse_pdmx_csv.py \
  --csv_path data/PDMX.csv \
  --output_dir data/processed \
  --base_dir data
```

**Output:** Metadata JSON files in `data/processed/`

### **Step 2: Convert MusicXML to Interleaved ABC Notation**

Convert MusicXML files to standard ABC notation, then to Interleaved ABC format (required by CLaMP3):

```bash
# Step 2a: MusicXML → Standard ABC
python preprocessing/abc/batch_xml2abc.py data/mxl data/abc

# Step 2b: Standard ABC → Interleaved ABC  
python preprocessing/abc/batch_interleaved_abc.py data/abc data/abc_standard
```

**Inputs:** MusicXML files in `data/mxl/`  
**Outputs:** Interleaved ABC files in `data/abc_standard/`

### **Step 3: Convert MIDI to MTF (MIDI Text Format)**

Convert MIDI files to MTF format (required by CLaMP3):

```bash
python preprocessing/midi/batch_midi2mtf.py data/mid data/mtf --m3_compatible
```

⚠️ **Important:** The `--m3_compatible` flag is required for compatibility with CLaMP3's symbolic encoder.

**Inputs:** MIDI files in `data/mid/`  
**Outputs:** MTF-formatted files in `data/mtf/`

### **Step 4: Generate Training JSONL Files**

Create training and evaluation JSONL files from converted data:

```bash
python preprocessing/generate_training_jsonl.py \
  --metadata_dir data/processed \
  --data_dir data \
  --output_dir data/training \
  --verify_files
```

**Outputs:**
- `data/training/clamp3_train_abc.jsonl` - ABC training pairs (from PDMX)
- `data/training/clamp3_train_mtf.jsonl` - MTF training pairs (from PDMX)

### **Step 5: MidiCaps Dataset Split**

For MIDI-specific training, split the MidiCaps dataset into train/validation/test sets:

```bash
python preprocessing/split_midicaps.py \
  --input data/MidiCaps.jsonl \
  --output_dir data/midicaps_splits \
  --test_size 1000 \
  --val_size 1000
```

**Outputs:**
- `data/midicaps_splits/midicaps_train.jsonl` - 420,420 MIDI-text training pairs
- `data/midicaps_splits/midicaps_val.jsonl` - 1,000 validation pairs
- `data/midicaps_splits/midicaps_test.jsonl` - 1,000 test pairs

Convert evaluation data using same preprocessing scripts as above.

## **Quick Start**

### **Prerequisites Before Training**

1. **Verify data structure:**
```bash
ls data/training/  # Should contain clamp3_train_abc.jsonl and clamp3_train_mtf.jsonl
ls data/abc_standard/  # Should contain converted ABC files
ls data/mtf/  # Should contain converted MTF files
```

2. **Review configuration:**
```python
# Edit code/config.py to configure training:
LORA_R = 4                      # LoRA rank
LORA_ALPHA = 8                  # LoRA alpha scaling
LORA_NUM_EPOCHS = 5             # Epochs per adapter
LORA_BATCH_SIZE = 32            # Batch size per GPU
LORA_LEARNING_RATE = 2e-3       # Learning rate (2e-3 for MidiCaps)

# For PDMX training (ABC notation from sheet music):
LORA_ABC_TRAIN_JSONL = "data/training/clamp3_train_abc.jsonl"
TRAIN_ABC_ADAPTER = True        # Enable ABC adapter training

# For MidiCaps training (MIDI format):
LORA_MTF_TRAIN_JSONL = "data/midicaps_splits/midicaps_train.jsonl"
LORA_MTF_VAL_JSONL = "data/midicaps_splits/midicaps_val.jsonl"
TRAIN_ABC_ADAPTER = False       # Disable ABC, train only MTF
```

3. **Verify model weights:**
```bash
ls code/weights_clamp3_*.pth  # Should have pretrained CLaMP3 weights (C2 version)
```

### **1. Training LoRA Adapters**

Train separate LoRA adapters for ABC (PDMX) and MTF (MidiCaps) modalities:

```bash
# Single GPU training
python code/train_clamp3_lora.py

# Multi-GPU training (e.g., 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --use_env code/train_clamp3_lora.py
```

**Training Options:**

**Option A: PDMX Training (Sheet Music → ABC notation)**
- Set `TRAIN_ABC_ADAPTER = True` in `config.py`
- Uses PDMX dataset converted to ABC format
- Trains both ABC and MTF adapters

**Option B: MidiCaps Training (MIDI → MTF format)**
- Set `TRAIN_ABC_ADAPTER = False` in `config.py`
- Uses MidiCaps dataset (420K MIDI-text pairs)
- Trains only MTF adapter on large-scale MIDI data

**What the training script does:**
1. Loads pretrained CLaMP3 model weights (C2 version)
2. Applies LoRA to symbolic encoder (query, key, value attention layers)
3. Trains enabled adapters (ABC and/or MTF)
4. Saves best adapters to:
   - `code/adapters/lora_abc_adapter/` (if ABC enabled)
   - `code/adapters/lora_mtf_adapter/`

**Training outputs:**
- `code/logs/lora_training/ABC_training.log` - ABC training metrics
- `code/logs/lora_training/MTF_training.log` - MTF training metrics
- `code/logs/lora_training/ABC_history.json` - Per-epoch loss tracking
- `code/logs/lora_training/MTF_history.json` - Per-epoch loss tracking
- Periodic checkpoints saved (last 3 epochs kept)

**Resume training from checkpoint:**
```python
# In code/train_clamp3_lora.py, set:
RESUME_CHECKPOINT = "code/logs/lora_training/ABC_checkpoint_epoch5.pth"
RESUME_ADAPTER = "code/logs/lora_training/ABC_checkpoint_epoch5_adapter"
START_EPOCH = 6
```

### **2. Evaluation on Test Datasets**

**Evaluate both MidiCaps and WikiMT-X (default):**

```bash
python lora_eval/evaluate_adapters.py
```

**Evaluate only MidiCaps Test Set (1000 held-out samples):**

```bash
python lora_eval/evaluate_adapters.py --eval_midicaps
```

**Evaluate only WikiMT-X Test Set:**

```bash
# Default: uses 'analysis' field
python lora_eval/evaluate_adapters.py --eval_wikimt

# Evaluate specific text field (background, description, or scene)
python lora_eval/evaluate_adapters.py --eval_wikimt --wikimt_text_field background

# Evaluate ALL text fields
python lora_eval/evaluate_adapters.py --eval_wikimt --wikimt_text_field all
```

**Use custom adapter paths:**

```bash
python lora_eval/evaluate_adapters.py \
  --midicaps_adapter path/to/mtf_adapter \
  --wikimt_adapter path/to/abc_adapter
```

**Outputs:**
- Baseline vs LoRA comparison tables
- Text-to-Music and Music-to-Text retrieval metrics (MRR, Hit@1/5/10)
- Results saved to `lora_eval/evaluation_results.json`

## **Results**

### **MidiCaps (MTF Adapter)**

**Text-to-Music:**

| Metric | Baseline | LoRA | %Change |
|--------|----------|------|---------|
| MRR    | 0.4414   | 0.6504 | +47.34% |
| Hit@1  | 30.95    | 50.95 | +64.62% |
| Hit@5  | 57.86    | 85.00 | +46.91% |
| Hit@10 | 70.95    | 91.67 | +29.19% |

**Music-to-Text:**

| Metric | Baseline | LoRA | %Change |
|--------|----------|------|---------|
| MRR    | 0.4482   | 0.6285 | +40.21% |
| Hit@1  | 30.48    | 49.05 | +60.94% |
| Hit@5  | 61.67    | 80.24 | +30.12% |
| Hit@10 | 72.38    | 88.81 | +22.70% |

### **WikiMT-X (ABC Adapter) - Background Field**

**Text-to-Music:**

| Metric | Baseline | LoRA | %Change |
|--------|----------|------|---------|
| MRR    | 0.1534   | 0.1788 | +16.50% |
| Hit@1  | 9.10     | 11.90 | +30.77% |
| Hit@5  | 20.20    | 22.20 | +9.90% |
| Hit@10 | 26.80    | 29.90 | +11.57% |

**Music-to-Text:**

| Metric | Baseline | LoRA | %Change |
|--------|----------|------|---------|
| MRR    | 0.0429   | 0.0683 | +59.34% |
| Hit@1  | 1.50     | 2.60  | +73.33% |
| Hit@5  | 4.80     | 9.00  | +87.50% |
| Hit@10 | 8.80     | 15.00 | +70.45% |

### **Key Findings**

- **Parameter Efficiency**: 221K trainable parameters (0.05% of 458M total)
- **MTF Adapter**: Strong improvements on MidiCaps (+47% MRR, +65% Hit@1)
- **ABC Adapter**: Consistent gains on WikiMT-X across all text fields
- **No Degradation**: All metrics improved after LoRA fine-tuning

## **References**

For complete bibliography including all dependencies, see `REFERENCES.bib`