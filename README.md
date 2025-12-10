# **CLaMP 3 with LoRA: Specialized Symbolic Music Encoder**

## **Overview**

This project extends [**CLaMP 3**](https://sanderwood.github.io/clamp3/) with [**LoRA (Low-Rank Adaptation)**](https://github.com/microsoft/LoRA) on their symbolic music encoder to enable specialized learning for **sheet music (ABC notation)** and **MIDI** formats. By fine-tuning only adapter layers, we achieve efficient specialization to symbolic modalities while maintaining the pre-trained knowledge of the base model.

## **Project Objectives**

- Improve symbolic music (ABC & MIDI) retrieval performance over base CLaMP 3
- Demonstrate efficient fine-tuning via LoRA on the symbolic encoder
- Evaluate on specialized datasets (PDMX training, Lakh & WikiMT testing)  

## **What Can This Project Do?**

<!-- TODO: Fill in specific capabilities and applications -->

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
- `data/training/clamp3_train_abc.jsonl` - ABC training pairs
- `data/training/clamp3_train_mtf.jsonl` - MTF training pairs

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
# Edit code/config.py to verify:
LORA_R = 8                      # LoRA rank
LORA_ALPHA = 16                 # LoRA alpha scaling
LORA_NUM_EPOCHS = 10            # Epochs per adapter
LORA_BATCH_SIZE = 32            # Batch size per GPU
LORA_LEARNING_RATE = 1e-4       # Learning rate
LORA_ABC_TRAIN_JSONL = "data/training/clamp3_train_abc.jsonl"
LORA_MTF_TRAIN_JSONL = "data/training/clamp3_train_mtf.jsonl"
```

3. **Verify model weights:**
```bash
ls code/weights_clamp3_*.pth  # Should have pretrained CLaMP3 weights (C2 version)
```

### **1. Training LoRA Adapters**

Train separate LoRA adapters for ABC and MTF symbolic modalities using distributed training:

```bash
# Single GPU training
python -m torch.distributed.launch --nproc_per_node=1 --use_env code/train_clamp3_lora.py

# Multi-GPU training (e.g., 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --use_env code/train_clamp3_lora.py
```

**What the training script does:**
1. Loads pretrained CLaMP3 model weights (C2 version)
2. Applies LoRA configuration to symbolic encoder attention layers
3. Trains ABC adapter (10 epochs by default)
4. Trains MTF adapter (10 epochs by default)
5. Saves best adapters to:
   - `code/adapters/lora_abc_adapter/` 
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

Evaluate LoRA-adapted model against baseline using real datasets (WikiMT-X for ABC, Lakh MIDI for MTF):

```bash
# Prepare test datasets (100 ABC samples + 100 MTF samples)
python prepare_test_data.py --num_abc 100 --num_mtf 100

# Run complete evaluation pipeline
python test_pipeline.py
python evaluate_baseline.py
python evaluate_lora.py
python compare_results.py

# View results
cat evaluation_summary.txt
python -m json.tool < evaluation_comparison_report.json
```

**What's evaluated:**
- **ABC Dataset (WikiMT-X)**: Sheet music with rich text annotations
  - Text-to-music semantic similarity
  - Feature representation quality
  
- **MTF Dataset (Lakh MIDI)**: MIDI performances with minimal annotations
  - Cross-modal alignment
  - Embedding consistency

**Outputs generated:**
- `evaluation_summary.txt` - Human-readable results
- `evaluation_lora_report.json` - LoRA model metrics
- `evaluation_baseline_report.json` - Baseline model metrics
- `evaluation_comparison_report.json` - Side-by-side comparison

## **Results**

We evaluated LoRA-adapted CLaMP 3 against the original model on text-to-music alignment tasks using two datasets:

- **WikiMT-X** (CLaMP 3 authors' dataset) - ABC notation with rich text annotations
- **Lakh MIDI** (unseen during CLaMP 3 training) - MIDI performances with text descriptions

**Key Finding:** LoRA specialization significantly improved MIDI text alignment:

| Model | Lakh MIDI (Text-MIDI Similarity) |
|-------|-----------|
| **Base CLaMP 3** | -0.0082 |
| **CLaMP 3 + LoRA** | 0.0512 |
| **Improvement** | +0.0594 ↑ |

The LoRA adapters enable CLaMP 3 to achieve positive semantic alignment on the Lakh dataset, which was not used during the original model's training. This demonstrates that specialized adapters improve symbolic music understanding through efficient fine-tuning with minimal additional parameters.

## **References**

For complete bibliography including all dependencies, see `REFERENCES.bib`

### **Key References**
- **CLaMP 3 and WikiMT Benchmark**: [Wu et al., 2025](https://arxiv.org/abs/2502.10362)
- **PDMX Dataset**: [Long et al., 2025](https://arxiv.org/abs/2502.10362)
- **LoRA**: [Hu et al., 2022](https://openreview.net/forum?id=nZeVKeeFYf9)
- **Lakh MIDI Dataset**