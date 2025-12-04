# LoRA Training - Production Features

## Key Improvements for Robust Training

### 1. **CUDA-Only Execution**
- Forces GPU usage - raises error if CUDA not available
- Explicit device management: `torch.device("cuda:0")`
- Clear GPU information displayed at startup
- Prevents accidental CPU training

### 2. **Comprehensive Progress Tracking**
- Named progress bars with adapter name and epoch: `[ABC] Epoch 5 Training`
- Real-time metrics: loss, learning rate displayed in progress bar
- Epoch summaries showing:
  - Train loss, Eval loss
  - Time per epoch (seconds and minutes)
  - Best eval loss so far
  - GPU memory usage
- Overall timing: stage timing and total training time in hours

### 3. **Robust Checkpoint System**
- **Best model checkpoints**: Saved whenever eval loss improves
- **Epoch checkpoints**: Last 3 epochs kept, older ones auto-deleted
- **Emergency checkpoints**: Saved on Ctrl+C or unexpected errors
- **Periodic checkpoints**: Saved 10 times per epoch during training
- **Checkpoint contents**:
  - LoRA adapter weights
  - Optimizer state (for resuming)
  - Scheduler state
  - Scaler state
  - Epoch and loss information

### 4. **Error Handling & Recovery**

#### Out-of-Memory (OOM) Protection
```python
except RuntimeError as e:
    if "out of memory" in str(e):
        print("\nOOM Error! Clearing cache and skipping batch...")
        torch.cuda.empty_cache()
        continue
```
- Catches CUDA OOM errors
- Clears GPU cache
- Skips problematic batch
- Continues training (doesn't crash)

#### Keyboard Interrupt (Ctrl+C) Handling
```python
except KeyboardInterrupt:
    print("Training interrupted by user!")
    print("Saving emergency checkpoint...")
    save_checkpoint(emergency_path, epoch, eval_loss, adapter_name)
    raise
```
- Saves current progress before exiting
- Creates emergency checkpoint with all states
- Allows clean interruption without losing progress

#### General Exception Handling
```python
except Exception as e:
    print(f"Training failed with error: {e}")
    save_checkpoint(emergency_path, epoch, 999.9, adapter_name)
    raise
```
- Catches unexpected errors
- Attempts to save emergency checkpoint
- Re-raises exception for debugging

#### Invalid Loss Detection
```python
if torch.isnan(loss) or torch.isinf(loss):
    print(f"\nWarning: Invalid loss at batch {batch_idx}, skipping...")
    continue
```
- Detects NaN or Inf losses
- Skips problematic batches
- Prevents training divergence

### 5. **Memory Management**
- CUDA cache cleared between epochs: `torch.cuda.empty_cache()`
- Memory stats logged to training log file
- Memory cleared between ABC and MTF training stages
- Old checkpoints deleted to save disk space

### 6. **Detailed Logging**

#### Training Log File (`{adapter_name}_training.log`)
```
Epoch 5/10
Train Loss: 0.234567
Eval Loss: 0.345678
Time: 123.4s
Timestamp: Sat Nov 30 12:34:56 2025
GPU Memory Allocated: 4.52 GB
GPU Memory Reserved: 5.12 GB
```

#### Training History JSON (`{adapter_name}_history.json`)
```json
[
  {
    "epoch": 1,
    "train_loss": 0.5432,
    "eval_loss": 0.4321,
    "time": 145.2,
    "timestamp": "Sat Nov 30 12:00:00 2025"
  },
  ...
]
```

### 7. **Clear Visual Feedback**
- Startup banner with system info
- Stage indicators: `STAGE 1/2: Training ABC Adapter`
- Success indicators: ✓ for success, ⚠ for warnings, ✗ for errors
- Colored/formatted epoch separators
- Final summary with total time and file locations

### 8. **Distributed Training Support**
- Single GPU optimized (your setup)
- Multi-GPU ready (just set WORLD_SIZE environment variable)
- Proper DDP wrapping for distributed scenarios
- Barrier synchronization for multi-GPU

## File Outputs

### Adapter Files
- `code/adapters/lora_abc_adapter/` - Best ABC adapter
- `code/adapters/lora_mtf_adapter/` - Best MTF adapter

### Log Files (in `code/logs/lora/`)
- `ABC_training.log` - ABC training log
- `MTF_training.log` - MTF training log
- `ABC_history.json` - ABC training metrics history
- `MTF_history.json` - MTF training metrics history
- `ABC_checkpoint_epoch{N}.pth` - Last 3 epoch checkpoints
- `MTF_checkpoint_epoch{N}.pth` - Last 3 epoch checkpoints
- `ABC_emergency_checkpoint.pth` - Emergency backup (if interrupted)
- `MTF_emergency_checkpoint.pth` - Emergency backup (if interrupted)

## Safety Features Summary

✓ **No progress loss**: Emergency checkpoints on any failure
✓ **GPU memory protection**: OOM handling without crash
✓ **Clean interruption**: Ctrl+C saves and exits gracefully
✓ **Invalid loss handling**: NaN/Inf detection and skip
✓ **Checkpoint rotation**: Auto-cleanup of old checkpoints
✓ **Complete state saving**: Can resume from any checkpoint
✓ **Detailed logging**: Full training history preserved
✓ **Clear progress tracking**: Always know where you are

## Usage

### Run Setup Test (Recommended First)
```bash
cd code
python test_lora_setup.py
```

### Start Training
```bash
cd code
python train_clamp3_lora.py
```

### Monitor Progress
- Watch terminal output for real-time progress
- Check `code/logs/lora/ABC_training.log` for ABC training
- Check `code/logs/lora/MTF_training.log` for MTF training

### Interrupt Training
- Press `Ctrl+C` - saves emergency checkpoint automatically
- Training can be resumed from checkpoints

### After Training
- Best adapters automatically saved in `code/adapters/`
- Training history in JSON format for analysis
- Load adapters using `model.load_adapter(path)`
