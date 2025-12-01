"""
Train LoRA adapters for CLaMP3 symbolic encoder.
Trains separate adapters for ABC and MTF formats using pretrained weights.
"""

import os
import json
import time
import torch
import random
import numpy as np
from utils import *
from config import *
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, BertConfig, get_constant_schedule_with_warmup


def list_files_in_json(json_path):
    """Load JSONL training data."""
    file_list = []
    
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                file_list.append(item)

    return file_list


def collate_batch(batch):
    """Collate batch for DataLoader."""
    text_inputs, text_masks, music_inputs, music_masks = zip(*batch)

    text_inputs = torch.stack(text_inputs)
    text_masks = torch.stack(text_masks)
    music_inputs = torch.stack(music_inputs)
    music_masks = torch.stack(music_masks)

    return text_inputs, text_masks, music_inputs, music_masks


class TextMusicDataset(Dataset):
    """Dataset for LoRA training."""
    def __init__(self, items, mode, file_format):
        print(f"The number of {mode} data: {len(items)}")
        self.items = items
        self.mode = mode
        self.file_format = file_format  # 'abc' or 'mtf'
        self.datapath = os.path.dirname(LORA_ABC_TRAIN_JSONL if file_format == 'abc' else LORA_MTF_TRAIN_JSONL)
        self.datapath = os.path.abspath(self.datapath)

    def text_dropout(self, item):
        """Apply text dropout augmentation during training."""
        candidates = []
        if random.random() < 0.5:
            translations = item.get("translations", {})
            for key in translations.keys():
                if key != "language":
                    candidates.append(translations[key])
        candidates = [c for c in candidates if c is not None and len(c) > 0]
        
        if len(candidates) == 0:
            # Use analysis field
            if "analysis" in item and item["analysis"]:
                candidates = [item["analysis"]]

        candidates = [c for c in candidates if c is not None and len(c) > 0]
        if len(candidates) == 0:
            candidates = ["music"]  # Fallback
        
        candidates = list(set(candidates))
        candidates = "\\n".join(candidates).split("\\n")
        selected_candidates = [c for c in candidates if len(c) > 0 and random.random() < 0.5]
        if len(selected_candidates) == 0:
            selected_candidates = candidates
        random.shuffle(selected_candidates)
        text = tokenizer.sep_token.join(selected_candidates)

        return text

    def random_truncate(self, input_tensor, max_length):
        """Randomly truncate tensor to max_length."""
        if input_tensor.size(0) <= max_length:
            return input_tensor
            
        choices = ["head", "tail", "middle"]
        choice = random.choice(choices) if self.mode == 'train' else "head"
        
        if choice == "head":
            input_tensor = input_tensor[:max_length]
        elif choice == "tail":
            input_tensor = input_tensor[-max_length:]
        elif choice == "middle":
            start = random.randint(1, input_tensor.size(0) - max_length)
            input_tensor = input_tensor[start:start+max_length]
        
        return input_tensor
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]

        # Get text
        if self.mode == 'train' and TEXT_DROPOUT:
            text = self.text_dropout(item)
        else:
            text = item.get("analysis", "music")

        # Tokenize text
        text_inputs = tokenizer(text, return_tensors='pt')
        text_inputs = text_inputs['input_ids'].squeeze(0)
        if text_inputs.size(0) > MAX_TEXT_LENGTH:
            text_inputs = self.random_truncate(text_inputs, MAX_TEXT_LENGTH)
        text_masks = torch.ones(text_inputs.size(0))

        # Load music file
        if self.mode == 'train':
            filepath = random.choice(item["filepaths"])
        else:
            filepath = item["filepaths"][0]
        
        # Construct full path
        filepath = os.path.join(self.datapath, filepath)
        filepath = os.path.abspath(filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                music_content = f.read()
        except:
            # Fallback to empty content if file not found
            music_content = ""

        # Remove instrument info
        if random.random() < 0.9 and self.mode == 'train':
            music_content = remove_instrument_info(music_content)

        # Patchilize music
        music_inputs = patchilizer.encode(music_content, add_special_patches=True, truncate=True, random_truncate=(self.mode=="train"))
        music_inputs = torch.tensor(music_inputs)
        
        # Handle empty or 1D music_inputs (reshape to 2D if needed)
        if music_inputs.dim() == 0:
            # Scalar tensor - make it a 2D tensor with one empty patch
            music_inputs = torch.ones((1, PATCH_SIZE)).long() * patchilizer.pad_token_id
        elif music_inputs.dim() == 1:
            # 1D tensor - unsqueeze to make it 2D
            if music_inputs.size(0) == 0:
                music_inputs = torch.ones((1, PATCH_SIZE)).long() * patchilizer.pad_token_id
            else:
                # BOS/EOS tokens only - reshape
                music_inputs = music_inputs.unsqueeze(0).repeat(1, PATCH_SIZE // music_inputs.size(0) + 1)[:, :PATCH_SIZE]
        
        music_masks = torch.ones(music_inputs.size(0))
        
        # Pad text
        pad_indices = torch.ones(MAX_TEXT_LENGTH - text_inputs.size(0)).long() * tokenizer.pad_token_id
        text_inputs = torch.cat((text_inputs, pad_indices), 0)
        text_masks = torch.cat((text_masks, torch.zeros(MAX_TEXT_LENGTH - text_masks.size(0))), 0)

        # Pad music
        pad_indices = torch.ones((PATCH_LENGTH - music_inputs.size(0), PATCH_SIZE)).long() * patchilizer.pad_token_id
        music_inputs = torch.cat((music_inputs, pad_indices), 0)
        music_masks = torch.cat((music_masks, torch.zeros(PATCH_LENGTH - music_masks.size(0))), 0)
        
        return text_inputs, text_masks, music_inputs, music_masks


def process_one_batch(batch):
    """Process one batch through model."""
    text_inputs, text_masks, music_inputs, music_masks = batch
    
    # Move batch to GPU
    text_inputs = text_inputs.to(device)
    text_masks = text_masks.to(device)
    music_inputs = music_inputs.to(device)
    music_masks = music_masks.to(device)
    
    loss = model(text_inputs, text_masks, music_inputs, music_masks, "symbolic")

    # Reduce loss across GPUs if distributed
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss.mean()


def train_epoch(epoch, adapter_name):
    """Train for one epoch."""
    tqdm_train_set = tqdm(train_set, desc=f"[{adapter_name}] Epoch {epoch} Training")
    total_train_loss = 0
    iter_idx = 1
    model.train()
    train_steps = (epoch-1)*len(train_set)
    checkpoint_interval = max(1, len(train_set) // 10)  # Save 10 times per epoch

    for batch_idx, batch in enumerate(tqdm_train_set):
        try:
            with autocast(device_type='cuda'):
                loss = process_one_batch(batch)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: Invalid loss detected at batch {batch_idx}, skipping...")
                continue
                
            scaler.scale(loss).backward()
            total_train_loss += loss.item()
            scaler.step(optimizer)
            scaler.update()
            
            lr_scheduler.step()
            model.zero_grad(set_to_none=True)
            
            avg_loss = total_train_loss / iter_idx
            tqdm_train_set.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            train_steps += 1
            iter_idx += 1
            
            # Periodic checkpoint within epoch
            if batch_idx > 0 and batch_idx % checkpoint_interval == 0:
                checkpoint_path = os.path.join(LORA_LOGS_DIR, f'{adapter_name}_checkpoint_epoch{epoch}_batch{batch_idx}.pth')
                save_checkpoint(checkpoint_path, epoch, avg_loss, adapter_name)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM Error at batch {batch_idx}! Clearing cache and skipping batch...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        
    return total_train_loss / (iter_idx-1)


def eval_epoch(adapter_name):
    """Evaluate for one epoch."""
    tqdm_eval_set = tqdm(eval_set, desc=f"[{adapter_name}] Evaluating")
    total_eval_loss = 0
    iter_idx = 1
    model.eval()
  
    for batch in tqdm_eval_set:
        try:
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    loss = process_one_batch(batch)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_eval_loss += loss.item()
            
            avg_loss = total_eval_loss / iter_idx
            tqdm_eval_set.set_postfix({'eval_loss': f'{avg_loss:.4f}'})
            iter_idx += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM Error during eval! Clearing cache...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return total_eval_loss / (iter_idx-1)


def save_checkpoint(checkpoint_path, epoch, loss, adapter_name):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'adapter_name': adapter_name,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': lr_scheduler.state_dict(),
        'scaler_state': scaler.state_dict()
    }
    
    # Save LoRA adapter
    if world_size > 1:
        model.module.save_adapter(checkpoint_path.replace('.pth', '_adapter'))
    else:
        model.save_adapter(checkpoint_path.replace('.pth', '_adapter'))
    
    # Save optimizer states
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")


def train_adapter(adapter_name, train_jsonl_path, adapter_save_path):
    """Train a single LoRA adapter."""
    global train_set, eval_set, optimizer, lr_scheduler, scaler
    
    print("\n" + "="*80)
    print(f"Training {adapter_name} LoRA Adapter")
    print("="*80)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load training data
    print(f"Loading data from {train_jsonl_path}")
    train_files = list_files_in_json(train_jsonl_path)
    
    if not train_files:
        print(f"ERROR: No training data found in {train_jsonl_path}")
        return False
    
    # Split into train/eval
    train_files, eval_files = split_data(train_files, LORA_EVAL_SPLIT)
    
    train_batch_nums = int(len(train_files) / LORA_BATCH_SIZE)
    eval_batch_nums = int(len(eval_files) / LORA_BATCH_SIZE)

    train_files = train_files[:train_batch_nums*LORA_BATCH_SIZE]
    eval_files = eval_files[:eval_batch_nums*LORA_BATCH_SIZE]

    # Create datasets
    file_format = 'abc' if 'abc' in adapter_name.lower() else 'mtf'
    train_dataset = TextMusicDataset(train_files, 'train', file_format)
    eval_dataset = TextMusicDataset(eval_files, 'eval', file_format)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=global_rank)
    
    train_set = DataLoader(train_dataset, batch_size=LORA_BATCH_SIZE, collate_fn=collate_batch, sampler=train_sampler, shuffle=(train_sampler is None))
    eval_set = DataLoader(eval_dataset, batch_size=LORA_BATCH_SIZE, collate_fn=collate_batch, sampler=eval_sampler, shuffle=(train_sampler is None))

    # Reset optimizer and scheduler for this adapter
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LORA_LEARNING_RATE)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000)
    scaler = GradScaler()

    # Training loop
    best_epoch = 0
    min_eval_loss = float('inf')
    training_history = []
    
    print(f"\nStarting training for {LORA_NUM_EPOCHS} epochs...")
    print(f"Train batches: {len(train_set)}, Eval batches: {len(eval_set)}")
    print(f"Batch size: {LORA_BATCH_SIZE}, Learning rate: {LORA_LEARNING_RATE}")
    print(f"Saving adapters to: {adapter_save_path}\n")
    
    try:
        for epoch in range(1, LORA_NUM_EPOCHS + 1):
            train_sampler.set_epoch(epoch)
            eval_sampler.set_epoch(epoch)
            
            print('\n' + '=' * 80)
            print(f"[{adapter_name}] Epoch {epoch}/{LORA_NUM_EPOCHS}")
            print('=' * 80)
            
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = train_epoch(epoch, adapter_name)
            
            # Evaluation phase
            eval_loss = eval_epoch(adapter_name)
            
            epoch_time = time.time() - epoch_start_time
            
            # Record history
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'time': epoch_time,
                'timestamp': time.asctime()
            }
            training_history.append(history_entry)
            
            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"[{adapter_name}] Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Eval Loss:  {eval_loss:.4f}")
            print(f"  Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
            print(f"  Best Eval Loss: {min_eval_loss:.4f} (Epoch {best_epoch})")
            print(f"{'='*80}\n")
            
            if global_rank == 0:
                # Log to file
                log_path = os.path.join(LORA_LOGS_DIR, f'{adapter_name}_training.log')
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a') as f:
                    f.write(f"Epoch {epoch}/{LORA_NUM_EPOCHS}\n")
                    f.write(f"Train Loss: {train_loss:.6f}\n")
                    f.write(f"Eval Loss: {eval_loss:.6f}\n")
                    f.write(f"Time: {epoch_time:.1f}s\n")
                    f.write(f"Timestamp: {time.asctime()}\n")
                    f.write(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")
                    f.write(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB\n")
                    f.write("\n")
                
                # Save best model
                if eval_loss < min_eval_loss:
                    best_epoch = epoch
                    min_eval_loss = eval_loss
                    
                    # Save adapter
                    if world_size > 1:
                        model.module.save_adapter(adapter_save_path)
                    else:
                        model.save_adapter(adapter_save_path)
                    
                    print(f"✓ Saved BEST adapter at epoch {epoch} with eval_loss {eval_loss:.4f}\n")
                
                # Save epoch checkpoint (last 3 epochs)
                checkpoint_path = os.path.join(LORA_LOGS_DIR, f'{adapter_name}_checkpoint_epoch{epoch}.pth')
                save_checkpoint(checkpoint_path, epoch, eval_loss, adapter_name)
                
                # Clean up old checkpoints (keep last 3)
                if epoch > 3:
                    old_checkpoint = os.path.join(LORA_LOGS_DIR, f'{adapter_name}_checkpoint_epoch{epoch-3}.pth')
                    old_adapter = old_checkpoint.replace('.pth', '_adapter')
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                    if os.path.exists(old_adapter):
                        import shutil
                        shutil.rmtree(old_adapter, ignore_errors=True)
                
                # Save training history
                history_path = os.path.join(LORA_LOGS_DIR, f'{adapter_name}_history.json')
                with open(history_path, 'w') as f:
                    json.dump(training_history, f, indent=2)

            if world_size > 1:
                dist.barrier()
            
            # Clear CUDA cache between epochs
            torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\n" + "!"*80)
        print("Training interrupted by user!")
        print("Saving emergency checkpoint...")
        if global_rank == 0:
            emergency_path = os.path.join(LORA_LOGS_DIR, f'{adapter_name}_emergency_checkpoint.pth')
            # Use 0 for epoch and 999.9 for loss if not yet defined
            save_checkpoint(emergency_path, locals().get('epoch', 0), locals().get('eval_loss', 999.9), adapter_name)
            print(f"Emergency checkpoint saved to: {emergency_path}")
        print("!"*80)
        raise
    
    except Exception as e:
        print("\n" + "!"*80)
        print(f"Training failed with error: {e}")
        print("Saving emergency checkpoint...")
        if global_rank == 0:
            emergency_path = os.path.join(LORA_LOGS_DIR, f'{adapter_name}_emergency_checkpoint.pth')
            try:
                save_checkpoint(emergency_path, locals().get('epoch', 0), 999.9, adapter_name)
                print(f"Emergency checkpoint saved to: {emergency_path}")
            except:
                print("Failed to save emergency checkpoint")
        print("!"*80)
        raise

    if global_rank == 0:
        print(f"\n" + "="*80)
        print(f"✓ {adapter_name} Adapter Training Complete!")
        print(f"  Best Epoch: {best_epoch}/{LORA_NUM_EPOCHS}")
        print(f"  Best Eval Loss: {min_eval_loss:.4f}")
        print(f"  Final Adapter: {adapter_save_path}")
        print(f"  Training History: {history_path}")
        print("="*80)
    
    return True


if __name__ == "__main__":
    # Set up distributed training
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend='nccl')
    else:
        # Force CUDA for single GPU training
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! This script requires a GPU.")
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    
    print("\n" + "="*80)
    print("CLaMP3 LoRA Adapter Training")
    print("="*80)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"World Size: {world_size}")
    print(f"Global Rank: {global_rank}")
    print("="*80 + "\n")
        
    if CLAMP3_DETERMINISTIC:
        seed = 42 + global_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize configurations
    audio_config = BertConfig(vocab_size=1,
                            hidden_size=AUDIO_HIDDEN_SIZE,
                            num_hidden_layers=AUDIO_NUM_LAYERS,
                            num_attention_heads=AUDIO_HIDDEN_SIZE//64,
                            intermediate_size=AUDIO_HIDDEN_SIZE*4,
                            max_position_embeddings=MAX_AUDIO_LENGTH)
    symbolic_config = BertConfig(vocab_size=1,
                                hidden_size=M3_HIDDEN_SIZE,
                                num_hidden_layers=PATCH_NUM_LAYERS,
                                num_attention_heads=M3_HIDDEN_SIZE//64,
                                intermediate_size=M3_HIDDEN_SIZE*4,
                                max_position_embeddings=PATCH_LENGTH)
    
    # Load base CLaMP3 model with pretrained weights
    print("Loading pretrained CLaMP3 model...")
    base_model = CLaMP3Model(audio_config=audio_config,
                            symbolic_config=symbolic_config,
                            global_rank=global_rank,
                            world_size=world_size,
                            text_model_name=TEXT_MODEL_NAME,
                            hidden_size=CLAMP3_HIDDEN_SIZE,
                            load_m3=CLAMP3_LOAD_M3)
    
    # Load pretrained CLaMP3 weights
    if os.path.exists(CLAMP3_WEIGHTS_PATH):
        checkpoint = torch.load(CLAMP3_WEIGHTS_PATH, map_location='cpu', weights_only=True)
        base_model.load_state_dict(checkpoint['model'])
        print(f"Loaded pretrained CLaMP3 weights from epoch {checkpoint['epoch']}")
    else:
        print(f"Warning: Pretrained weights not found at {CLAMP3_WEIGHTS_PATH}")
        print("Continuing with M3-initialized weights only")
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    # Wrap model with LoRA
    print("Applying LoRA to symbolic encoder...")
    model = CLaMP3ModelWithLoRA(base_model, lora_config)
    model = model.to(device)
    
    # Initialize tokenizer and patchilizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    patchilizer = M3Patchilizer()

    # Freeze base model, enable LoRA adapters
    freeze_list = ["text_model", "text_proj", "audio_model", "audio_proj", "symbolic_model", "symbolic_proj"]
    model.set_trainable(freeze_list)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (LoRA): {trainable_params:,}")
    print(f"Percentage Trainable: {100 * trainable_params / total_params:.2f}%")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Train ABC adapter
    print("\n" + "="*80)
    print("STAGE 1/2: Training ABC Adapter")
    print("="*80)
    stage1_start = time.time()
    
    try:
        success = train_adapter("ABC", LORA_ABC_TRAIN_JSONL, LORA_ABC_ADAPTER_PATH)
        stage1_time = time.time() - stage1_start
        
        if not success:
            print("\n⚠ ABC adapter training failed!")
        else:
            print(f"\n✓ ABC adapter training completed successfully in {stage1_time/60:.1f} minutes!")
    except Exception as e:
        print(f"\n✗ ABC adapter training crashed: {e}")
        print("Check emergency checkpoint in logs directory")
        raise
    
    # Clear memory before Stage 2
    torch.cuda.empty_cache()
    
    # Reload base model for MTF adapter training
    print("\nReloading base model for MTF adapter...")
    base_model = CLaMP3Model(audio_config=audio_config,
                            symbolic_config=symbolic_config,
                            global_rank=global_rank,
                            world_size=world_size,
                            text_model_name=TEXT_MODEL_NAME,
                            hidden_size=CLAMP3_HIDDEN_SIZE,
                            load_m3=CLAMP3_LOAD_M3)
    
    if os.path.exists(CLAMP3_WEIGHTS_PATH):
        checkpoint = torch.load(CLAMP3_WEIGHTS_PATH, map_location='cpu', weights_only=True)
        base_model.load_state_dict(checkpoint['model'])
    
    model = CLaMP3ModelWithLoRA(base_model, lora_config)
    model = model.to(device)
    model.set_trainable(freeze_list)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Train MTF adapter
    print("\n" + "="*80)
    print("STAGE 2/2: Training MTF Adapter")
    print("="*80)
    stage2_start = time.time()
    
    try:
        success = train_adapter("MTF", LORA_MTF_TRAIN_JSONL, LORA_MTF_ADAPTER_PATH)
        stage2_time = time.time() - stage2_start
        
        if not success:
            print("\n⚠ MTF adapter training failed!")
        else:
            print(f"\n✓ MTF adapter training completed successfully in {stage2_time/60:.1f} minutes!")
    except Exception as e:
        print(f"\n✗ MTF adapter training crashed: {e}")
        print("Check emergency checkpoint in logs directory")
        raise
    
    total_time = time.time() - stage1_start
    
    print("\n" + "="*80)
    print("✓ LoRA ADAPTER TRAINING COMPLETE")
    print("="*80)
    print(f"ABC Adapter: {LORA_ABC_ADAPTER_PATH}")
    print(f"MTF Adapter: {LORA_MTF_ADAPTER_PATH}")
    print(f"Total Training Time: {total_time/3600:.2f} hours")
    print(f"Logs Directory: {LORA_LOGS_DIR}")
    print("="*80 + "\n")
