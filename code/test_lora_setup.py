"""
Quick test script to verify LoRA training setup before full training.
"""

import os
import sys
import torch
from peft import LoraConfig
from transformers import AutoTokenizer, BertConfig

# Add code directory to path
sys.path.append(os.path.dirname(__file__))

from utils import CLaMP3Model, CLaMP3ModelWithLoRA, M3Patchilizer
from config import *

print("="*80)
print("Testing LoRA Training Setup")
print("="*80)

# Test 1: CUDA availability
print("\n1. CUDA Test:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("   ❌ ERROR: CUDA not available!")
    sys.exit(1)

# Test 2: Configuration files
print("\n2. Configuration Test:")
print(f"   LoRA R: {LORA_R}")
print(f"   LoRA Alpha: {LORA_ALPHA}")
print(f"   LoRA Dropout: {LORA_DROPOUT}")
print(f"   Target Modules: {LORA_TARGET_MODULES}")
print(f"   Learning Rate: {LORA_LEARNING_RATE}")
print(f"   Batch Size: {LORA_BATCH_SIZE}")
print(f"   Epochs: {LORA_NUM_EPOCHS}")

# Test 3: Training data files
print("\n3. Training Data Test:")
abc_exists = os.path.exists(LORA_ABC_TRAIN_JSONL)
mtf_exists = os.path.exists(LORA_MTF_TRAIN_JSONL)
print(f"   ABC JSONL: {'✓' if abc_exists else '❌'} {LORA_ABC_TRAIN_JSONL}")
print(f"   MTF JSONL: {'✓' if mtf_exists else '❌'} {LORA_MTF_TRAIN_JSONL}")

if not abc_exists or not mtf_exists:
    print("   ❌ ERROR: Training data files not found!")
    sys.exit(1)

# Count entries
import json
with open(LORA_ABC_TRAIN_JSONL, 'r', encoding='utf-8') as f:
    abc_count = sum(1 for _ in f)
with open(LORA_MTF_TRAIN_JSONL, 'r', encoding='utf-8') as f:
    mtf_count = sum(1 for _ in f)
print(f"   ABC Entries: {abc_count:,}")
print(f"   MTF Entries: {mtf_count:,}")

# Test 4: Pretrained weights
print("\n4. Pretrained Weights Test:")
weights_exist = os.path.exists(CLAMP3_WEIGHTS_PATH)
print(f"   Weights File: {'✓' if weights_exist else '❌'} {CLAMP3_WEIGHTS_PATH}")

if weights_exist:
    try:
        checkpoint = torch.load(CLAMP3_WEIGHTS_PATH, map_location='cpu', weights_only=True)
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Model Keys: {len(checkpoint['model'].keys())}")
        print("   ✓ Weights loaded successfully")
    except Exception as e:
        print(f"   ❌ ERROR loading weights: {e}")
        sys.exit(1)
else:
    print("   ❌ ERROR: Pretrained weights not found!")
    sys.exit(1)

# Test 5: Model initialization
print("\n5. Model Initialization Test:")
device = torch.device("cuda:0")

try:
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
    
    base_model = CLaMP3Model(audio_config=audio_config,
                            symbolic_config=symbolic_config,
                            global_rank=0,
                            world_size=1,
                            text_model_name=TEXT_MODEL_NAME,
                            hidden_size=CLAMP3_HIDDEN_SIZE,
                            load_m3=CLAMP3_LOAD_M3)
    
    checkpoint = torch.load(CLAMP3_WEIGHTS_PATH, map_location='cpu', weights_only=True)
    base_model.load_state_dict(checkpoint['model'])
    print("   ✓ Base model loaded")
    
except Exception as e:
    print(f"   ❌ ERROR initializing base model: {e}")
    sys.exit(1)

# Test 6: LoRA wrapper
print("\n6. LoRA Wrapper Test:")
try:
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    model = CLaMP3ModelWithLoRA(base_model, lora_config)
    model = model.to(device)
    
    freeze_list = ["text_model", "text_proj", "audio_model", "audio_proj", "symbolic_model", "symbolic_proj"]
    model.set_trainable(freeze_list)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Percentage Trainable: {100 * trainable_params / total_params:.2f}%")
    print("   ✓ LoRA wrapper initialized")
    
except Exception as e:
    print(f"   ❌ ERROR initializing LoRA: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Tokenizer and Patchilizer
print("\n7. Tokenizer/Patchilizer Test:")
try:
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    print(f"   ✓ Tokenizer loaded: {TEXT_MODEL_NAME}")
    
    patchilizer = M3Patchilizer()
    print(f"   ✓ Patchilizer initialized")
    
except Exception as e:
    print(f"   ❌ ERROR loading tokenizer/patchilizer: {e}")
    sys.exit(1)

# Test 8: Forward pass test
print("\n8. Forward Pass Test:")
try:
    # Create dummy inputs
    batch_size = 2
    text_inputs = torch.randint(0, 1000, (batch_size, MAX_TEXT_LENGTH)).to(device)
    text_masks = torch.ones(batch_size, MAX_TEXT_LENGTH).to(device)
    music_inputs = torch.randint(0, 100, (batch_size, PATCH_LENGTH, PATCH_SIZE)).to(device)
    music_masks = torch.ones(batch_size, PATCH_LENGTH).to(device)
    
    model.eval()
    with torch.no_grad():
        loss = model(text_inputs, text_masks, music_inputs, music_masks, "symbolic")
    
    print(f"   Loss shape: {loss.shape}")
    print(f"   Loss value: {loss.item():.4f}")
    print("   ✓ Forward pass successful")
    
except Exception as e:
    print(f"   ❌ ERROR in forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Adapter save/load test
print("\n9. Adapter Save/Load Test:")
try:
    test_adapter_path = os.path.join(LORA_LOGS_DIR, "test_adapter")
    os.makedirs(os.path.dirname(test_adapter_path), exist_ok=True)
    
    model.save_adapter(test_adapter_path)
    print(f"   ✓ Adapter saved to: {test_adapter_path}")
    
    model.load_adapter(test_adapter_path)
    print(f"   ✓ Adapter loaded from: {test_adapter_path}")
    
    # Clean up test adapter
    import shutil
    if os.path.exists(test_adapter_path):
        shutil.rmtree(test_adapter_path)
        print(f"   ✓ Test adapter cleaned up")
    
except Exception as e:
    print(f"   ❌ ERROR saving/loading adapter: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: GPU Memory Test
print("\n10. GPU Memory Test:")
print(f"   Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"   Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"   Max Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Calculate estimated memory for batch
estimated_batch_memory = (total_params * 4 / 1e9) * 2  # Model + gradients
print(f"   Estimated per-batch: ~{estimated_batch_memory:.2f} GB")

total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
available = total_memory - torch.cuda.memory_allocated()/1e9
print(f"   Available: {available:.2f} GB")

if available < estimated_batch_memory:
    print(f"   ⚠ WARNING: May need to reduce batch size!")
else:
    print(f"   ✓ Sufficient memory available")

print("\n" + "="*80)
print("✓ ALL TESTS PASSED - Ready for Training!")
print("="*80)
print("\nYou can now run:")
print("  cd code")
print("  python train_clamp3_lora.py")
print("="*80 + "\n")
