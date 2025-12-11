import os

EVAL_SPLIT = 0.01  # Fraction of training data used for evaluation
WANDB_KEY = "<YOUR_WANDB_KEY>"  # Weights and Biases API key

# -------------------- Configuration for M3 Training --------------------
M3_TRAIN_FOLDERS = [
    "<YOUR_TRAINING_DATA_FOLDER>"  # Directory containing training data for M3
]

M3_EVAL_FOLDERS = [
    "<YOUR_EVALUATION_DATA_FOLDER>"  # Directory containing evaluation data for M3 (optional)
]

PATCH_SIZE = 64  # Size of each patch
PATCH_LENGTH = 512  # Length of the patches
PATCH_NUM_LAYERS = 12  # Number of layers in the encoder
TOKEN_NUM_LAYERS = 3  # Number of layers in the decoder
M3_HIDDEN_SIZE = 768  # Size of the hidden layer

M3_NUM_EPOCH = 100  # Maximum number of epochs for training
M3_LEARNING_RATE = 1e-4  # Learning rate for the optimizer
M3_BATCH_SIZE = 16  # Batch size per GPU (single card) during training
M3_MASK_RATIO = 0.45  # Ratio of masked elements during training
M3_DETERMINISTIC = True  # Ensures deterministic results with random seeds
M3_WANDB_LOG = True  # Enable logging to Weights and Biases
M3_LOAD_CKPT = True  # Load model weights from a checkpoint if available

M3_WEIGHTS_PATH = (
    "weights_m3"+
    "_h_size_" + str(M3_HIDDEN_SIZE) +
    "_t_layers_" + str(TOKEN_NUM_LAYERS) +
    "_p_layers_" + str(PATCH_NUM_LAYERS) +
    "_p_size_" + str(PATCH_SIZE) +
    "_p_length_" + str(PATCH_LENGTH) +
    "_lr_" + str(M3_LEARNING_RATE) +
    "_batch_" + str(M3_BATCH_SIZE) +
    "_mask_" + str(M3_MASK_RATIO) + ".pth"
)  # Path to store the model weights
M3_LOGS_PATH = M3_WEIGHTS_PATH.replace("weights", "logs").replace("pth", "txt")  # Path to save training logs

# -------------------- Configuration for CLaMP3 Training ----------------
CLAMP3_TRAIN_JSONL = "<YOUR_TRAINING_JSONL_FILE>"  # Path to the JSONL file with training data for CLaMP3
CLAMP3_EVAL_JSONL = "<YOUR_EVALUATION_JSONL_FILE>"  # Path to the JSONL file with evaluation data for CLaMP3 (optional)

CLAMP3_HIDDEN_SIZE = 768  # Size of the hidden layer
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"  # Name of the pre-trained text model
MAX_TEXT_LENGTH = 128  # Maximum allowed length for text input

AUDIO_HIDDEN_SIZE = 768  # Size of the hidden layer for audio features
AUDIO_NUM_LAYERS = 12  # Number of layers in the audio encoder
MAX_AUDIO_LENGTH = 128  # Maximum allowed length for audio input

CLAMP3_NUM_EPOCH = 100  # Maximum number of epochs for training
CLAMP3_LEARNING_RATE = 1e-5  # Learning rate for the optimizer
CLAMP3_BATCH_SIZE = 256  # Batch size per GPU (single card) during training
LOGIT_SCALE = 1  # Scaling factor for contrastive loss

FREEZE_TEXT = False  # Freeze the weights of the text model and text projection layer
TEXT_DROPOUT = True  # Whether to apply dropout during text processing
CLAMP3_DETERMINISTIC = True  # Ensures deterministic results with random seeds
CLAMP3_LOAD_M3 = True  # Load weights from the M3 model
CLAMP3_WANDB_LOG = True  # Enable logging to Weights and Biases
CLAMP3_LOAD_CKPT = True  # Load weights from a checkpoint if available
SAVE_EVERY = 5  # Save model weights every SAVE_EVERY epochs

CLAMP3_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "weights_clamp3_c2" +
    "_h_size_" + str(CLAMP3_HIDDEN_SIZE) +
    "_t_model_" + TEXT_MODEL_NAME.replace("/", "_") +
    "_t_length_" + str(MAX_TEXT_LENGTH) +
    "_a_size_" + str(AUDIO_HIDDEN_SIZE) +
    "_a_layers_" + str(AUDIO_NUM_LAYERS) +
    "_a_length_" + str(MAX_AUDIO_LENGTH) +
    "_s_size_" + str(M3_HIDDEN_SIZE) +
    "_s_layers_" + str(PATCH_NUM_LAYERS) +
    "_p_size_" + str(PATCH_SIZE) +
    "_p_length_" + str(PATCH_LENGTH) + ".pth"
)  # Path to store CLaMP3 model weights
CLAMP3_LOGS_PATH = CLAMP3_WEIGHTS_PATH.replace("weights", "logs").replace("pth", "txt")  # Path to save training logs

# -------------------- Configuration for LoRA Adapter Training ----------------
# LoRA Hyperparameters (recommended settings for transformers)
LORA_R = 4                    # LoRA rank (REDUCED for faster training)
LORA_ALPHA = 8                # LoRA alpha scaling factor (typically 2 * rank)
LORA_DROPOUT = 0.1            # LoRA dropout for regularization
LORA_TARGET_MODULES = ["query", "key", "value"]  # Attention projection layers in BertModel

# LoRA Training Configuration - 2 HOUR training per adapter
LORA_LEARNING_RATE = 2e-3     # 0.002 - High learning rate for faster convergence
LORA_NUM_EPOCHS = 20          # 20 epochs for ~2 hour training (6 min/epoch × 20 = 120 min)
LORA_BATCH_SIZE = 32          # Batch size per GPU (keep at 32 to avoid OOM)
LORA_TRAIN_SAMPLES = 12000    # ~12K samples (375 batches/epoch × 20 epochs)
LORA_EVAL_SAMPLES = 500       # 500 eval samples for faster validation

# LoRA Adapter Paths
LORA_ABC_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "adapters", "lora_abc_adapter")
LORA_MTF_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "adapters", "lora_mtf_adapter")

# LoRA Training Data Paths (relative to repository root)
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
# ABC (sheet music) - PDMX dataset for sheet music training
LORA_ABC_TRAIN_JSONL = os.path.join(REPO_ROOT, "data", "training", "clamp3_train_abc.jsonl")
# MTF (MIDI) - MidiCaps dataset for MIDI training
LORA_MTF_TRAIN_JSONL = os.path.join(REPO_ROOT, "data", "midicaps_splits", "midicaps_train.jsonl")
LORA_MTF_VAL_JSONL = os.path.join(REPO_ROOT, "data", "midicaps_splits", "midicaps_val.jsonl")
LORA_MTF_TEST_JSONL = os.path.join(REPO_ROOT, "data", "midicaps_splits", "midicaps_test.jsonl")

# Training Mode: Enable/disable adapter training
TRAIN_ABC_ADAPTER = True   # Enable ABC training (PDMX sheet music)
TRAIN_MTF_ADAPTER = True   # Enable MTF training (MidiCaps MIDI)

# LoRA Logs
LORA_LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs", "lora_training")
