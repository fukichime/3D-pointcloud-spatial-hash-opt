DATA_ROOT = "data/processed"
CATEGORIES = ["airplane", "car", "chair", "table", "sofa", "bed", "bookshelf", "desk", "dresser", "monitor"]
NUM_CLASSES = len(CATEGORIES)
GRID_SIZE = 32


FEATURE_DIM = 8
HASH_TABLE_SIZES = [2**20, 2**16, 2**12, 2**8]
DEFAULT_T = 2**14 


HIDDEN_DIMS = [64, 32]
DROPOUT = 0.1


EPOCHS = 10 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5


CNN_HIDDEN_CHANNELS = [16, 32, 64]
CNN_DROPOUT = 0.1

SEED = 42  
SAVE_MODELS = True 
OUTPUT_DIR = "experiment_results"