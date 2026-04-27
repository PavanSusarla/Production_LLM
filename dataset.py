import torch
from torch.utils.data import Dataset
from config import Config
from tokenizer import BPETokenizer
from pathlib import Path


class TextDataset(Dataset):
    """
    PyTorch Dataset for autoregressive language modeling.

    WHY THIS CLASS EXISTS:
    ---------------------------------------------------------
    Converts a long token sequence into training samples.

    Each sample:
        x = [t1, t2, ..., tN]
        y = [t2, t3, ..., tN+1]

    This aligns with next-token prediction objective.

    DESIGN CHOICE:
    ---------------------------------------------------------
    Uses sliding window over token sequence.

    WHY:
    - Avoids storing all subsequences (memory efficient)
    - Generates samples dynamically

    TRADE-OFF:
    ---------------------------------------------------------
    + Memory efficient
    + Simple implementation
    - Overlapping sequences → redundant computation
    """


    def __init__(self, data: torch.Tensor, block_size: int):
        """
        PARAMETERS:
        ---------------------------------------------------------
        data : torch.Tensor
            Tokenized dataset (1D tensor of token IDs)

        block_size : int
            Length of each training sequence

        WHY block_size:
        - Defines context window
        - Must match model input size

        WHAT IF OMITTED:
        - Cannot create fixed-size sequences
        """

        self.data = data
        self.block_size = block_size


    def __len__(self):
        """
        Returns number of possible training samples.

        WHY:
        ---------------------------------------------------------
        Each sample needs block_size + 1 tokens (for x and y)

        FORMULA:
        len(data) - block_size

        EXAMPLE:
        data length = 1000, block_size = 128
        → 872 samples

        EDGE CASE:
        If len(data) < block_size:
        → returns negative → should be handled upstream
        """

        return len(self.data) - self.block_size


    def __getitem__(self, idx):
        """
        Returns one training sample.

        PARAMETERS:
        ---------------------------------------------------------
        idx : int
            Starting index

        RETURNS:
        ---------------------------------------------------------
        x : (block_size,)
        y : (block_size,)

        LOGIC:
        ---------------------------------------------------------
        x = tokens[idx : idx + block_size]
        y = tokens[idx + 1 : idx + block_size + 1]

        WHY SHIFT BY 1:
        - Aligns input with next-token prediction

        TIME COMPLEXITY:
        - O(block_size)
        """

        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]

        return x, y



def load_datasets(config: Config):
    """
    Loads, tokenizes, and splits dataset.

    WHY THIS FUNCTION EXISTS:
    ---------------------------------------------------------
    Centralizes data loading pipeline.

    STEPS:
    ---------------------------------------------------------
    1. Load raw text
    2. Tokenize
    3. Split into train/validation
    4. Wrap into Dataset objects
    """

    tokenizer = BPETokenizer()


    # Step 1: Load file
    # ---------------------------------------------------------
    if not Path(config.data_path).exists():
        raise FileNotFoundError(f"{config.data_path} not found")

    with open(config.data_path, 'r', encoding='utf-8') as f:
        text = f.read()

        # Optional truncation
        # -----------------------------------------------------
        # WHY:
        # - Prevent loading huge datasets
        if config.max_data_size:
            text = text[:config.max_data_size]


    print(f"📖 Loaded {len(text):,} characters")


    # Step 2: Tokenization
    # ---------------------------------------------------------
    data = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long  # REQUIRED for embedding lookup
    )

    print(f"🔤 Tokenized to {len(data):,} tokens")


    # Step 3: Train-validation split
    # ---------------------------------------------------------
    n = int(config.train_split * len(data))

    train_data = TextDataset(data[:n], config.block_size)
    val_data = TextDataset(data[n:], config.block_size)


    return train_data, val_data



class BatchSampler:
    """
    Custom batch sampler (manual batching instead of DataLoader).

    WHY THIS CLASS EXISTS:
    ---------------------------------------------------------
    Provides fine control over:
    - Sampling strategy
    - Device transfer
    - Memory optimization

    TRADE-OFF:
    ---------------------------------------------------------
    + Simpler than DataLoader
    + Full control
    - No multiprocessing
    - Slower for large datasets
    """


    def __init__(self, dataset, batch_size: int, device: str):
        """
        PARAMETERS:
        ---------------------------------------------------------
        dataset : TextDataset
        batch_size : int
        device : str ('cpu' or 'cuda')

        WHY is_cuda flag:
        - Avoid repeated string comparison
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.is_cuda = device == 'cuda'


    def get_batch(self, eval_mode: bool = False):
        """
        Generates one batch.

        PARAMETERS:
        ---------------------------------------------------------
        eval_mode : bool
            (Currently unused, placeholder for future logic)

        RETURNS:
        ---------------------------------------------------------
        xb : (batch_size, block_size)
        yb : (batch_size, block_size)
        """

        # Step 1: Sample random indices
        # ---------------------------------------------------------
        # WHY random:
        # - Prevents model overfitting sequence order
        ix = torch.randint(
            0,
            len(self.dataset),
            (self.batch_size,)
        )


        xb, yb = [], []


        # Step 2: Collect samples
        # ---------------------------------------------------------
        for i in ix:
            x, y = self.dataset[i]
            xb.append(x)
            yb.append(y)


        # Step 3: Stack into tensors
        # ---------------------------------------------------------
        xb = torch.stack(xb)
        yb = torch.stack(yb)


        # Step 4: Device transfer
        # ---------------------------------------------------------
        # WHY pin_memory:
        # - Faster CPU → GPU transfer
        #
        # WHY non_blocking:
        # - Overlaps data transfer with computation
        if self.is_cuda:
            xb = xb.pin_memory().to(self.device, non_blocking=True)
            yb = yb.pin_memory().to(self.device, non_blocking=True)
        else:
            xb = xb.to(self.device)
            yb = yb.to(self.device)


        return xb, yb