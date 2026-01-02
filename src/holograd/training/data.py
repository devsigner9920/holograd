from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, List, Union, Protocol
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


@dataclass
class BatchData:
    input_ids: NDArray[np.int64]
    labels: NDArray[np.int64]
    indices: NDArray[np.int64]
    batch_seed: int


class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        pass

    def get_batch(self, indices: NDArray[np.int64]) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        input_ids = []
        labels = []
        for idx in indices:
            inp, lbl = self[idx]
            input_ids.append(inp)
            labels.append(lbl)
        return np.stack(input_ids), np.stack(labels)


class SyntheticDataset(BaseDataset):
    def __init__(
        self,
        vocab_size: int = 50257,
        num_samples: int = 10000,
        seq_length: int = 256,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.data = rng.integers(0, vocab_size, size=(num_samples, seq_length + 1), dtype=np.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]


class WikiTextDataset(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        seq_length: int = 256,
        dataset_name: str = "wikitext-103-raw-v1",
        tokenizer_name: str = "gpt2",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.seq_length = seq_length
        self.split = split
        self._load_data(dataset_name, tokenizer_name, max_samples, cache_dir)

    def _load_data(
        self,
        dataset_name: str,
        tokenizer_name: str,
        max_samples: Optional[int],
        cache_dir: Optional[str],
    ) -> None:
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "WikiTextDataset requires 'datasets' and 'transformers' packages. "
                "Install with: pip install datasets transformers"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = self.tokenizer.vocab_size

        dataset = load_dataset("wikitext", dataset_name, split=self.split, cache_dir=cache_dir)

        all_tokens = []
        for item in dataset:
            text = item["text"]
            if text.strip():
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)

        chunk_size = self.seq_length + 1
        num_chunks = len(all_tokens) // chunk_size

        if max_samples is not None:
            num_chunks = min(num_chunks, max_samples)

        all_tokens = all_tokens[: num_chunks * chunk_size]
        self.data = np.array(all_tokens, dtype=np.int64).reshape(num_chunks, chunk_size)

        self.num_samples = num_chunks

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]


class DataLoader:
    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)
        self._base_seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[BatchData]:
        n = len(self.dataset)
        indices = np.arange(n)

        if self.shuffle:
            self._rng.shuffle(indices)

        batch_idx = 0
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n:
                if self.drop_last:
                    break
                end = n

            batch_indices = indices[start:end]
            input_ids, labels = self.dataset.get_batch(batch_indices)

            batch_seed = self._base_seed + self._epoch * 10000 + batch_idx

            yield BatchData(
                input_ids=input_ids,
                labels=labels,
                indices=batch_indices,
                batch_seed=batch_seed,
            )
            batch_idx += 1

        self._epoch += 1

    def get_deterministic_batch(self, step: int) -> BatchData:
        rng = np.random.default_rng(self._base_seed + step)
        indices = rng.choice(len(self.dataset), size=self.batch_size, replace=False)
        input_ids, labels = self.dataset.get_batch(indices)

        return BatchData(
            input_ids=input_ids,
            labels=labels,
            indices=indices,
            batch_seed=self._base_seed + step,
        )


def create_synthetic_data(
    vocab_size: int = 50257,
    num_train_samples: int = 10000,
    num_val_samples: int = 1000,
    seq_length: int = 256,
    batch_size: int = 8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = SyntheticDataset(
        vocab_size=vocab_size,
        num_samples=num_train_samples,
        seq_length=seq_length,
        seed=seed,
    )

    val_dataset = SyntheticDataset(
        vocab_size=vocab_size,
        num_samples=num_val_samples,
        seq_length=seq_length,
        seed=seed + 1000000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    return train_loader, val_loader


def create_wikitext_data(
    seq_length: int = 256,
    batch_size: int = 8,
    seed: int = 42,
    dataset_name: str = "wikitext-103-raw-v1",
    tokenizer_name: str = "gpt2",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, int]:
    train_dataset = WikiTextDataset(
        split="train",
        seq_length=seq_length,
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_samples=max_train_samples,
        cache_dir=cache_dir,
    )

    val_dataset = WikiTextDataset(
        split="validation",
        seq_length=seq_length,
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_samples=max_val_samples,
        cache_dir=cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    return train_loader, val_loader, train_dataset.vocab_size
