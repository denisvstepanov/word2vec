import os
import numpy as np

from datetime import datetime

from typing import Dict, Any, Tuple, List, NamedTuple, Set

from pathlib import Path
import random
from tqdm import tqdm

from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import nltk
from nltk.corpus import stopwords

LOG_DIR = Path.home() / 'train_logs'


def prepare_log_dir(model_name: str, base_dir=LOG_DIR):
    now = datetime.now()
    log_dir = base_dir / f'{model_name}-{now:%Y%m%d-%H%M-%S}'
    log_dir.mkdir(parents=True)
    return log_dir


def exclude_stop_words(words: List[str]):
    return [word for word in words if word not in stopwords.words('english')]


class TextProcessor:
    def __init__(self, vocab: Set[str]) -> None:
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(vocab)}

    def decode_ids(self, ids: List[int]) -> str:
        words = [self.idx_to_word[idx] for idx in ids]
        return ' '.join(words)

    def encode_as_ids(self, text: str) -> List[int]:
        text = text.lower()
        words = word_tokenize(text)
        words = exclude_stop_words(words)
        ids = [self.word_to_idx[word] for word in words]
        return ids

    @staticmethod
    def from_content(content: str) -> 'TextProcessor':
        content = content.lower()
        words = word_tokenize(content)
        words = exclude_stop_words(words)
        vocab = set(words)
        return TextProcessor(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_idx)


class Word2VecDatasetItem(NamedTuple):
    center_word: int
    context_words: List[int]


class Word2VecDataset(Dataset):
    def __init__(self, items: List[Word2VecDatasetItem]):
        super(Word2VecDataset, self).__init__()
        self.items = items

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def from_path(path: str, context_size: int) -> 'Tuple[Word2VecDataset, TextProcessor]':
        items: List[Word2VecDatasetItems] = []
        with open(path, 'r') as f:
            content = f.read()
            text_processor = TextProcessor.from_content(content)
            ids = text_processor.encode_as_ids(content)
            for i in range(context_size, len(ids) - context_size):
                context_words = [ids[j] for j in range(i - context_size, i + context_size + 1) if j not in [i]]
                center_word = ids[i]
                items.append(Word2VecDatasetItem(center_word, context_words))
        return Word2VecDataset(items), text_processor


class Word2Vec(nn.Module):
    def __init__(self, emb_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size, vocab_size)
        self._reset_parameters()

    def forward(self, context_word):
        emb = self.emb(context_word)
        hidden = self.linear(emb)
        return F.log_softmax(hidden, dim=-1).squeeze()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device():
    if torch.cuda.device_count() > 0:
        device_no = torch.cuda.current_device()
        return torch.device(f'cuda:{device_no}')
    else:
        return torch.device('cpu')


def load_model(log_dir: Path, epoch: int, model: nn.Module) -> nn.Module:
    model_path = log_dir / f'model-{epoch}'
    device = choose_device()
    with model_path.open('br') as f:
        model_data = torch.load(f, map_location=device)
        model.load_state_dict(model_data['module'])
    return model


def parse_batch(batch, device) -> Tuple[Tensor, Tensor]:
    center_word = batch.center_word
    context_words = batch.context_words
    context_words = torch.stack(context_words, dim=-1)
    batch_size, seq_len = context_words.size()
    return center_word.repeat_interleave(seq_len).to(device), context_words.view(1, -1).to(device)


def normalize_step(batch_num: int, batches: int, epoch: int, base: int = 1000):
    return batch_num * base // batches + base * epoch


def train_model(emb_size=300, epochs=50, batch_size=100, file_name='got.txt'):
    fix_seed(42)
    device = choose_device()
    log_dir = prepare_log_dir(model_name='word2vec')
    writer = SummaryWriter(log_dir=str(log_dir))

    data, text_processor = Word2VecDataset.from_path(file_name, context_size=10)
    model = Word2Vec(emb_size=emb_size, vocab_size=text_processor.vocab_size)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        total_batches = len(data_loader)
        total_loss = 0
        model.train()
        for batch_no, batch in enumerate(tqdm(data_loader, ncols=40, desc=f'Epoch {epoch}')):
            norm_step = normalize_step(batch_no, total_batches, epoch)
            optimizer.zero_grad()
            center_word, context_words = parse_batch(batch, device)
            log_probs = model(context_words)
            loss = loss_function(log_probs, center_word)
            loss.backward()
            optimizer.step()
            writer.add_scalar('main/loss', loss.item(), global_step=norm_step)
            total_loss += loss.item()
        scheduler.step()
        save_model(Path('model'), epoch, model)
        print(f'Loss: {total_loss / total_batches}')


def save_model(log_dir: Path, epoch: int, model: nn.Module) -> None:
    model_path = log_dir / f'model-{epoch}'
    print(f'Saving model to {str(model_path)} ...')
    if not os.path.exists(str(log_dir)):
        os.makedirs('model')
    with model_path.open('bw') as f:
        torch.save({
            'module': model.state_dict()
        }, f)
    print('Done')


if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('punkt')
    train_model()