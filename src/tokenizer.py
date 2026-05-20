import torch
from torch.utils.data import Dataset


class SimpleTokenizer:
    def __init__(self, vocab_file=None, vocab_size=50000):
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            self._build_basic_vocab(vocab_size)
    
    def _build_basic_vocab(self, vocab_size):
        self.char_to_idx = {chr(i): i for i in range(256)}
        self.idx_to_char = {i: chr(i) for i in range(256)}
        
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']
        for i, token in enumerate(special_tokens):
            self.char_to_idx[token] = 256 + i
            self.idx_to_char[256 + i] = token
        
        self.vocab_size = 256 + len(special_tokens)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
    
    def encode(self, text, max_length=None):
        tokens = []
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.unk_token_id))
        
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        text = []
        for idx in token_ids:
            if skip_special_tokens and idx in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            text.append(self.idx_to_char.get(idx, self.idx_to_char[self.unk_token_id]))
        return ''.join(text)
    
    def save_vocab(self, path):
        import json
        with open(path, 'w') as f:
            json.dump({'char_to_idx': self.char_to_idx, 'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}}, f)
    
    def load_vocab(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self._load_texts(file_path)
    
    def _load_texts(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            return texts
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Creating empty dataset.")
            return []
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text, max_length=self.max_length)
        return {
            'input_ids': torch.tensor(encoding, dtype=torch.long),
            'labels': torch.tensor(encoding, dtype=torch.long)
        }


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    attention_mask = (input_ids_padded != 0).long()
    
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded,
        'attention_mask': attention_mask
    }
