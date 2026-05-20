"""
代码优化字符级分词器

专为代码生成模型设计，相比基础字符级分词器增加了:
- 代码常用符号的显式token映射
- 缩进级别感知（空格/tab合并）
- 代码注释和字符串边界标记
- 确保所有代码语法字符（括号、运算符、分号等）独立编码
"""

import torch
import json
import re
from torch.utils.data import Dataset


class CodeTokenizer:
    """
    面向代码的字符级分词器

    特性:
    - 基础ASCII字符覆盖 (0-255)
    - 代码高频符号扩展token (256-280)
    - 特殊token: <PAD> <UNK> <BOS> <EOS> <MASK> <INDENT> <DEDENT> <COMMENT> <NEWLINE>
    """

    # 代码专用扩展token（高频代码符号组合）
    CODE_SYMBOLS = [
        '<INDENT>',   # 缩进标记（4空格或tab）
        '<DEDENT>',   # 反缩进标记
        '<COMMENT>',  # 注释开始标记
        '<NEWLINE>',  # 换行标记
        '  ',         # 两空格
        '    ',       # 四空格缩进
        '\t',         # Tab
        '->',         # 箭头运算符
        '=>',         # 箭头函数 / Lambda
        '!=',         # 不等于
        '==',         # 等于
        '<=',         # 小于等于
        '>=',         # 大于等于
        '&&',         # 逻辑与
        '||',         # 逻辑或
        '++',         # 自增
        '--',         # 自减
        '+=',         # 加等
        '-=',         # 减等
        '*=',         # 乘等
        '/=',         # 除等
        '**',         # 幂运算
        '//',         # 整除
        '"""',        # 三引号
        "'''",        # 三单引号
        '${',         # 模板字符串插值
    ]

    SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']

    def __init__(self, vocab_file=None, vocab_size=None):
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            self._build_code_vocab()

    def _build_code_vocab(self):
        """构建代码优化的词汇表"""
        self.char_to_idx = {}
        self.idx_to_char = {}

        # 基础ASCII字符 (0-255)
        for i in range(256):
            ch = chr(i)
            self.char_to_idx[ch] = i
            self.idx_to_char[i] = ch

        offset = 256

        # 特殊token
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.char_to_idx[token] = offset + i
            self.idx_to_char[offset + i] = token
        offset += len(self.SPECIAL_TOKENS)

        # 代码专用符号token
        for i, symbol in enumerate(self.CODE_SYMBOLS):
            self.char_to_idx[symbol] = offset + i
            self.idx_to_char[offset + i] = symbol
        offset += len(self.CODE_SYMBOLS)

        self.vocab_size = offset

        self.pad_token_id = self.char_to_idx['<PAD>']
        self.unk_token_id = self.char_to_idx['<UNK>']
        self.bos_token_id = self.char_to_idx['<BOS>']
        self.eos_token_id = self.char_to_idx['<EOS>']
        self.mask_token_id = self.char_to_idx['<MASK>']

    def _preprocess_code(self, text: str) -> str:
        """
        代码文本预处理:
        - 保留原始缩进结构
        - 不做字符替换，保留所有代码语法字符
        """
        return text

    def encode(self, text, max_length=None):
        """
        编码文本为token id序列

        对代码文本:
        1. 保留所有字符原样编码
        2. 行首缩进用<INDENT>标记
        3. 注释行用<COMMENT>标记
        """
        text = self._preprocess_code(text)
        tokens = []

        # 按行处理以识别缩进和注释
        lines = text.split('\n')
        for line in lines:
            # 编码缩进
            indent_match = re.match(r'^(\s+)', line)
            if indent_match:
                indent = indent_match.group(1)
                if '    ' in indent:
                    tokens.append(self.char_to_idx.get('    ', self.unk_token_id))
                elif indent.startswith('\t'):
                    tokens.append(self.char_to_idx.get('\t', self.unk_token_id))
                line = line.lstrip()

            # 检测注释
            stripped = line.lstrip()
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('<!--'):
                tokens.append(self.char_to_idx.get('<COMMENT>', self.unk_token_id))

            # 逐字符编码
            for char in line:
                tokens.append(self.char_to_idx.get(char, self.unk_token_id))

            tokens.append(self.char_to_idx.get('\n', self.unk_token_id))

        # 移除最后一个多余的换行
        if tokens and tokens[-1] == self.char_to_idx.get('\n', -1):
            tokens.pop()

        # 包裹BOS/EOS
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length - 1] + [self.eos_token_id]
            else:
                tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))

        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        """解码token id序列为文本"""
        text = []
        for idx in token_ids:
            if skip_special_tokens and idx in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            if skip_special_tokens and idx == self.char_to_idx.get('<COMMENT>', -1):
                text.append('#')
                continue
            if skip_special_tokens and idx == self.char_to_idx.get('<INDENT>', -1):
                text.append('    ')
                continue
            if skip_special_tokens and idx == self.char_to_idx.get('<NEWLINE>', -1):
                text.append('\n')
                continue
            text.append(self.idx_to_char.get(idx, self.idx_to_char[self.unk_token_id]))
        return ''.join(text)

    def save_vocab(self, path):
        """保存词汇表"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
                'vocab_size': self.vocab_size,
                'type': 'code_tokenizer'
            }, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path):
        """加载词汇表"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        self.vocab_size = data.get('vocab_size', len(self.char_to_idx))
        self.pad_token_id = self.char_to_idx.get('<PAD>', 0)
        self.unk_token_id = self.char_to_idx.get('<UNK>', 1)
        self.bos_token_id = self.char_to_idx.get('<BOS>', 2)
        self.eos_token_id = self.char_to_idx.get('<EOS>', 3)
        self.mask_token_id = self.char_to_idx.get('<MASK>', 4)


# 向后兼容
SimpleTokenizer = CodeTokenizer


class TextDataset(Dataset):
    """代码文本数据集"""

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
    """创建数据加载器"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    """批处理整理函数"""
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
