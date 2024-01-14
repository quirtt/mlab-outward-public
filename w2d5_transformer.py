from ast import keyword
import math
from collections import OrderedDict
from typing import Optional

import einops
import torch
from torch import Tensor, nn
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer
import utils
from w2d1_solution import BertConfig, BertSelfAttention


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.batch_first = batch_first
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe[None, : x.size(1), :]
        else:
            x = x + self.pe[: x.size(0), None, :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_hid):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = BertSelfAttention(BertConfig(hidden_size=d_model, num_heads=nhead, head_size=d_model // nhead))

        self.linear1 = nn.Linear(d_model, d_hid)
        self.linear2 = nn.Linear(d_hid, d_model)
        self.activation = nn.ReLU()

    def forward(self, x, padding_mask):
        x = x + self.attn(x, padding_mask)
        x = x + self.mlp(x)
        return x

    def attn(self, x, padding_mask):
        x = self.norm1(x)
        additive_mask = torch.where(padding_mask, -10000, 0)[:, None, None, :]  # [batch, head=1, qpos=1, kpos]
        # print(additive_mask)
        x = self.self_attn(x, additive_attention_mask=additive_mask)
        return x

    def mlp(self, x):
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class ParenTransformer(nn.Module):
    def __init__(
        self, ntoken: int, nclasses: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.0
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers: utils.StaticModuleList[TransformerBlock] = utils.StaticModuleList(
            [TransformerBlock(d_model, nhead, d_hid) for _ in range(nlayers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.nhead = nhead

        self.decoder = nn.Linear(d_model, nclasses)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        padding_mask = x == SimpleTokenizer.PAD_TOKEN
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for l in self.layers:
            x = l(x, padding_mask)
        x = self.norm(x)
        x = self.decoder(x)
        return self.softmax(x[:, 0, :])

    def load_simple_transformer_state_dict(self, state_dict):
        new_dict = OrderedDict()
        for key, weight in state_dict.items():
            key = key.replace("transformer_encoder.", "").replace("out_proj", "project_output")
            if "in_proj_" in key:
                q, k, v = torch.tensor_split(weight, 3)
                # maps both in_proj_weight -> project_query.weight and in_proj_bias -> project_query.bias
                new_dict[key.replace("in_proj_", "project_query.")] = q
                new_dict[key.replace("in_proj_", "project_key.")] = k
                new_dict[key.replace("in_proj_", "project_value.")] = v
            else:
                if key == "pos_encoder.pe":
                    weight = weight[:, 0, :]  # remove extra dimension from posencoder due to earlier architechture
                new_dict[key] = weight
        self.load_state_dict(new_dict)


class SimpleTokenizer:
    START_TOKEN = 0
    PAD_TOKEN = 1
    END_TOKEN = 2
    base_d = {"[start]": START_TOKEN, "[pad]": PAD_TOKEN, "[end]": END_TOKEN}

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        # the 3 is because there are 3 special tokens (defined just above)
        self.t_to_i = {**{c: i + 3 for i, c in enumerate(alphabet)}, **self.base_d}
        self.i_to_t = {i: c for c, i in self.t_to_i.items()}

    def tokenize(self, strs: list[str], max_len: Optional[int] = None) -> torch.Tensor:
        def c_to_int(c: str) -> int:
            if c in self.t_to_i:
                return self.t_to_i[c]
            else:
                raise ValueError(c)

        if max_len is None:
            max_len = max((max(len(s) for s in strs), 1))

        ints = [
            [self.START_TOKEN] + [c_to_int(c) for c in s] + [self.END_TOKEN] + [self.PAD_TOKEN] * (max_len - len(s))
            for s in strs
        ]
        return torch.tensor(ints)

    def decode(self, tokens) -> list[str]:
        def int_to_c(c: int) -> str:
            if c < len(self.i_to_t):
                return self.i_to_t[c]
            else:
                raise ValueError(c)

        return [
            "".join(int_to_c(i.item()) for i in seq[1:] if i != self.PAD_TOKEN and i != self.END_TOKEN)
            for seq in tokens
        ]
