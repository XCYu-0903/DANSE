import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, GRU, Linear, MultiheadAttention


class FFNSwiGLU(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super().__init__()
        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        self.linear = Linear(d_model * 2 if bidirectional else d_model, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.linear(x1 * F.silu(x2))
        x = self.dropout(x)
        return x


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        super().__init__()
        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // num_groups
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = bias
        if bias:
            self.beta = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.eps = eps

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))
        norm = input.norm(2, dim=-1, keepdim=True)
        rms = norm * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta
        return output


class TFCrossModalTransformerMultimodal(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0):
        super().__init__()
        self.norm1 = RMSGroupNorm(num_groups=4, dim=d_model)
        self.norm2 = RMSGroupNorm(num_groups=4, dim=d_model)
        self.attention_AA = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attention_AT = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.norm3 = RMSGroupNorm(num_groups=4, dim=d_model)
        self.ffn = FFNSwiGLU(d_model, bidirectional=True)
        self.dropout3 = Dropout(dropout)
        self.norm4 = RMSGroupNorm(num_groups=4, dim=d_model)

    def forward(
            self,
            audio_features,
            text_features,
            text_key_padding_mask=None,
            text_time_attn_bias=None,
            text_time_token_mask=None,
    ):
        batch_size = text_features.size(0)
        k = text_features.size(1)
        torf = audio_features.size(0) // batch_size
        fort = audio_features.size(1)
        c = audio_features.size(2)

        text_kv = text_features.unsqueeze(-1)
        text_kv = text_kv.expand(-1, -1, -1, torf)
        text_kv = text_kv.permute(0, 3, 1, 2).contiguous().view(batch_size * torf, k, c)

        if text_key_padding_mask is not None:
            text_key_padding_mask = text_key_padding_mask.unsqueeze(1)
            text_key_padding_mask = text_key_padding_mask.expand(-1, torf, -1)
            text_key_padding_mask = text_key_padding_mask.contiguous().view(batch_size * torf, k)

        attn_mask = None
        if text_time_attn_bias is not None:
            text_time_attn_bias = text_time_attn_bias.to(device=audio_features.device, dtype=audio_features.dtype)
            if text_time_attn_bias.size(0) != batch_size or text_time_attn_bias.size(1) != fort or text_time_attn_bias.size(2) != k:
                raise ValueError(
                    f"text_time_attn_bias must be [B, T, k] = [{batch_size}, {fort}, {k}], "
                    f"got {list(text_time_attn_bias.shape)}"
                )
            text_time_attn_bias = text_time_attn_bias.unsqueeze(1)
            text_time_attn_bias = text_time_attn_bias.expand(-1, torf, -1, -1)
            text_time_attn_bias = text_time_attn_bias.contiguous().view(batch_size * torf, fort, k)
            attn_mask = text_time_attn_bias.repeat_interleave(self.attention_AT.num_heads, dim=0)

        if text_time_token_mask is not None:
            if text_time_token_mask.size(0) != batch_size or text_time_token_mask.size(1) != fort or text_time_token_mask.size(2) != k:
                raise ValueError(
                    f"text_time_token_mask must be [B, T, k] = [{batch_size}, {fort}, {k}], "
                    f"got {list(text_time_token_mask.shape)}"
                )
            disallow_mask = ~text_time_token_mask.to(device=audio_features.device, dtype=torch.bool)
            disallow_mask = disallow_mask.unsqueeze(1)
            disallow_mask = disallow_mask.expand(-1, torf, -1, -1)
            disallow_mask = disallow_mask.contiguous().view(batch_size * torf, fort, k)
            disallow_mask = disallow_mask.repeat_interleave(self.attention_AT.num_heads, dim=0)
            if attn_mask is None:
                attn_mask = disallow_mask
            else:
                attn_mask = attn_mask.masked_fill(disallow_mask, float("-inf"))

        x = audio_features
        audio_qkv = self.norm1(x)
        attn_out_aa, _ = self.attention_AA(audio_qkv, audio_qkv, audio_qkv)
        x = x + self.dropout1(attn_out_aa)

        xt = self.norm2(x)
        attn_out_at, _ = self.attention_AT(
            xt,
            text_kv,
            text_kv,
            attn_mask=attn_mask,
            key_padding_mask=text_key_padding_mask,
        )
        x = x + self.dropout2(attn_out_at)

        xt = self.norm3(x)
        xt = self.ffn(xt)
        x = x + self.dropout3(xt)
        x = self.norm4(x)
        return x


class TFCrossModalTransformerUnimodal(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0):
        super().__init__()
        self.norm1 = RMSGroupNorm(num_groups=4, dim=d_model)
        self.norm2 = RMSGroupNorm(num_groups=4, dim=d_model)
        self.attention_AA = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = Dropout(dropout)
        self.ffn = FFNSwiGLU(d_model, bidirectional=True)
        self.dropout2 = Dropout(dropout)
        self.norm4 = RMSGroupNorm(num_groups=4, dim=d_model)

    def forward(self, audio_features):
        x = audio_features
        audio_qkv = self.norm1(x)
        attn_out_aa, _ = self.attention_AA(audio_qkv, audio_qkv, audio_qkv)
        x = x + self.dropout1(attn_out_aa)
        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)
        x = self.norm4(x)
        return x
