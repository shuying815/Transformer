import torch
import torch.nn as nn
import math

# -----------------------------------------------------------------------------
# 模型架构
# -----------------------------------------------------------------------------

# 多头注意力机制：q,k,v线性映射->Q,K,V，attn=Q*K.T(), output=attn*V
# embed_dim: 嵌入维度
# num_heads: 头数量
# dropout: 丢弃率
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0 # 保证数据可以平均分配到每个头

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([embed_dim // num_heads])).to("cuda") # 缩放权重

    # 自注意力中q,k,v值相同
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention = self.dropout(torch.softmax(attention, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.num_heads * (self.embed_dim // self.num_heads))
        x = self.fc(x)

        return x

# 全连接前馈层：线性->激活->线性
# d_model: 输入维度(模型嵌入维度)
# d_ff: 全连接层中间维度
# dropout: 丢弃率
# activation: 中间激活函数
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 激活函数一般是relu或者gelu
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Unsupported activation: choose 'relu' or 'gelu'")

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out

# 残差归一化层：残差相加->层归一化
# d_model: 模型维度
# dropout: 丢弃率
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # sublayer_output, MHA或FFN
    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))

# 位置编码
# d_model: 模型维度，方便与token相加
# max_len: 序列长度
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建位置矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数维使用sin，奇数维使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 升维
        pe = pe.unsqueeze(0)

        # 注册为buffer，不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# 编码器层：MHA->AddNorm->FFN->AddNorm
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.MHA = MultiheadAttention(d_model, num_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, x, src_mask=None):
        # 编码器中为自注意力
        attn_out = self.MHA(x, x, x, mask=src_mask)
        x = self.addnorm1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)
        return x

# 解码器层: MaskMHA->AddNorm->CrossMHA->AddNorm->FFN->AddNorm
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_MHA = MultiheadAttention(d_model, num_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)

        self.cross_MHA = MultiheadAttention(d_model, num_heads, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        self_attn_out = self.self_MHA(x, x, x, mask=tgt_mask)
        x = self.addnorm1(x, self_attn_out)

        cross_attn_out = self.cross_MHA(x, enc_out, enc_out, mask=memory_mask)
        x = self.addnorm2(x, cross_attn_out)

        ffn_out = self.ffn(x)
        x = self.addnorm3(x, ffn_out)
        return x

# Transformer: 编码器->解码器
class Transformer(nn.Module):
    def __init__(self, src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 max_len=5000,
                 dropout=0.1,
                 share_embeddings=False
                 ):
        super().__init__()

        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        if share_embeddings:
            assert src_vocab_size == tgt_vocab_size, "To share embeddings vocab size must match"
            self.tgt_embedding.weight = self.src_embedding.weight

        self.pos_embedding = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src_tokens, src_mask=None):
        x = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        x = self.pos_embedding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt_tokens, memory, tgt_mask=None, memory_mask=None):
        x = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)
        x = self.pos_embedding(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encode(src_tokens, src_mask)
        out = self.decode(tgt_tokens, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits = self.output_proj(out)
        return logits

# padding_mask: 填充序列到相同长度，方便矩阵运算
def make_padding_mask(tokens, pad_idx=0):
    mask = (tokens != pad_idx).unsqueeze(1).unsqueeze(2)
    return (tokens != pad_idx)  # [B, T]

# 源token掩码：pad_mask
def make_src_mask(src_mask, pad_idx=0):
    pad_mask = (src_mask != pad_idx)  # (B, T_src)
    return pad_mask.unsqueeze(1).expand(-1, src_mask.size(1), -1)  # (B, T_src, T_src)

# causal_mask+pad_mask
def make_tgt_mask(tgt_tokens, pad_idx=0):
    B, T = tgt_tokens.size()
    causal = torch.tril(torch.ones((T, T), dtype=torch.bool, device=tgt_tokens.device))  # (T, T)
    pad_mask = (tgt_tokens != pad_idx).unsqueeze(1).expand(-1, T, -1)  # (B, T, T)
    return pad_mask & causal.unsqueeze(0)  # (B, T, T)

# 目标token掩码，防止看到未来数据
# tgt_mask: 目标token
def causal_mask(tgt_tokens):
    B, T = tgt_tokens.size()
    causal = torch.tril(torch.ones((T, T), dtype=torch.bool, device=tgt_tokens.device))  # (T, T)
    #mask = causal.expand(B, T, T)
    return causal# (B, T, T)