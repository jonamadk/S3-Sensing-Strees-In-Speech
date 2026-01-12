"""
Transformer-based Speech-to-Text Model
Implements an encoder-decoder transformer architecture for ASR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot product attention"""
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """Combine the heads"""
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query tensor [batch_size, seq_len, d_model]
            K: Key tensor [batch_size, seq_len, d_model]
            V: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """Single decoder layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention with residual connection
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class SpeechToTextTransformer(nn.Module):
    """
    Complete Speech-to-Text Transformer Model

    Args:
        input_dim: Dimension of input features (e.g., mel-spectrogram channels)
        d_model: Model dimension
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        vocab_size: Size of output vocabulary
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(self, input_dim=80, d_model=512, num_encoder_layers=6,
                 num_decoder_layers=6, num_heads=8, d_ff=2048,
                 vocab_size=5000, max_seq_length=5000, dropout=0.1):
        super(SpeechToTextTransformer, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Input projection for audio features
        self.input_projection = nn.Linear(input_dim, d_model)

        # Embedding for text tokens
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encodings
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """Generate mask for decoder to prevent attending to future positions"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def encode(self, src, src_mask=None):
        """
        Encode audio features

        Args:
            src: Audio features [batch_size, seq_len, input_dim]
            src_mask: Optional source mask

        Returns:
            Encoded features [batch_size, seq_len, d_model]
        """
        # Project input features to model dimension
        x = self.input_projection(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)

        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Decode text tokens

        Args:
            tgt: Target tokens [batch_size, seq_len]
            memory: Encoded audio features [batch_size, seq_len, d_model]
            tgt_mask: Optional target mask
            memory_mask: Optional memory mask

        Returns:
            Decoded features [batch_size, seq_len, d_model]
        """
        # Embed target tokens
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_decoder(x.transpose(0, 1)).transpose(0, 1)

        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x, memory, memory_mask, tgt_mask)

        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass

        Args:
            src: Audio features [batch_size, seq_len, input_dim]
            tgt: Target tokens [batch_size, seq_len]
            src_mask: Optional source mask
            tgt_mask: Optional target mask

        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        # Encode audio
        memory = self.encode(src, src_mask)

        # Decode text
        output = self.decode(tgt, memory, tgt_mask, src_mask)

        # Project to vocabulary
        logits = self.output_projection(output)

        return logits

    def greedy_decode(self, src, max_len=100, start_token=1, end_token=2):
        """
        Greedy decoding for inference

        Args:
            src: Audio features [batch_size, seq_len, input_dim]
            max_len: Maximum generation length
            start_token: Start token ID
            end_token: End token ID

        Returns:
            Decoded token IDs [batch_size, seq_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode audio
        memory = self.encode(src)

        # Initialize with start token
        ys = torch.ones(batch_size, 1).fill_(start_token).long().to(device)

        for i in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(
                ys.size(1)).to(device)

            # Decode
            out = self.decode(ys, memory, tgt_mask=tgt_mask)

            # Get next token
            prob = self.output_projection(out[:, -1, :])
            next_word = torch.argmax(prob, dim=-1).unsqueeze(1)

            # Append to output
            ys = torch.cat([ys, next_word], dim=1)

            # Check if all sequences have generated end token
            if (next_word == end_token).all():
                break

        return ys


if __name__ == "__main__":
    # Test model
    model = SpeechToTextTransformer(
        input_dim=80,
        d_model=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        d_ff=2048,
        vocab_size=5000,
        dropout=0.1
    )

    # Test forward pass
    batch_size = 2
    src_seq_len = 100
    tgt_seq_len = 20

    src = torch.randn(batch_size, src_seq_len, 80)
    tgt = torch.randint(0, 5000, (batch_size, tgt_seq_len))

    output = model(src, tgt)
    print(f"Input shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")

    # Test greedy decoding
    decoded = model.greedy_decode(src, max_len=30)
    print(f"Decoded shape: {decoded.shape}")
