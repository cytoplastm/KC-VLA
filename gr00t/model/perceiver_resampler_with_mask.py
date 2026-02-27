import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn

class SquaredReLU(nn.Module):
    """Squared ReLU activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


def feed_forward_layer(dim: int, mult: int = 4, activation: str = 'gelu'):
    """Feed forward layer with given activation function"""

    activations = dict(gelu=nn.GELU, sqrelu=SquaredReLU, relu=nn.ReLU)
    assert activation in activations, f'activation can only be one of {activations.keys()}'

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        activations[activation](),
        nn.Linear(inner_dim, dim, bias=False),
    )

class PerceiverAttentionLayer(nn.Module):
    """Perceiver Attention Layer"""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # trainable components of PerceiverAttentionLayer
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features, latents, key_attention_mask=None):
        """Latent vectors are cross-attending to the features x

        Args:
            features: Batch of features with shape (batch_size, n_features, dim)
            latents: Latent learnt vectors which are used to compute queries with shape (batch_size, n_latents, dim)
            key_attention_mask: Attention mask for the keys, shape (batch_size, n_features)
        Returns:
            Attention score with shape (batch_size, n_latents, dim)
        """
        assert features.ndim == 3
        assert latents.ndim == 3
        assert features.shape[0] == latents.shape[0]
        assert features.shape[2] == latents.shape[2]

        n_heads = self.heads
        n_batch, n_features, dim = features.shape
        n_queries = latents.shape[1]

        # Layer normalization
        x = self.norm_media(features)
        latents = self.norm_latents(latents)

        # Compute the queries from the latents, for all attention heads simultaneously
        q = self.to_q(latents)
        q = rearrange(q, 'b q (h d) -> b h q d', h=n_heads)
        assert q.shape == torch.Size([n_batch, n_heads, n_queries, self.dim_head])

        # Keys and values for all attention heads
        kv_input = torch.cat((x, latents), dim=-2)
        n_features_latents = n_features + n_queries
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        k, v = rearrange_many((k, v), 'b f (h d) -> b h f d', h=n_heads)
        assert v.shape == torch.Size([n_batch, n_heads, n_features_latents, self.dim_head])

        q = q * self.scale

        # Attention scores
        sim = einsum('b h q d, b h f d -> b h q f', q, k)
        # TODO: add support for key_padding_mask
        if key_attention_mask is not None:
            mask = key_attention_mask.unsqueeze(1)       # (batch_size, 1, n_features)
            # 对 features 部分做mask，latents部分默认全1
            latents_mask = torch.ones((n_batch, n_queries), dtype=mask.dtype, device=mask.device)
            full_mask = torch.cat([mask.squeeze(1), latents_mask], dim=1)  # (batch_size, n_features + n_queries)
            full_mask = full_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, n_features + n_queries)
            sim = sim.masked_fill(full_mask==0, float('-inf'))

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        out = einsum('b h q f, b h f v -> b h q v', alphas, v)
        out = rearrange(out, 'b h q v -> b q (h v)')

        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler with multi-head attention layer"""

    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        max_tokens: int = 512,
        ff_mult: int = 4,
        activation: str = 'gelu',
        trainable: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_queries = num_latents

        self.latents = nn.Parameter(torch.randn(num_latents, dim))  # type: ignore[reportPrivateUsage]
        self.time_pos_emb = nn.Parameter(torch.randn(max_tokens,  dim))  # type: ignore[reportPrivateUsage]

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads),
                        feed_forward_layer(dim=dim, mult=ff_mult, activation=activation),
                    ]
                )
            )

        # Layer normalization takes as input the query vector length
        self.norm = nn.LayerNorm(dim)

        self._update_trainable_state(trainable)
        
    def _update_trainable_state(self, trainable: bool = True):
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, x_f: torch.Tensor, key_padding_mask: torch.BoolTensor = None):
        """Run perceiver resampler on the input embeddings

        Args:
            x_f: Input embeddings of shape (batch_size, n_features, d_features)
            key_padding_mask: Mask for the input embeddings of shape (batch_size, n_features)

        Returns:
            Resampler features of shape (batch_size, num_queries, d_features)
        """
        assert x_f.ndim == 3

        batch_size, n_feat, dim = x_f.shape
        key_padding_mask = ~key_padding_mask

        assert dim == self.dim

        # Mask the position embeddings for the padded tokens
        pos_table = self.time_pos_emb
        valid_lens = key_padding_mask.sum(dim=1)
        idx = torch.arange(n_feat, device=key_padding_mask.device).unsqueeze(0).expand(batch_size, -1)
        idx = idx - (n_feat - valid_lens).unsqueeze(1)
        idx = idx.clamp(min=0)
        pos_emb = pos_table[idx]
        pos_emb = pos_emb * key_padding_mask.unsqueeze(-1)

        # Apply the position embeddings
        x_f = x_f + pos_emb

        # Copy the latents for every element in the batch
        x = repeat(self.latents, 'q d -> b q d', b=batch_size)

        # Apply attention and feed forward layer
        for attn, ffw in self.layers:
            x = x + attn(x_f, x, key_padding_mask)
            x = x + ffw(x)

        assert x.shape == torch.Size([batch_size, self.num_queries, self.dim])

        norm = self.norm(x)
        return norm