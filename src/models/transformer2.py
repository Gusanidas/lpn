from typing import Optional
import chex
import jax.numpy as jnp
import jax
from flax import linen as nn
from src.data_utils import make_all_pairs
import time
from src.models.utils import EncoderTransformerConfig, DecoderTransformerConfig, TransformerLayer


class EncoderTransformer(nn.Module):
    config: EncoderTransformerConfig

    @nn.compact
    def __call__(
        self,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
    ) -> tuple[chex.Array, Optional[chex.Array]]:
        """Applies Transformer Encoder on the (input, output) pairs.

        Args:
            pairs: input data as tokens. Shape (*B, P, R, C, 2).
                - P: number of pairs.
                - R: number of rows.
                - C: number of columns.
                - 2: two channels (input and output)
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, P, 2, 2). The last two dimension
                represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].
                Expects grid shapes values to be in [1, max_rows] and [1, max_cols].
            dropout_eval: if false dropout is applied otherwise it is not.

        Returns:
            latent_mu: output of shape (*B, H) representing the mean latent embeddings of the (input, output)
                pairs.
            latent_logvar: output of shape (*B, H) representing the log-variance of the latent embeddings of
                the (input, output) pairs.
        """

        
        pairs = make_all_pairs(pairs, axis=-4)
        grid_shapes = make_all_pairs(grid_shapes, axis=-3)
        x = self.embed_grids(pairs, grid_shapes, dropout_eval)

        # Transformer block.
        pad_mask = self.make_pad_mask(grid_shapes)
        for _ in range(self.config.num_layers):
            x = TransformerLayer(self.config.transformer_layer)(
                embeddings=x,
                dropout_eval=dropout_eval,
                pad_mask=pad_mask,
            )

        # Extract the CLS embedding.
        cls_embed = x[..., 0, :]
        # Project the cls embedding to the program space.
        cls_embed = nn.LayerNorm(
            dtype=self.config.dtype,
            use_bias=self.config.transformer_layer.use_bias,
            use_scale=False,
            name="cls_layer_norm",
        )(cls_embed)
        latent_mu = nn.Dense(
            self.config.latent_dim, use_bias=self.config.latent_projection_bias, dtype=self.config.dtype
        )(cls_embed).astype(jnp.float32)
        if self.config.variational:
            latent_logvar = nn.Dense(
                self.config.latent_dim, use_bias=self.config.latent_projection_bias, dtype=self.config.dtype
            )(cls_embed).astype(jnp.float32)
        else:
            latent_logvar = None

        return latent_mu, latent_logvar

    def embed_grids(self, pairs: chex.Array, grid_shapes: chex.Array, dropout_eval: bool) -> chex.Array:
        config = self.config

        # Position embedding block.
        if self.config.scaled_position_embeddings:
            pos_row_embed = nn.Embed(
                num_embeddings=1,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_row_embed",
            )(jnp.zeros(config.max_rows, dtype=jnp.uint8))
            pos_col_embed = nn.Embed(
                num_embeddings=1,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_col_embed",
            )(jnp.zeros(config.max_cols, dtype=jnp.uint8))
            pos_row_embeds = jnp.arange(1, config.max_rows + 1)[:, None] * pos_row_embed
            pos_col_embeds = jnp.arange(1, config.max_cols + 1)[:, None] * pos_col_embed
            pos_embed = pos_row_embeds[:, None, None, :] + pos_col_embeds[None, :, None, :]
        else:
            pos_row_embed = nn.Embed(
                num_embeddings=config.max_rows,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_row_embed",
            )(jnp.arange(config.max_rows, dtype=jnp.uint8))
            pos_col_embed = nn.Embed(
                num_embeddings=config.max_cols,
                features=config.emb_dim,
                dtype=config.dtype,
                name="pos_col_embed",
            )(jnp.arange(config.max_cols, dtype=jnp.uint8))
            pos_embed = pos_row_embed[:, None, None, :] + pos_col_embed[None, :, None, :]

        # Colors embedding block.
        colors_embed = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.emb_dim,
            dtype=config.dtype,
            name="colors_embed",
        )(pairs)

        # Channels embedding block.
        channels_embed = nn.Embed(
            num_embeddings=2,
            features=config.emb_dim,
            dtype=config.dtype,
            name="channels_embed",
        )(jnp.arange(2, dtype=jnp.uint8))

        examples_embed = nn.Embed(
            num_embeddings=2,
            features=config.emb_dim,
            dtype=config.dtype,
            name="examples_embed",
        )(jnp.arange(2, dtype=jnp.uint8))
        # Combine all the embeddings into a sequence x of shape (*B, 1+2*(R*C), H)
        x = colors_embed + pos_embed + channels_embed + examples_embed[:, None, None, None, :]
        # Flatten the rows, columns and channels.
        x = jnp.reshape(x, (*x.shape[:-5], -1, x.shape[-1]))  # (*B, 2*R*C, H)

        # Embed the grid shape tokens.
        # TODO: potentially switch grid_shapes embeddings to linear embedding for better interpolation
        grid_shapes_row_embed = nn.Embed(
            num_embeddings=config.max_rows,
            features=config.emb_dim,
            dtype=config.dtype,
            name="grid_shapes_row_embed",
        )(grid_shapes[..., 0, :] - 1)
        grid_shapes_row_embed += channels_embed
        grid_shapes_row_embed += examples_embed[:, None, :]
        grid_shapes_col_embed = nn.Embed(
            num_embeddings=config.max_cols,
            features=config.emb_dim,
            dtype=config.dtype,
            name="grid_shapes_col_embed",
        )(grid_shapes[..., 1, :] - 1)
        grid_shapes_col_embed += channels_embed
        grid_shapes_col_embed += examples_embed[:, None, :]
        grid_shapes_embed = jnp.concatenate([grid_shapes_row_embed, grid_shapes_col_embed], axis=-2)
        grid_shapes_embed = jnp.reshape(grid_shapes_embed, (*grid_shapes_embed.shape[:-3], -1, grid_shapes_embed.shape[-1]))
        x = jnp.concatenate([grid_shapes_embed, x], axis=-2)  # (*B, 8+4*R*C, H)

        # Add the cls token.
        cls_token = nn.Embed(
            num_embeddings=1,
            features=config.emb_dim,
            dtype=config.dtype,
            name="cls_token",
        )(jnp.zeros_like(x[..., 0:1, 0], jnp.uint8))
        x = jnp.concatenate([cls_token, x], axis=-2)  # (*B, 1+8+4*R*C, H)
        assert x.shape[-2] == 1 + 8 + 4 * config.max_len  # 3609
        x = nn.Dropout(rate=config.transformer_layer.dropout_rate, name="embed_dropout")(x, dropout_eval)
        return x

    def make_pad_mask(self, grid_shapes: chex.Array) -> chex.Array:
        """Make the pad mask False outside of the grid shapes and True inside.

        Args:
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, 2, 2). The last two dimension
                represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].

        Returns:
            pad mask of shape (*B, 1, T, T) with T = 1 + 8 + 4 * max_rows * max_cols.
        """
        batch_ndims = len(grid_shapes.shape[:-2])
        row_arange_broadcast = jnp.arange(self.config.max_rows).reshape(
            (*batch_ndims * (1,), self.config.max_rows, 1)
        )
        row_mask = row_arange_broadcast < grid_shapes[..., 0:1, :]
        col_arange_broadcast = jnp.arange(self.config.max_cols).reshape(
            (*batch_ndims * (1,), self.config.max_cols, 1)
        )
        col_mask = col_arange_broadcast < grid_shapes[..., 1:2, :]
        pad_mask = row_mask[..., :, None, :] & col_mask[..., None, :, :]
        # Flatten the rows, columns and channels.
        pad_mask = jnp.reshape(pad_mask, (*pad_mask.shape[:-4], 1, -1))
        # Add the masks corresponding to the cls token and grid shapes tokens.
        pad_mask = jnp.concatenate([jnp.ones((*pad_mask.shape[:-1], 1 + 8), bool), pad_mask], axis=-1)
        # Outer product to make the self-attention mask.
        pad_mask = pad_mask[..., :, None] & pad_mask[..., None, :]
        return pad_mask



if __name__ == "__main__":
    import jax

    batch_size = 2
    mini_batch_size = 4
    max_rows = 25
    max_cols = 25
    vocab_size = 10

    # Transformer Encoder.
    encoder_config = EncoderTransformerConfig(
        vocab_size=vocab_size, max_rows=max_rows, max_cols=max_cols, variational=True
    )
    encoder = EncoderTransformer(encoder_config)

    pairs = jax.random.randint(
        jax.random.PRNGKey(0),
        (batch_size, mini_batch_size, max_rows, max_cols, 2),
        minval=0,
        maxval=vocab_size,
    )
    grid_shapes = jnp.full((batch_size, mini_batch_size, 2, 2), 15, jnp.int32)
    variables = encoder.init(jax.random.PRNGKey(0), pairs, grid_shapes, dropout_eval=False)
    num_parameters = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
    print(f"Encoder -> number of parameters: {num_parameters:,}")
    apply_fn = jax.jit(encoder.apply, static_argnames="dropout_eval")
    rngs = {"dropout": jax.random.PRNGKey(0)}
    print("Input shape:", pairs.shape, grid_shapes.shape)

    # Warm-up run for JIT compilation
    _ = apply_fn(variables, pairs, grid_shapes, dropout_eval=False, rngs=rngs)
    
    # Timing the encoder
    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        latent_mu, latent_logvar = apply_fn(variables, pairs, grid_shapes, dropout_eval=False, rngs=rngs)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    print(f"Encoder timing: {avg_time:.4f} seconds per run (average over {num_runs} runs)")

    latent_mu, latent_logvar = apply_fn(variables, pairs, grid_shapes, dropout_eval=False, rngs=rngs)
    print(f"latent_mu shape = {latent_mu.shape}")
    assert latent_mu.shape == (batch_size, mini_batch_size*(mini_batch_size-1)//2, encoder_config.latent_dim)
    if latent_logvar is not None:
        print("Output shape (latent_mu):", latent_mu.shape)
        print("Output shape (latent_logvar):", latent_logvar.shape)
        assert latent_logvar.shape == (batch_size, mini_batch_size*(mini_batch_size-1)//2, encoder_config.latent_dim)
    else:
        print("Output shape:", latent_mu.shape)

    