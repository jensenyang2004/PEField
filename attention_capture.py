"""Capture text->{target-latent, conditioning-image} attention for the mice text-position script.

Only meaningful for calls that use the 3-scale multi-scale RoPE layout produced when both
`input_img_ids` (a list of 3 tensors) and the model's own target-latent ids are present --
i.e. exactly the `infer_mice_text_position.py` setup. See `transformer_flux.py`'s
`len(img_ids) == 6` branch and `attention_processor.py`'s `len(image_rotary_emb) == 3` branch
for the underlying layout this relies on.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0


class AttentionMapRecorder:
    """Running average, over every attention block and every denoising step, of how much
    each instance's source-phrase tokens attend to the target-latent tokens and to the
    conditioning-image tokens. Each instance's multiple tokens are averaged together before
    accumulating, per the request to average within an instance's own tokens."""

    def __init__(self, token_indices):
        self.token_indices = token_indices
        self.current_timestep = -1
        self._latent_sum = None
        self._image_sum = None
        self._count = 0

    def start_timestep(self, module, inputs):
        self.current_timestep += 1

    def record(self, latent_probs, image_probs):
        """latent_probs/image_probs: (512, N) attention weight rows for the text tokens,
        already averaged over heads, for one attention block at the current timestep."""
        if self._latent_sum is None:
            self._latent_sum = torch.zeros(len(self.token_indices), latent_probs.shape[-1], dtype=torch.float64)
            self._image_sum = torch.zeros(len(self.token_indices), image_probs.shape[-1], dtype=torch.float64)
        for i, indices in enumerate(self.token_indices):
            self._latent_sum[i] += latent_probs[indices].mean(dim=0).double().cpu()
            self._image_sum[i] += image_probs[indices].mean(dim=0).double().cpu()
        self._count += 1

    def finalize(self):
        if self._count == 0:
            raise RuntimeError('No attention was recorded -- did the pipeline actually run?')
        return (self._latent_sum / self._count).float().numpy(), (self._image_sum / self._count).float().numpy()


class FluxAttnProcessorWithMaps(FluxAttnProcessor2_0):
    """Identical to `FluxAttnProcessor2_0` (same rotary application, same fused SDPA call for
    the real model output) but additionally does one extra unfused QK^T + softmax for the 512
    text rows against the full key sequence, purely as a side channel for `recorder`."""

    def __init__(self, recorder: AttentionMapRecorder):
        super().__init__()
        self.recorder = recorder

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        if not (isinstance(image_rotary_emb, list) and len(image_rotary_emb) == 3):
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        from diffusers.models.embeddings import apply_rotary_emb, apply_rotary_emb_headwise

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        part1_len = image_rotary_emb[1][0].shape[0]
        part2_len = image_rotary_emb[2][0].shape[0]

        query_first, query_second, query_third = torch.split(query, [512, part1_len, part2_len], dim=2)
        key_first, key_second, key_third = torch.split(key, [512, part1_len, part2_len], dim=2)

        query_first = apply_rotary_emb(query_first, image_rotary_emb[0])
        key_first = apply_rotary_emb(key_first, image_rotary_emb[0])
        query_second = apply_rotary_emb_headwise(query_second, image_rotary_emb[1])
        key_second = apply_rotary_emb_headwise(key_second, image_rotary_emb[1])
        query_third = apply_rotary_emb_headwise(query_third, image_rotary_emb[2])
        key_third = apply_rotary_emb_headwise(key_third, image_rotary_emb[2])

        query = torch.cat([query_first, query_second, query_third], dim=2)
        key = torch.cat([key_first, key_second, key_third], dim=2)

        assert query.shape[0] == 1, 'Attention capture only supports batch size 1.'
        with torch.no_grad():
            scale = head_dim ** -0.5
            scores = torch.matmul(query_first.float(), key.float().transpose(-2, -1)) * scale
            probs = scores.softmax(dim=-1)[0].mean(dim=0)  # (512, total_len), averaged over heads
            latent_probs = probs[:, 512:512 + part1_len]
            image_probs = probs[:, 512 + part1_len:512 + part1_len + part2_len]
            self.recorder.record(latent_probs, image_probs)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def enable_attention_capture(transformer, token_indices):
    """Swap every attention block's processor for a capturing one and return the recorder."""
    recorder = AttentionMapRecorder(token_indices)
    transformer.register_forward_pre_hook(recorder.start_timestep)
    processors = {name: FluxAttnProcessorWithMaps(recorder) for name in transformer.attn_processors}
    transformer.set_attn_processor(processors)
    return recorder


def _render_heatmap(map_1d, coarse_height, coarse_width, centroid_rc, base_image_bgr, out_path):
    heat = map_1d.reshape(coarse_height, coarse_width).astype(np.float32)
    heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-8)
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_color = cv2.resize(
        heat_color, (base_image_bgr.shape[1], base_image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    overlay = cv2.addWeighted(base_image_bgr, 0.5, heat_color, 0.5, 0)

    row, col = centroid_rc
    x = int(col / coarse_width * base_image_bgr.shape[1])
    y = int(row / coarse_height * base_image_bgr.shape[0])
    radius = max(base_image_bgr.shape[1] // 80, 4)
    cv2.circle(overlay, (x, y), radius, (0, 0, 255), thickness=2)
    cv2.circle(overlay, (x, y), 2, (0, 0, 255), thickness=-1)
    cv2.imwrite(str(out_path), overlay)


def save_instance_attention_maps(
    recorder, centroid_ids, source_phrases, coarse_height, coarse_width,
    input_image_rgb, generated_image_rgb, output_dir,
):
    """Write, per instance, a text->conditioning-image heatmap (over the input photo) and a
    text->target-latent heatmap (over the generated result), each with a red circle at that
    instance's mask centroid so you can see whether attention actually lands there."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_maps, image_maps = recorder.finalize()

    input_bgr = cv2.cvtColor(np.asarray(input_image_rgb), cv2.COLOR_RGB2BGR)
    generated_bgr = cv2.cvtColor(np.asarray(generated_image_rgb), cv2.COLOR_RGB2BGR)

    for i, phrase in enumerate(source_phrases):
        centroid_rc = (centroid_ids[i][1], centroid_ids[i][2])
        safe_name = phrase.replace(' ', '_')

        _render_heatmap(
            image_maps[i], coarse_height, coarse_width, centroid_rc, input_bgr,
            output_dir / f'{i:02d}_{safe_name}_text_to_image.png',
        )
        _render_heatmap(
            latent_maps[i], coarse_height, coarse_width, centroid_rc, generated_bgr,
            output_dir / f'{i:02d}_{safe_name}_text_to_latent.png',
        )
        np.save(output_dir / f'{i:02d}_{safe_name}_text_to_image.npy', image_maps[i].reshape(coarse_height, coarse_width))
        np.save(output_dir / f'{i:02d}_{safe_name}_text_to_latent.npy', latent_maps[i].reshape(coarse_height, coarse_width))
