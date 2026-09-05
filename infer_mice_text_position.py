#!/usr/bin/env python3
"""Run one MICE-Bench case with instance-centroid text RoPE coordinates."""

import argparse
import json
import shlex
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from moge.model import import_model_class_by_version


ROOT = Path(__file__).resolve().parent
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1280),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1280, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]


def split_into_2x2_local_grids(x: torch.Tensor) -> torch.Tensor:
    height, width, channels = x.shape
    if height % 2 or width % 2:
        raise ValueError(f'Multi-scale grid must be even, received {height}x{width}.')
    return x.view(height // 2, 2, width // 2, 2, channels).permute(0, 2, 1, 3, 4).reshape(
        height // 2, width // 2, 4 * channels
    )


def build_instruction(source_phrases, target_phrases):
    """Return simple edit text and character spans for every source phrase."""
    clauses, spans = [], []
    cursor = 0
    for source, target in zip(source_phrases, target_phrases):
        clause = f'change the {source} into {target}.'
        source_start = cursor + len('change the ')
        spans.append((source_start, source_start + len(source)))
        clauses.append(clause)
        cursor += len(clause) + 1
    return ' '.join(clauses), spans


def make_text_ids(tokenizer, prompt, source_spans, centroid_ids, dtype, device):
    """Give each source-object T5 token its matching [z, y, x] RoPE ID."""
    encoded = tokenizer(
        prompt,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors='pt',
    )
    offsets = encoded.offset_mapping[0].tolist()
    text_ids = torch.zeros((len(offsets), 3), dtype=dtype, device=device)
    matched = []
    for (start, end), centroid in zip(source_spans, centroid_ids):
        token_indices = [
            index for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        if not token_indices:
            raise ValueError(f'Could not map source phrase at character span [{start}, {end}) to T5 tokens.')
        text_ids[token_indices] = torch.as_tensor(centroid, dtype=dtype, device=device)
        matched.append(token_indices)
    return text_ids, matched


def prepare_image_ids_and_centroids(depth, masks, intrinsics, out_height, out_width, bind_mask_to_text_position=False):
    """Create unwarped image IDs and physical mask-centroid IDs in the same RoPE frame.

    If `bind_mask_to_text_position`, every image token that falls inside an instance mask is
    given that instance's centroid RoPE id (the same id injected into its source-phrase text
    tokens) instead of its natural depth-derived position, at every grid scale. Later masks in
    the list win over earlier ones on overlap.
    """
    height, width = depth.shape
    fx, fy = intrinsics[0, 0] * width, intrinsics[1, 1] * height
    cx, cy = width / 2, height / 2
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size == 0:
        raise ValueError('MoGe returned no finite depth values.')
    depth = np.nan_to_num(depth, nan=float(finite_depth.max()), posinf=float(finite_depth.max()), neginf=float(finite_depth.min()))
    yy, xx = np.mgrid[:height, :width]
    x_cam = (xx - cx) * depth / fx
    y_cam = (yy - cy) * depth / fy
    pointmap = np.stack((x_cam, y_cam, depth), axis=-1)

    z_min, z_max = depth.min(), depth.max()
    z_norm = (depth - z_min) / max(z_max - z_min, 1e-6) + 1.0
    coords = np.stack((z_norm, yy, xx), axis=-1)

    coarse_height, coarse_width = out_height // 16, out_width // 16
    centroid_ids = []
    for mask in masks:
        valid = mask & np.isfinite(pointmap).all(axis=-1)
        if not valid.any():
            raise ValueError('An instance mask contains no valid 3D points.')
        centroid_3d = pointmap[valid].mean(axis=0)
        u = fx * centroid_3d[0] / centroid_3d[2] + cx
        v = fy * centroid_3d[1] / centroid_3d[2] + cy
        centroid_ids.append([
            (centroid_3d[2] - z_min) / max(z_max - z_min, 1e-6) + 1.0,
            v / height * coarse_height,
            u / width * coarse_width,
        ])

    coords = torch.from_numpy(coords).float().permute(2, 0, 1).unsqueeze(0)
    image_ids, grid_level = [], 0
    for grid_height, grid_width in ((out_height // 16, out_width // 16), (out_height // 8, out_width // 8), (out_height // 4, out_width // 4)):
        grid = F.interpolate(coords, size=(grid_height, grid_width), mode='bilinear', align_corners=False)
        grid = grid.squeeze(0).permute(1, 2, 0)
        grid[..., 1] = grid[..., 1] / height * coarse_height
        grid[..., 2] = grid[..., 2] / width * coarse_width
        if bind_mask_to_text_position:
            for mask, centroid in zip(masks, centroid_ids):
                mask_small = cv2.resize(
                    mask.astype(np.uint8), (grid_width, grid_height), interpolation=cv2.INTER_NEAREST
                ) > 0
                grid[torch.from_numpy(mask_small)] = torch.as_tensor(centroid, dtype=grid.dtype)
        if grid_level:
            for _ in range(grid_level):
                grid = split_into_2x2_local_grids(grid)
        image_ids.append(grid.reshape(-1, grid.shape[-1]))
        grid_level += 1
    return image_ids, centroid_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=int, required=True, help='LoMOE case number, e.g. 0 for key "00".')
    parser.add_argument('--bench_dir', type=Path, default=ROOT / 'mice_bench')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'outputs' / 'mice_text_position')
    parser.add_argument('--moge_checkpoint_path', type=Path, default=ROOT / 'moge-2-vitl-normal' / 'model.pt')
    parser.add_argument('--transformer_checkpoint_path', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--flux_kontext_path', type=Path, default=ROOT / 'FLUX.1-Kontext-dev')
    parser.add_argument(
        '--device_map', choices=('none', 'balanced'), default='none',
        help='Use `balanced` to shard the PE-Field transformer across all visible GPUs (requires accelerate >= 0.28).',
    )
    parser.add_argument(
        '--max_gpu_memory', type=str, default='28GiB',
        help='Per-GPU allocation limit used with --device_map balanced. Leave headroom for MoGe and activations.',
    )
    parser.add_argument(
        '--no_text_position_encoding', action='store_true',
        help='Ablation: skip centroid RoPE injection for text tokens and just pass the plain concatenated prompt.',
    )
    parser.add_argument(
        '--bind_mask_to_text_position', action='store_true',
        help='Ablation: override image tokens inside each instance mask with that instance\'s text centroid RoPE id.',
    )
    parser.add_argument(
        '--capture_attention', action='store_true',
        help='Record text->target-latent and text->conditioning-image attention, averaged per instance over its '
             'source-phrase tokens and over every block/timestep, and save heatmaps with the mask centroid marked.',
    )
    args = parser.parse_args()

    with (args.bench_dir / 'LoMOE.json').open() as file:
        record = json.load(file).get(f'{args.case:02d}')
    if record is None:
        raise KeyError(f'No LoMOE case {args.case:02d}.')
    mask_paths = [args.bench_dir / path for path in shlex.split(record['mask_path'])]
    source_phrases, target_phrases = shlex.split(record['source_prompt']), shlex.split(record['fg_prompt'])
    if not (len(mask_paths) == len(source_phrases) == len(target_phrases)):
        raise ValueError('Mask, source-prompt, and target-prompt counts must match.')
    prompt, source_spans = build_instruction(source_phrases, target_phrases)

    image_path = args.bench_dir / record['image_path']
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_np.shape[:2]
    masks = []
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(mask > 0)

    aspect_ratio = width / height
    _, output_width, output_height = min(
        (abs(aspect_ratio - candidate_width / candidate_height), candidate_width, candidate_height)
        for candidate_width, candidate_height in PREFERRED_KONTEXT_RESOLUTIONS
    )
    device = 'cuda'
    moge = import_model_class_by_version('v2').from_pretrained(args.moge_checkpoint_path).to(device).eval()
    moge_input = torch.tensor(image_np / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.inference_mode():
        moge_output = moge.infer(moge_input, resolution_level=9, use_fp16=False)
    depth = moge_output['depth'].cpu().numpy().squeeze()
    intrinsics = moge_output['intrinsics'].cpu().numpy()
    image_ids, centroid_ids = prepare_image_ids_and_centroids(
        depth, masks, intrinsics, output_height, output_width,
        bind_mask_to_text_position=args.bind_mask_to_text_position,
    )

    if args.device_map == 'balanced':
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                '--device_map balanced requires at least two visible CUDA GPUs. '
                'Use CUDA_VISIBLE_DEVICES to select them, or use --device_map none.'
            )

        # Pipeline-level `device_map="balanced"` only assigns whole components and
        # therefore puts the entire 23.8 GB transformer on one GPU.  Shard the
        # transformer itself instead, which distributes its DiT blocks over both
        # visible GPUs. Keep the text encoders and VAE on GPU 0: placing T5-XXL on
        # GPU 1 alongside its transformer blocks leaves too little room for the
        # large attention workspace used during denoising.
        max_memory = {gpu_id: args.max_gpu_memory for gpu_id in range(torch.cuda.device_count())}
        transformer = FluxTransformer2DModel.from_pretrained(
            args.transformer_checkpoint_path,
            subfolder='transformer',
            torch_dtype=torch.bfloat16,
            device_map='balanced',
            max_memory=max_memory,
        )
        pipe = FluxKontextPipeline.from_pretrained(
            args.flux_kontext_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        auxiliary_device = 'cuda:0'
        pipe.vae.to(auxiliary_device)
        pipe.text_encoder.to(auxiliary_device)
        pipe.text_encoder_2.to(auxiliary_device)
        if pipe.image_encoder is not None:
            pipe.image_encoder.to(auxiliary_device)
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.transformer_checkpoint_path, subfolder='transformer', torch_dtype=torch.bfloat16
        )
        pipe = FluxKontextPipeline.from_pretrained(
            args.flux_kontext_path, transformer=transformer, torch_dtype=torch.bfloat16
        ).to(device)
    pipe.set_progress_bar_config(disable=True)
    text_ids, token_indices = make_text_ids(
        pipe.tokenizer_2, prompt, source_spans, centroid_ids, pipe.text_encoder.dtype, pipe._execution_device
    )

    recorder = None
    if args.capture_attention:
        from attention_capture import enable_attention_capture
        recorder = enable_attention_capture(pipe.transformer, token_indices)

    input_image = torch.from_numpy(image_np.copy()).permute(2, 0, 1).unsqueeze(0).float()
    input_image = F.interpolate(input_image, size=(output_height, output_width), mode='bilinear', align_corners=False)
    input_image = input_image / 127.5 - 1.0
    pipe_kwargs = dict(
        image=input_image,
        height=output_height,
        width=output_width,
        prompt=prompt,
        input_img_ids=image_ids,
        use_multi_scale_position=True,
    )
    if not args.no_text_position_encoding:
        pipe_kwargs['input_text_ids'] = text_ids
    result = pipe(**pipe_kwargs).images[0]

    output_dir = args.output_dir / f'{args.case:02d}'
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir / 'output.png')

    if recorder is not None:
        from attention_capture import save_instance_attention_maps
        display_input = cv2.resize(image_np, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        save_instance_attention_maps(
            recorder, centroid_ids, source_phrases,
            coarse_height=output_height // 16, coarse_width=output_width // 16,
            input_image_rgb=display_input, generated_image_rgb=result,
            output_dir=output_dir / 'attention_maps',
        )
    with (output_dir / 'metadata.json').open('w') as file:
        json.dump({
            'case': args.case, 'prompt': prompt, 'source_phrases': source_phrases,
            'target_phrases': target_phrases, 'centroid_ids': centroid_ids,
            'source_token_indices': token_indices,
            'text_position_encoding': not args.no_text_position_encoding,
            'bind_mask_to_text_position': args.bind_mask_to_text_position,
        }, file, indent=2)
    print(f'Saved {output_dir / "output.png"}')


if __name__ == '__main__':
    main()
