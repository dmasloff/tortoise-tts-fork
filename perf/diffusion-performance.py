import torch
from datetime import datetime
from tortoise.models.diffusion_decoder import DiffusionTts
from tortoise.api import get_model_path, do_spectrogram_diffusion, load_discrete_vocoder_diffuser

DEVICE = torch.device('cuda:5')
MODEL_FILE_NAME = 'diffusion_decoder.pth'

for ATTENTION_BACKBONE in ['legacy', 'modern', 'legacy/cache', 'modern/cache']:

    diffusion_kwargs = {
        'model_channels': 1024,
        'num_layers': 10,
        'in_channels': 100,
        'out_channels': 200,
        'in_latent_channels': 1024,
        'in_tokens': 8193,
        'dropout': 0,
        'use_fp16': False,
        'num_heads': 16,
        'layer_drop': 0,
        'unconditioned_percentage': 0,
        'attention_backbone': ATTENTION_BACKBONE,
    }

    load_discrete_vocoder_diffuser_kwargs = {
        'trained_diffusion_steps': 2000,
        'desired_diffusion_steps': 200,
        'cond_free': True,
        'cond_free_k': 2.0,
    }

    diffuser = load_discrete_vocoder_diffuser(**load_discrete_vocoder_diffuser_kwargs)

    diffusion_model = DiffusionTts(**diffusion_kwargs).cpu().eval()
    diffusion_model.load_state_dict(torch.load(get_model_path(MODEL_FILE_NAME)))

    latents = torch.load('latents.pt', map_location=torch.device('cpu'))
    conditioning_latents = torch.load('conditioning_latents.pt', map_location=torch.device('cpu'))

    do_spectrogram_diffusion_kwargs = {
        'diffusion_model': diffusion_model.to(DEVICE),
        'diffuser': diffuser,
        'latents': latents.to(DEVICE),
        'conditioning_latents': conditioning_latents.to(DEVICE),
        'temperature': 1.0,
        'verbose': True,
    }


    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        do_spectrogram_diffusion(**do_spectrogram_diffusion_kwargs)

    prof.export_chrome_trace(f'./profiler-output/{ATTENTION_BACKBONE[:6]}_{ATTENTION_BACKBONE[7:]}_16_11_2025.json')
