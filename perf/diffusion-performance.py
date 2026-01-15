import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tortoise.models.diffusion_decoder import DiffusionTts
from tortoise.api import get_model_path, do_spectrogram_diffusion, load_discrete_vocoder_diffuser

from dataclasses import dataclass
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


@dataclass
class PerformanceTestConfig:
    config_name: str
    attention_backbone: str
    sdp_backend: SDPBackend | None


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DATE = str(datetime.now().date()).replace('-', '_')
TEST_LENGTH = 128
LOAD_CHECKPOINTS = False
LOG_DIRECTORY = f'profiler_output__{DATE}'


PERFORMANCE_CONFIGS = [
    # legacy
    PerformanceTestConfig(config_name='legacy', attention_backbone='legacy', sdp_backend=None),
    PerformanceTestConfig(config_name='legacy_cache', attention_backbone='legacy/cache', sdp_backend=None),
    # modern
    PerformanceTestConfig(config_name='modern', attention_backbone='modern', sdp_backend=None),
    PerformanceTestConfig(config_name='modern__math', attention_backbone='modern', sdp_backend=SDPBackend.MATH),
    PerformanceTestConfig(config_name='modern__efficient', attention_backbone='modern', sdp_backend=SDPBackend.EFFICIENT_ATTENTION),
    #modern cached
    PerformanceTestConfig(config_name='modern_cache', attention_backbone='modern/cache', sdp_backend=None),
    PerformanceTestConfig(config_name='modern_cache__math', attention_backbone='modern/cache', sdp_backend=SDPBackend.MATH),
    PerformanceTestConfig(config_name='modern_cache__efficient', attention_backbone='modern/cache', sdp_backend=SDPBackend.EFFICIENT_ATTENTION),
    # rope
    PerformanceTestConfig(config_name='rope', attention_backbone='rope', sdp_backend=None),
    PerformanceTestConfig(config_name='rope__math', attention_backbone='rope', sdp_backend=SDPBackend.MATH),
    PerformanceTestConfig(config_name='rope__efficiient', attention_backbone='rope', sdp_backend=SDPBackend.EFFICIENT_ATTENTION),
    PerformanceTestConfig(config_name='rope__cudnn', attention_backbone='rope', sdp_backend=SDPBackend.CUDNN_ATTENTION),
    PerformanceTestConfig(config_name='rope__flash', attention_backbone='rope', sdp_backend=SDPBackend.FLASH_ATTENTION),
]


os.makedirs(f'./{LOG_DIRECTORY}', mode=0o777, exist_ok=True)


for config in PERFORMANCE_CONFIGS:
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
        'attention_backbone': config.attention_backbone,
        'use_optimized_backend': True if config.sdp_backend in [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION] else False,
        'use_contigious_tensors': True if config.sdp_backend is not None else False
    }

    load_discrete_vocoder_diffuser_kwargs = {
        'trained_diffusion_steps': 2000,
        'desired_diffusion_steps': 200,
        'cond_free': True,
        'cond_free_k': 2.0,
    }

    diffuser = load_discrete_vocoder_diffuser(**load_discrete_vocoder_diffuser_kwargs)
    diffusion_model = DiffusionTts(**diffusion_kwargs).cpu().eval()

    if LOAD_CHECKPOINTS:
        diffusion_model.load_state_dict(torch.load(get_model_path(MODEL_FILE_NAME)))
        latents = torch.load('latents.pt', map_location=torch.device('cpu'))
        conditioning_latents = torch.load('conditioning_latents.pt', map_location=torch.device('cpu'))
    else:
        latents = torch.randn((1, TEST_LENGTH, 1024))
        conditioning_latents = torch.randn((1, 2048))


    do_spectrogram_diffusion_kwargs = {
        'diffusion_model': diffusion_model.to(DEVICE),
        'diffuser': diffuser,
        'latents': latents.to(DEVICE),
        'conditioning_latents': conditioning_latents.to(DEVICE),
        'temperature': 1.0,
        'verbose': False,
    }

    if config.sdp_backend is None:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(3):
                torch.cuda.synchronize()
                do_spectrogram_diffusion(**do_spectrogram_diffusion_kwargs)
                torch.cuda.synchronize()

                prof.step()
    else:
        with sdpa_kernel(config.sdp_backend):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for _ in range(3):
                    torch.cuda.synchronize()
                    do_spectrogram_diffusion(**do_spectrogram_diffusion_kwargs)
                    torch.cuda.synchronize()

                    prof.step()

    prof.export_chrome_trace(f'./{LOG_DIRECTORY}/{config.config_name}.json')
