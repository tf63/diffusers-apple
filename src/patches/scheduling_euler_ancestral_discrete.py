# ruff: noqa

# Copyright 2024 Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
    betas_for_alpha_bar,
    rescale_zero_terminal_snr,
)


class PachedEulerAncestralDiscreteScheduler(EulerAncestralDiscreteScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = 'epsilon',
        timestep_spacing: str = 'linspace',
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == 'scaled_linear':
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == 'squaredcos_cap_v2':
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f'{beta_schedule} is not implemented for {self.__class__}')

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        # PATCHED --------------------------------------------------------
        # self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # ----------------------------------------------------------------
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).cpu()
        # ----------------------------------------------------------------

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            # FP16 smallest positive subnormal works well here
            self.alphas_cumprod[-1] = 2**-24

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.is_scale_input_called = False

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to('cpu')  # to avoid too much CPU/GPU communication


from diffusers.schedulers import scheduling_euler_ancestral_discrete

scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler = PachedEulerAncestralDiscreteScheduler
