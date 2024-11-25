# ruff: noqa

# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver

from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
    betas_for_alpha_bar,
    rescale_zero_terminal_snr,
)
from diffusers.utils.deprecation_utils import deprecate
from diffusers.utils.import_utils import is_scipy_available


class PachedDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        solver_order: int = 2,
        prediction_type: str = 'epsilon',
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = 'dpmsolver++',
        solver_type: str = 'midpoint',
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        use_lu_lambdas: Optional[bool] = False,
        final_sigmas_type: Optional[str] = 'zero',  # "zero", "sigma_min"
        lambda_min_clipped: float = -float('inf'),
        variance_type: Optional[str] = None,
        timestep_spacing: str = 'linspace',
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError('Make sure to install scipy if you want to use beta sigmas.')
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                'Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used.'
            )
        if algorithm_type in ['dpmsolver', 'sde-dpmsolver']:
            deprecation_message = f'algorithm_type {algorithm_type} is deprecated and will be removed in a future version. Choose from `dpmsolver++` or `sde-dpmsolver++` instead'
            deprecate('algorithm_types dpmsolver and sde-dpmsolver', '1.0.0', deprecation_message)

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

        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # settings for DPM-Solver
        if algorithm_type not in ['dpmsolver', 'dpmsolver++', 'sde-dpmsolver', 'sde-dpmsolver++']:
            if algorithm_type == 'deis':
                self.register_to_config(algorithm_type='dpmsolver++')
            else:
                raise NotImplementedError(f'{algorithm_type} is not implemented for {self.__class__}')

        if solver_type not in ['midpoint', 'heun']:
            if solver_type in ['logrho', 'bh1', 'bh2']:
                self.register_to_config(solver_type='midpoint')
            else:
                raise NotImplementedError(f'{solver_type} is not implemented for {self.__class__}')

        if algorithm_type not in ['dpmsolver++', 'sde-dpmsolver++'] and final_sigmas_type == 'zero':
            raise ValueError(
                f'`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}. Please choose `sigma_min` instead.'
            )

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to('cpu')  # to avoid too much CPU/GPU communication

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary timesteps schedule. If `None`, timesteps will be generated
                based on the `timestep_spacing` attribute. If `timesteps` is passed, `num_inference_steps` and `sigmas`
                must be `None`, and `timestep_spacing` attribute will be ignored.
        """
        if num_inference_steps is None and timesteps is None:
            raise ValueError('Must pass exactly one of `num_inference_steps` or `timesteps`.')
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError('Can only pass one of `num_inference_steps` or `custom_timesteps`.')
        if timesteps is not None and self.config.use_karras_sigmas:
            raise ValueError('Cannot use `timesteps` with `config.use_karras_sigmas = True`')
        if timesteps is not None and self.config.use_lu_lambdas:
            raise ValueError('Cannot use `timesteps` with `config.use_lu_lambdas = True`')
        if timesteps is not None and self.config.use_exponential_sigmas:
            raise ValueError('Cannot set `timesteps` with `config.use_exponential_sigmas = True`.')
        if timesteps is not None and self.config.use_beta_sigmas:
            raise ValueError('Cannot set `timesteps` with `config.use_beta_sigmas = True`.')

        if timesteps is not None:
            timesteps = np.array(timesteps).astype(np.int64)
        else:
            # Clipping the minimum of all lambda(t) for numerical stability.
            # This is critical for cosine (squaredcos_cap_v2) noise schedule.
            clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.config.lambda_min_clipped)

            # PATCHED --------------------------------------------------------
            # last_timestep = ((self.config.num_train_timesteps - clipped_idx).numpy()).item()
            # ----------------------------------------------------------------
            last_timestep = ((self.config.num_train_timesteps - clipped_idx).cpu().numpy()).item()
            # ----------------------------------------------------------------

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == 'linspace':
                timesteps = (
                    np.linspace(0, last_timestep - 1, num_inference_steps + 1)
                    .round()[::-1][:-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.config.timestep_spacing == 'leading':
                step_ratio = last_timestep // (num_inference_steps + 1)
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (
                    (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
                )
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == 'trailing':
                step_ratio = self.config.num_train_timesteps / num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.config.use_karras_sigmas:
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
        elif self.config.use_lu_lambdas:
            lambdas = np.flip(log_sigmas.copy())
            lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
            sigmas = np.exp(lambdas)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.config.final_sigmas_type == 'sigma_min':
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == 'zero':
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to('cpu')  # to avoid too much CPU/GPU communication


from diffusers.schedulers import scheduling_dpmsolver_multistep

scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler = PachedDPMSolverMultistepScheduler
