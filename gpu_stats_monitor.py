# Copyright The PyTorch Lightning team.
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
"""
Device Stats Monitor
====================

Monitors and logs device stats during training.

"""

# TAKEN FROM PYTORCH LIGHTNING ADAPTED BY CORNELIUS EMDE

import logging
import os
import shutil
import subprocess
from typing import Any, Dict, Optional, Union, List

import torch

_log = logging.getLogger(__name__)

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT


class GPUStatsMonitor(Callback):
    r"""
    Automatically monitors and logs device stats during training stage. ``DeviceStatsMonitor``
    is a special callback as it requires a ``logger`` to passed as argument to the ``Trainer``.

    Raises:
        MisconfigurationException:
            If ``Trainer`` has no logger.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import DeviceStatsMonitor
        >>> device_stats = DeviceStatsMonitor() # doctest: +SKIP
        >>> trainer = Trainer(callbacks=[device_stats]) # doctest: +SKIP
    """

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if not trainer.logger:
            raise MisconfigurationException("Cannot use DeviceStatsMonitor callback with Trainer that has no logger.")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if not trainer.logger_connector.should_update_logs:
            return

        device_stats = self.get_device_stats(pl_module.device)
        prefixed_device_stats = prefix_metrics_keys(device_stats, f"GPU_{pl_module.device.index}")
        assert trainer.logger is not None
        trainer.logger.log_metrics(prefixed_device_stats, step=trainer.global_step)

    @staticmethod
    def get_device_stats(device: Union[str, torch.device]) -> Dict[str, Any]:
        """Get GPU stats including memory, fan speed, and temperature from nvidia-smi.

        Args:
            device: GPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.

        Raises:
            FileNotFoundError:
                If nvidia-smi installation not found
        """
        gpu_stat_metrics = [
            ("utilization.gpu", "%"),
            ("memory.used", "MB"),
            ("memory.free", "MB"),
            ("utilization.memory", "%"),
            ("fan.speed", "%"),
            ("temperature.gpu", "°C"),
            ("temperature.memory", "°C"),
        ]
        gpu_stat_keys = [k for k, _ in gpu_stat_metrics]
        gpu_query = ",".join(gpu_stat_keys)

        gpu_id = _get_gpu_id(device.index)
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path is None:
            raise FileNotFoundError("nvidia-smi: command not found")
        result = subprocess.run(
            [nvidia_smi_path, f"--query-gpu={gpu_query}", "--format=csv,nounits,noheader", f"--id={gpu_id}"],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True,
        )

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except ValueError:
                return 0.0

        s = result.stdout.strip()
        stats = [_to_float(x) for x in s.split(", ")]

        gpu_stats = {}
        for i, (x, unit) in enumerate(gpu_stat_metrics):
            gpu_stats[f"{x} ({unit})"] = stats[i]
        return gpu_stats

    # @staticmethod
    # def auto_device_count() -> int:
    #     """Get the devices when set to auto."""
    #     return torch.cuda.device_count()


def _get_gpu_id(device_id: int) -> str:
    """Get the unmasked real GPU IDs."""
    # All devices if `CUDA_VISIBLE_DEVICES` unset
    default = ",".join(str(i) for i in range(torch.cuda.device_count()))
    cuda_visible_devices: List[str] = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
    return cuda_visible_devices[device_id].strip()


def prefix_metrics_keys(metrics_dict: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {prefix + "/" + k: v for k, v in metrics_dict.items()}

