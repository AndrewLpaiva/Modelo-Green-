import os
import torch
import importlib


# Dynamically import model modules from the `models` package.
# If a model's submodule or symbol fails to import (missing optional deps), we skip it
# so the experiment framework can still run with the available models.
MODEL_NAMES = [
    'TimesNet', 'Autoformer', 'Transformer', 'Nonstationary_Transformer', 'DLinear', 'FEDformer',
    'Informer', 'LightTS', 'Reformer', 'ETSformer', 'Pyraformer', 'PatchTST', 'MICN', 'Crossformer',
    'FiLM', 'iTransformer', 'Koopa', 'TiDE', 'FreTS', 'Corrformer', 'Mamba'
]


def _load_available_models():
    model_dict = {}
    for name in MODEL_NAMES:
        try:
            module = importlib.import_module(f'models.{name}')
            # Register the module object so calling code can access module.Model
            model_dict[name] = module
        except Exception:
            # Skip models that cannot be imported due to optional dependencies
            # (e.g. Mamba requiring mamba_ssm). This avoids failing startup.
            continue
    return model_dict


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # Build model dict from available model modules
        self.model_dict = _load_available_models()
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            # device_count = torch.cuda.device_count()
            # current_device = torch.cuda.current_device()
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
