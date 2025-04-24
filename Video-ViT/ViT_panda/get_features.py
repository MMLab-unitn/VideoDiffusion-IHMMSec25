from video_llama.common.registry import registry
from video_llama.common.config import Config

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class FeatureExtractor:
    def __init__(self, args, device=None):
        cfg = DotDict({"cfg_path":"ViT_panda/eval_configs/panda70M_eval.yaml"})
        self.cfg = Config(cfg)
        if device is None:
            self.device = args.device
        else:
            self.device = device
        self.model_config = self.cfg.model_cfg
        self.model_cls = registry.get_model_class(self.model_config.arch)
        self.model = self.model_cls.from_config(self.model_config).to(self.device)
        self.model.eval()

    def get_features(self, video_batch):
        outputs = self.model.inference(video_batch.to(self.device))
        
        return outputs

def get_extractor(args, device=None):
    return FeatureExtractor(args, device=device)
