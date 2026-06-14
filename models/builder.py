from .cfif_gf import CFIFGF
from .wavlm_att import WavLMAtt


MODEL_REGISTRY = {
    "WavLM_Att": WavLMAtt,
    "CFIF-GF": CFIFGF,
}


def build_model(config):
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {supported}")
    return MODEL_REGISTRY[model_name](model_cfg)

