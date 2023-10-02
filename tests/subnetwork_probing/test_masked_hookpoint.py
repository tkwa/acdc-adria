import functools
import torch
from typing import Iterable, Tuple
from acdc.docstring.utils import get_all_docstring_things
from subnetwork_probing.train import do_random_resample_caching
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformer import HookedTransformer
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from transformer_lens.hook_points import HookPoint
from transformer_lens.ActivationCache import ActivationCache


def sample_mask(mask_scores, BETA=2/3, GAMMA=-0.1, ZETA=1.1):
    # reparam trick taken from their code
    uniform_sample = (
        torch.zeros_like(mask_scores).uniform_().clamp(0.0001, 0.9999)
    )
    s = torch.sigmoid(
        (uniform_sample.log() - (1 - uniform_sample).log() + mask_scores)
        / BETA
    )
    s_bar = s * (ZETA - GAMMA) + GAMMA
    mask = s_bar.clamp(min=0.0, max=1.0)
    return mask

def mask_this_activation(cache: ActivationCache, mask_scores: torch.Tensor, hook_point_out: torch.Tensor, hook: HookPoint):
    mask = sample_mask(mask_scores)
    out = mask * hook_point_out + (1-mask) * cache[hook.name]
    return out

def names_and_scores_of_all_mlps_and_heads(model) -> Iterable[Tuple[str, torch.Tensor]]:
    for layer_index, layer in enumerate(model.blocks):
        yield (f"blocks.{layer_index}.hook_mlp_out", layer.hook_mlp_out.mask_scores)
        for q_k_v in ["q", "k", "v"]:
            yield (f"blocks.{layer_index}.attn.hook_{q_k_v}", getattr(layer.attn, f"hook_{q_k_v}").mask_scores)

def test_induction_mask_reimplementation_correct():
    all_task_things = get_all_docstring_things(
        num_examples=6,
        seq_len=300,
        device=torch.device("cpu"),
        metric_name="kl_div",
        correct_incorrect_wandb=False,
    )

    kwargs = dict(**all_task_things.tl_model.cfg.__dict__)
    for kwarg_string in [
        "use_split_qkv_input",
        "n_devices",
        "gated_mlp",
        "use_attn_in",
        "use_hook_mlp_in",
    ]:
        if kwarg_string in kwargs:
            del kwargs[kwarg_string]

    cfg = HookedTransformerConfig(**kwargs)
    # Create a model using the old version of SP, which uses MaskedHookPoints and a modified TransformerLens
    legacy_model = HookedTransformer(cfg, is_masked=True)
    model = all_task_things.tl_model

    legacy_model.load_state_dict(model.state_dict(), strict=False)

    # Cache the un-patched data in each MaskedHookPoint
    _ = do_random_resample_caching(legacy_model, all_task_things.validation_patch_data)
    # Cache all the activations for the new model
    _, cache = model.run_with_cache(all_task_things.validation_patch_data, return_cache_object=True)

    for name, mask_scores in names_and_scores_of_all_mlps_and_heads(legacy_model):
        model.add_hook(name, functools.partial(mask_this_activation, cache, mask_scores))

    rng_state = torch.get_rng_state()
    with torch.no_grad():
        torch.set_rng_state(rng_state)
        out_legacy = legacy_model(all_task_things.validation_data)

        torch.set_rng_state(rng_state)
        out_hooks = model(all_task_things.validation_data)

    assert torch.allclose(out_legacy, out_hooks)
