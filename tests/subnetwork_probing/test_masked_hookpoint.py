import functools
import torch
from typing import Iterable, Tuple
from acdc.docstring.utils import get_all_docstring_things
from subnetwork_probing.train import do_random_resample_caching, MaskedTransformer
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformer import HookedTransformer
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from transformer_lens.hook_points import HookPoint
from transformer_lens.ActivationCache import ActivationCache


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
    legacy_model.load_state_dict(all_task_things.tl_model.state_dict(), strict=False)

    model = MaskedTransformer(all_task_things.tl_model)

    # Cache the un-patched data in each MaskedHookPoint
    _ = do_random_resample_caching(legacy_model, all_task_things.validation_patch_data)
    # Cache all the activations for the new model
    model.do_random_resample_caching(all_task_things.validation_patch_data)

    rng_state = torch.get_rng_state()
    with torch.no_grad():
        torch.set_rng_state(rng_state)
        out_legacy = legacy_model(all_task_things.validation_data)

        torch.set_rng_state(rng_state)
        with model.with_fwd_hooks() as masked_model:
            out_hooks = masked_model(all_task_things.validation_data)

    assert torch.allclose(out_legacy, out_hooks)
