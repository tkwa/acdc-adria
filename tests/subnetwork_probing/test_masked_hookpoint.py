import torch
from acdc.docstring.utils import get_all_docstring_things
from subnetwork_probing.train import do_random_resample_caching
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformer import HookedTransformer
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformerConfig import HookedTransformerConfig

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

    # Cache the un-patched data in each MaskedHookPoint
    _ = do_random_resample_caching(legacy_model, all_task_things.validation_patch_data)

    model = all_task_things.tl_model

    rng_state = torch.get_rng_state()
    with torch.no_grad():
        torch.set_rng_state(rng_state)
        out_legacy = legacy_model(all_task_things.validation_data)

        torch.set_rng_state(rng_state)
        out_hooks = model(all_task_things.validation_data)

    assert torch.allclose(out_legacy, out_hooks)
