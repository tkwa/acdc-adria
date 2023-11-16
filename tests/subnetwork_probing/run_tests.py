# %% 
import torch
from tests.subnetwork_probing.test_masked_hookpoint import test_cache_writeable_forward_pass
from acdc.tracr_task.utils import get_all_tracr_things
from acdc.docstring.utils import get_all_docstring_things
from subnetwork_probing.train import MaskedTransformer


# %%

# all_task_things = get_all_docstring_things(
#     num_examples=6,
#     seq_len=41,
#     device=torch.device("cpu"),
#     metric_name="kl_div",
#     correct_incorrect_wandb=False,
# )
# %%

test_cache_writeable_forward_pass()
# %%
all_task_things = get_all_tracr_things(
    task="reverse", metric_name="l2", num_examples=6, device=torch.device("cpu")
)
masked_model = MaskedTransformer(all_task_things.tl_model)
# %%
masked_model.do_zero_caching()
# rng_state = torch.random.get_rng_state()
masked_model.do_random_resample_caching(all_task_things.validation_patch_data)
context_args = dict(ablation='resample', ablation_data=all_task_things.validation_patch_data)
with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
    out1 = hooked_model(all_task_things.validation_data)

print(masked_model.forward_cache.cache_dict.keys())
# %%
# %%

# Test that the hooked model without ablations is the same as the unhooked model
def test_f_cache_implementation():
    """
    Verifies that the network running on forward cache with nothing resampled is
    identical to the unhooked network
    """
    global all_task_things
    global masked_model
    all_task_things = get_all_tracr_things(
    task="reverse", metric_name="l2", num_examples=6, device=torch.device("cpu")
)
    masked_model = MaskedTransformer(all_task_things.tl_model, no_ablate=True)
    context_args = dict(ablation='resample', ablation_data=all_task_things.validation_patch_data)
    out1 = masked_model.model(all_task_things.validation_data)
    with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
        out2 = hooked_model(all_task_things.validation_data)

    print(out1, out2)
    assert torch.allclose(out1, out2)
test_f_cache_implementation()
# %%