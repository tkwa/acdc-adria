# %% 
import torch
from tests.subnetwork_probing.test_masked_hookpoint import test_cache_writeable_forward_pass
from acdc.docstring.utils import get_all_docstring_things
from subnetwork_probing.train import MaskedTransformer

# %%

test_cache_writeable_forward_pass()
# %%
all_task_things = get_all_docstring_things(
    num_examples=6,
    seq_len=41,
    device=torch.device("cpu"),
    metric_name="kl_div",
    correct_incorrect_wandb=False,
)
masked_model = MaskedTransformer(all_task_things.tl_model)
# %%
masked_model.do_zero_caching()
rng_state = torch.random.get_rng_state()
masked_model.do_random_resample_caching(all_task_things.validation_patch_data)
context_args = dict(ablation='resample', ablation_data=all_task_things.validation_patch_data)
with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
    out1 = hooked_model(all_task_things.validation_data)
# %%
