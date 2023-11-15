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
