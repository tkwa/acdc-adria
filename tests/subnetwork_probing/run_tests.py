# %% 
import torch
# from tests.subnetwork_probing.test_masked_hookpoint import test_cache_writeable_forward_pass
from acdc.tracr_task.utils import get_all_tracr_things
from acdc.docstring.utils import get_all_docstring_things
from subnetwork_probing.train import MaskedTransformer, train_sp, edge_level_corr, print_stats
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.docstring.utils import AllDataThings, get_all_docstring_things, get_docstring_subgraph_true_edges
from acdc.tracr_task.utils import get_tracr_proportion_edges, get_tracr_reverse_edges
from acdc.greaterthan.utils import get_all_greaterthan_things, get_greaterthan_true_edges
from acdc.induction.utils import get_all_induction_things #, get_induction_true_edges
from acdc.ioi.utils import get_all_ioi_things, get_ioi_true_edges
from acdc.acdc_utils import get_node_stats, get_edge_stats

import argparse
import random
import matplotlib.pyplot as plt

# %%

# all_task_things = get_all_docstring_things(
#     num_examples=6,
#     seq_len=41,
#     device=torch.device("cpu"),
#     metric_name="kl_div",
#     correct_incorrect_wandb=False,
# )
# %%

# test_cache_writeable_forward_pass()
# %%
all_task_things = get_all_tracr_things(
    task="reverse", metric_name="l2", num_examples=6, device=torch.device("cpu")
)
masked_model = MaskedTransformer(all_task_things.tl_model, use_pos_embed=True)
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

    assert torch.allclose(out1, out2)
    print(f"Outputs of shape {out1.shape} are close:")
test_f_cache_implementation()
# %%


def test_reverse_gt_correct():
    """
    Tests that the ground truth circuit is a complete description of the model on the reverse task
    """
    args = argparse.Namespace(
    wandb_name=None, wandb_project="subnetwork-probing", wandb_entity='tkwa', 
    wandb_group='snp', wandb_dir="/tmp/wandb", wandb_mode="online",
    device="cuda", lr=0.001, loss_type='l2', epochs=10000, verbose=1,
    lambda_reg=100, zero_ablation=0, reset_subject=0, seed=random.randint(0, 2**31 - 1),
    num_examples=50, seq_len=300, n_loss_average_runs=4, task='tracr-xproportion',
    torch_num_threads=0, print_stats=1, f=None
    )
    all_task_things = get_all_tracr_things(
    task="reverse", metric_name="l2", num_examples=6, device=torch.device("cpu")
    )
    gt_edges = get_tracr_reverse_edges()
    masked_model = MaskedTransformer(all_task_things.tl_model, no_ablate=True, use_pos_embed=True)
    masked_model.freeze_weights()

    # Zero out the model logits...
    for logit_name in masked_model._mask_logits_dict:
        masked_model._mask_logits_dict[logit_name].data.fill_(0)
    # Put the ground truth edges in the model logits...
    for c, ci, p, pi in gt_edges:
        # TODO deal with multiple heads
        if c not in masked_model.parent_node_names: 
            print(f"SKIPPING edge {c}, {ci}, {p}, {pi}")
            continue
        print(f"checking edge {c}, {ci}, {p}, {pi}")
        p_index = masked_model.parent_node_names[c].index(p)
        # TODO deal with ci, pi
        masked_model._mask_logits_dict[c].data[p_index] = 1

    n_active_logits = sum((l != 0).sum().item() for l in masked_model._mask_logits_dict.values())
    n_total_logits = sum(l.numel() for l in masked_model._mask_logits_dict.values())
    print(f"Model has {n_active_logits}/{n_total_logits} active logits and {len(gt_edges)} ground truth edges")
    
    # Run the model once without hooks
    rng_state = torch.random.get_rng_state()
    out1 = masked_model.model(all_task_things.validation_data)
    
    # Now run the masked model
    masked_model.do_random_resample_caching(all_task_things.validation_patch_data)
    context_args = dict(ablation='resample', ablation_data=all_task_things.validation_patch_data)
    with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
        out2 = hooked_model(all_task_things.validation_data)

    assert torch.allclose(out1, out2), "Outputs of the masked model and the unmasked model are not close"
test_reverse_gt_correct()
        


# %%
