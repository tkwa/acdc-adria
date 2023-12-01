# %%

import argparse
import collections
import gc
import math
import random
from typing import Callable, ContextManager, Dict, List, Optional, Tuple

import torch
import wandb
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint

from acdc.acdc_utils import filter_nodes, get_edge_stats, get_node_stats, get_present_nodes, reset_network
from acdc.docstring.utils import AllDataThings, get_all_docstring_things, get_docstring_subgraph_true_edges
from acdc.tracr_task.utils import get_tracr_proportion_edges, get_tracr_reverse_edges
from acdc.greaterthan.utils import get_all_greaterthan_things, get_greaterthan_true_edges
from acdc.induction.utils import get_all_induction_things #, get_induction_true_edges
from acdc.ioi.utils import get_all_ioi_things, get_ioi_true_edges
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import Edge, EdgeType, TorchIndex
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.tracr_task.utils import get_all_tracr_things


def iterative_correspondence_from_mask(
    model: HookedTransformer,
    nodes_to_mask: List[TLACDCInterpNode], # Can be empty
    use_pos_embed: bool = False,
    corr: Optional[TLACDCCorrespondence] = None,
    head_parents: Optional[List] = None,
) -> Tuple[TLACDCCorrespondence, List]:
    """Given corr has some nodes masked, also mask the nodes_to_mask"""

    assert (corr is None) == (head_parents is None), "Ensure we're either masking from scratch or we provide details on `head_parents`"

    if corr is None:
        corr = TLACDCCorrespondence.setup_from_model(model, use_pos_embed=use_pos_embed)
    if head_parents is None:
        head_parents = collections.defaultdict(lambda: 0)

    additional_nodes_to_mask = []

    for node in nodes_to_mask:
        additional_nodes_to_mask.append(
            TLACDCInterpNode(node.name.replace(".attn.", ".") + "_input", node.index, EdgeType.ADDITION)
        )

        if node.name.endswith("_q") or node.name.endswith("_k") or node.name.endswith("_v"):
            child_name = node.name.replace("_q", "_result").replace("_k", "_result").replace("_v", "_result")
            head_parents[(child_name, node.index)] += 1

            if head_parents[(child_name, node.index)] == 3:
                additional_nodes_to_mask.append(TLACDCInterpNode(child_name, node.index, EdgeType.PLACEHOLDER))

            # Forgot to add these in earlier versions of Subnetwork Probing, and so the edge counts were inflated
            additional_nodes_to_mask.append(TLACDCInterpNode(child_name + "_input", node.index, EdgeType.ADDITION))

        if node.name.endswith(("mlp_in", "resid_mid")):
            additional_nodes_to_mask.append(
                TLACDCInterpNode(
                    node.name.replace("resid_mid", "mlp_out").replace("mlp_in", "mlp_out"),
                    node.index,
                    EdgeType.DIRECT_COMPUTATION,
                )
            )

    assert all([v <= 3 for v in head_parents.values()]), "We should have at most three parents (Q, K and V, connected via placeholders)"

    for node in nodes_to_mask + additional_nodes_to_mask:
        # Mark edges where this is child as not present
        rest2 = corr.edges[node.name][node.index]
        for rest3 in rest2.values():
            for edge in rest3.values():
                edge.present = False

        # Mark edges where this is parent as not present
        for rest1 in corr.edges.values():
            for rest2 in rest1.values():
                if node.name in rest2 and node.index in rest2[node.name]:
                    rest2[node.name][node.index].present = False

    return corr, head_parents




def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


class MaskedTransformer(torch.nn.Module):
    """
    A wrapper around HookedTransformer that allows edge-level subnetwork probing.

    There are two sets of hooks:
    - `activation_mask_hook`s change the input to a node. The input to a node is the sum
      of several residual stream terms; ablated edges are looked up from `ablation_cache`
      and non-ablated edges from `forward_cache`, then the sum is taken.
    - `caching_hook`s save the output of a node to `forward_cache` for use in later layers.
    """
    model: HookedTransformer
    ablation_cache: ActivationCache
    forward_cache: ActivationCache
    mask_logits: torch.nn.ParameterList
    # what is the purpose of this? why can't you just use dict.keys()? -tkwa
    mask_logits_names: List[str]
    _mask_logits_dict: Dict[str, torch.nn.Parameter]

    def __init__(self, model:HookedTransformer, beta=2 / 3, gamma=-0.1, zeta=1.1, mask_init_p=0.9, no_ablate=False):
        super().__init__()

        self.model = model
        self.n_heads = model.cfg.n_heads
        self.n_mlp = 0 if model.cfg.attn_only else 1
        self.mask_logits = torch.nn.ParameterList()
        self.mask_logits_names = []
        self._mask_logits_dict = {}
        self.no_ablate = no_ablate
        self.device = self.model.parameters().__next__().device

        # Stores the cache keys that correspond to each mask,
        # e.g. ...1.hook_mlp_in -> ["blocks.0.attn.hook_result", "blocks.0.hook_mlp_out", "blocks.1.attn.hook_result"]
        # Logits are attention in-edges, then MLP in-edges
        # TODO this is ugly, maybe change to single list and use names to reshape correctly?
        self.parent_node_names:Dict[str, Tuple[list[str], list[str]]] = {}
        self.forward_cache_names = []

        self.ablation_cache = ActivationCache({}, self.model)
        self.forward_cache = ActivationCache({}, self.model)
        # Hyperparameters
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.mask_init_p = mask_init_p

        # Copied from subnetwork probing code. Similar to log odds (but not the same)
        p = (self.mask_init_p - self.gamma) / (self.zeta - self.gamma)
        self.mask_init_constant = math.log(p / (1 - p))

        model.cfg.use_hook_mlp_in = True # We need to hook the MLP input to do subnetwork probing

        # Add mask logits for ablation cache
        # Mask logits have a variable dimension depending on the number of in-edges (increases with layer)
        for layer_i in range(model.cfg.n_layers):
            # QKV: in-edges from all previous layers
            if layer_i > 0:
                for q_k_v in ["q", "k", "v"]:

                    self._setup_mask_logits(
                        mask_name=f"blocks.{layer_i}.hook_{q_k_v}_input",
                        parent_nodes=([f"blocks.{l}.attn.hook_result" for l in range(layer_i)],
                                           [f"blocks.{l}.hook_mlp_out" for l in range(layer_i)] if not model.cfg.attn_only else []),
                        out_dim=self.n_heads)

            # MLP: in-edges from all previous layers and current layer's attention heads
            if not model.cfg.attn_only:
                parent_nodes = (
                    [f"blocks.{l}.attn.hook_result" for l in range(layer_i + 1)],
                    [f"blocks.{l}.hook_mlp_out" for l in range(layer_i)]
                )
                self._setup_mask_logits(
                    mask_name = f"blocks.{layer_i}.hook_mlp_in",
                    parent_nodes=parent_nodes,
                    out_dim=1)
            
        self._setup_mask_logits(
            mask_name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
            parent_nodes=([f"blocks.{l}.attn.hook_result" for l in range(model.cfg.n_layers)],
                               [f"blocks.{l}.hook_mlp_out" for l in range(model.cfg.n_layers)] if not model.cfg.attn_only else []),
            out_dim=1
        )

        # Add hook points for forward cache
        self.forward_cache_names.append("blocks.0.hook_resid_pre") # not counted as a node in gt, but in resid stream
        for layer_i in range(model.cfg.n_layers):
            # print(f"adding forward cache for layer {layer_index}")
            if not model.cfg.attn_only:
                self.forward_cache_names.append(f"blocks.{layer_i}.hook_mlp_out")
            self.forward_cache_names.append(f"blocks.{layer_i}.attn.hook_result")
        assert all([name in self.forward_cache_names for ckl in self.parent_node_names.values() for attn_or_mlp in ckl for name in attn_or_mlp])

    def _setup_mask_logits(self, mask_name, parent_nodes, out_dim):
        """
        Adds a mask logit for the given mask name and parent nodes
        Parent nodes are (attention, MLP)
        """
        self.parent_node_names[mask_name] = parent_nodes
        self.mask_logits.append(torch.nn.Parameter(
            torch.full((len(parent_nodes[0])*self.n_heads + len(parent_nodes[1]), out_dim), self.mask_init_constant, device=self.device)
        ))
        self.mask_logits_names.append(mask_name)
        self._mask_logits_dict[mask_name] = self.mask_logits[-1]

    def sample_mask(self, mask_name) -> torch.Tensor:
        """Samples a binary-ish mask from the mask_scores for the particular `mask_name` activation"""
        mask_scores = self._mask_logits_dict[mask_name]
        uniform_sample = torch.zeros_like(mask_scores).uniform_().clamp_(0.0001, 0.9999)
        s = torch.sigmoid((uniform_sample.log() - (1 - uniform_sample).log() + mask_scores) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = s_bar.clamp(min=0.0, max=1.0)
        return mask

    def regularization_loss(self) -> torch.Tensor:
        center = self.beta * math.log(-self.gamma / self.zeta)
        per_parameter_loss = [
            torch.sigmoid(scores - center).mean()
            for scores in self.mask_logits
        ]
        return torch.mean(torch.stack(per_parameter_loss))

    # def mask_logits_names_filter(self, name):
    #     return name in self.mask_logits_names

    def do_random_resample_caching(self, patch_data) -> torch.Tensor:
        # Only cache the tensors needed to fill the masked out positions
        with torch.no_grad():
            model_out, self.ablation_cache = self.model.run_with_cache(
                patch_data, names_filter=lambda name: name in self.forward_cache_names, return_cache_object=True
            )
        return model_out

    def do_zero_caching(self):
        """Caches zero for every possible mask point.
        """
        patch_data = torch.zeros((1, 1), device=self.device, dtype=torch.int64) # batch pos
        self.do_random_resample_caching(patch_data)
        self.ablation_cache.cache_dict = \
            {name: torch.zeros_like(scores) for name, scores in self.ablation_cache.cache_dict.items()}

    def get_mask_values(self, names, cache:ActivationCache):
        """
        Returns a single tensor of the mask values used for a given hook.
        Attention is shape batch, seq, heads, head_size while MLP out is batch, seq, d_model
        so we need to reshape things to match
        """
        attn_names, mlp_names = names
        result = []
        for name in attn_names:
            value = cache[name] # b s n_heads d
            result.append(value)
        for name in mlp_names:
            value = repeat(cache[name], 'b s d -> b s 1 d')
            result.append(value)
        return torch.cat(result, dim=2)

    def activation_mask_hook(self, hook_point_out: torch.Tensor, hook: HookPoint):
        """
        For edge-level SP, we discard the hook_point_out value and resum the residual stream.
        """
        is_attn = 'mlp' not in hook.name and 'resid_post' not in hook.name
        # print(f"Doing ablation of {hook.name}")
        mask = self.sample_mask(hook.name) # in_edges, nodes_per_mask, ...
        if self.no_ablate: mask = torch.ones_like(mask) # for testing only

        # Get values from ablation cache and forward cache
        names = self.parent_node_names[hook.name]
        a_values = self.get_mask_values(names, self.ablation_cache) # in_edges, ...
        f_values = self.get_mask_values(names, self.forward_cache) # in_edges, ...
        # print(f"{a_values.shape=}, {f_values.shape=}, {mask.shape=}, target shape={hook_point_out.shape}")

        # Add embedding and biases
        out = (self.forward_cache['blocks.0.hook_resid_pre']).unsqueeze(2) # b s 1 d
        if is_attn: out = out.repeat(1, 1, self.n_heads, 1) # b s n_heads d
        # Resum the residual stream
        weighted_a_values = torch.einsum("b s i d, i o -> b s o d", a_values, 1 - mask)
        weighted_f_values = torch.einsum("b s i d, i o -> b s o d", f_values, mask)
        out += weighted_a_values + weighted_f_values
        if not is_attn:
            out = rearrange(out, 'b s 1 d -> b s d')

        block_num = int(hook.name.split('.')[1])
        for layer in self.model.blocks[:block_num if 'mlp' in hook.name else 1]:
            out += layer.attn.b_O

        # if not torch.allclose(hook_point_out, out):
        #     print(f"Warning: hook_point_out and out are not close for {hook.name}")
        #     print(f"{hook_point_out.mean()=}, {out.mean()=}")
        # print(f"{out.shape=}")
        return out
    
    def caching_hook(self, hook_point_out: torch.Tensor, hook: HookPoint):
        self.forward_cache.cache_dict[hook.name] = hook_point_out
        return hook_point_out

    def fwd_hooks(self) -> List[Tuple[str, Callable]]:
        return [(n, self.activation_mask_hook) for n in self.mask_logits_names] + \
            [(n, self.caching_hook) for n in self.forward_cache_names]


    def with_fwd_hooks_and_new_cache(self, ablation='resample', ablation_data=None) -> ContextManager[HookedTransformer]:
        assert ablation in ['zero', 'resample']
        if ablation == 'zero':
            self.do_zero_caching()
        else:
            assert ablation_data is not None
            self.do_random_resample_caching(ablation_data)
        return self.model.hooks(self.fwd_hooks())

    def freeze_weights(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def num_edges(self):
        values = []
        for name, mask in self._mask_logits_dict.items():
            mask_value = self.sample_mask(name)
            values.extend(mask_value.flatten().tolist())
        values = torch.tensor(values)
        return (values > 0.5).sum().item()


def edge_level_corr(masked_model: MaskedTransformer, use_pos_embed:bool=False) -> TLACDCCorrespondence:
    corr = TLACDCCorrespondence.setup_from_model(masked_model.model, use_pos_embed=use_pos_embed)
    # Sample masks for all edges
    masks = dict()
    for name in masked_model._mask_logits_dict.keys():
        sampled_mask = masked_model.sample_mask(name)
        masks[name] = sampled_mask
    # Define edges
    for child, sampled_mask in masks.items():
        # not sure if this is the right way to do indexing
        child_index = TorchIndex((None,) if 'mlp' in child or 'resid' in child else (None, None, 0))
        attn_parents, mlp_parents = masked_model.parent_node_names[child]
        parents = attn_parents + mlp_parents
        for i, parent in enumerate(parents):
            parent_index = TorchIndex((None,) if 'mlp' in parent or 'resid' in parent else (None, None, 0))

            edge = corr.edges[child][child_index][parent][parent_index]
            edge.present = (sampled_mask[i] >= 0.5).item()
    
    # Delete a node's incoming edges if it has no outgoing edges
    def get_nodes_with_out_edges(corr):
        nodes_with_out_edges = set()
        for (receiver_name, receiver_index, sender_name, sender_index), edge in corr.all_edges().items():
            nodes_with_out_edges.add(sender_name)
        return nodes_with_out_edges
    for (receiver_name, receiver_index, sender_name, sender_index), edge in corr.all_edges().items():
        if receiver_name not in get_nodes_with_out_edges(corr):
            edge.present = False
    return corr

# %%

def visualize_mask(masked_model: MaskedTransformer) -> tuple[int, list[TLACDCInterpNode]]:
    # This is bad code, shouldn't combine visualizing and getting the nodes to mask
    number_of_heads = masked_model.model.cfg.n_heads
    number_of_layers = masked_model.model.cfg.n_layers
    node_name_list = []
    mask_scores_for_names = []
    total_nodes = 0
    nodes_to_mask: list[TLACDCInterpNode] = []
    for layer_index in range(number_of_layers):
        for head_index in range(number_of_heads):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                node_name = f"blocks.{layer_index}.attn.hook_{q_k_v}"
                mask_sample = masked_model.sample_mask(node_name)[head_index].cpu().item()

                node_name_with_index = f"{node_name}[{head_index}]"
                node_name_list.append(node_name_with_index)
                node = TLACDCInterpNode(
                    node_name, TorchIndex((None, None, head_index)), incoming_edge_type=EdgeType.ADDITION
                )

                mask_scores_for_names.append(mask_sample)
                if mask_sample < 0.5:
                    nodes_to_mask.append(node)

        # MLPs
        # This is actually fairly wrong for getting the exact nodes and edges we keep in the circuit but in the `filter_nodes` function
        # used in post-processing (in roc_plot_generator.py we process hook_resid_mid/mlp_in and mlp_out hooks together properly) we iron
        # these errors so that plots are correct
        node_name = f"blocks.{layer_index}.hook_mlp_out"
        mask_sample = masked_model.sample_mask(node_name).cpu().item()
        mask_scores_for_names.append(mask_sample)
        node_name_list.append(node_name)

        for node_name, edge_type in [
            (f"blocks.{layer_index}.hook_mlp_out", EdgeType.PLACEHOLDER),
            (f"blocks.{layer_index}.hook_resid_mid", EdgeType.ADDITION),
        ]:
            node = TLACDCInterpNode(node_name, TorchIndex([None]), incoming_edge_type=edge_type)
            total_nodes += 1

            if mask_sample < 0.5:
                nodes_to_mask.append(node)

    # assert len(mask_scores_for_names) == 3 * number_of_heads * number_of_layers
    log_plotly_bar_chart(x=node_name_list, y=mask_scores_for_names)
    node_count = total_nodes - len(nodes_to_mask)
    return node_count, nodes_to_mask

def set_ground_truth_edges(canonical_circuit_subgraph: TLACDCCorrespondence, ground_truth_set: set):
    for (receiver_name, receiver_index, sender_name, sender_index), edge in canonical_circuit_subgraph.all_edges().items():
        key =(receiver_name, receiver_index.hashable_tuple, sender_name, sender_index.hashable_tuple)
        edge.present = (key in ground_truth_set)


def print_stats(recovered_corr, ground_truth_subgraph, do_print=True, wandb_log=True):
    """
    False postitive = present in recovered_corr but not in ground_truth_set
    """

    stats = get_node_stats(ground_truth=ground_truth_subgraph, recovered=recovered_corr)
    print(stats)
    node_tpr = stats["true positive"] / (stats["true positive"] + stats["false negative"])
    node_fpr = stats["false positive"] / (stats["false positive"] + stats["true negative"])
    if do_print:print(f"Node TPR: {node_tpr:.3f}. Node FPR: {node_fpr:.3f}")

    stats = get_edge_stats(ground_truth=ground_truth_subgraph, recovered=recovered_corr)
    edge_tpr = stats["true positive"] / (stats["true positive"] + stats["false negative"])
    edge_fpr = stats["false positive"] / (stats["false positive"] + stats["true negative"])
    if do_print:print(f"Edge TPR: {edge_tpr:.3f}. Edge FPR: {edge_fpr:.3f}")

    if wandb_log:
        wandb.log(
            {
                "node_tpr": node_tpr,
                "node_fpr": node_fpr,
                "edge_tpr": edge_tpr,
                "edge_fpr": edge_fpr,
            }
        )

def train_sp(
    args,
    masked_model: MaskedTransformer,
    all_task_things: AllDataThings,
    print_every:int=50,
    get_true_edges: Callable = None,
):
    epochs = args.epochs
    lambda_reg = args.lambda_reg

    torch.manual_seed(args.seed)

    wandb.init(
        name=args.wandb_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config=args,
        dir=args.wandb_dir,
        mode=args.wandb_mode,
    )
    test_metric_fns = all_task_things.test_metrics

    print("Reset subject:", args.reset_subject)
    if args.reset_subject:
        reset_network(args.task, args.device, masked_model.model)
        gc.collect()
        torch.cuda.empty_cache()
        masked_model.freeze_weights()

        with torch.no_grad():
            reset_logits = masked_model.model(all_task_things.validation_data)
            print("Reset validation metric: ", all_task_things.validation_metric(reset_logits))
            reset_logits = masked_model.model(all_task_things.test_data)
            print("Reset test metric: ", {k: v(reset_logits).item() for k, v in all_task_things.test_metrics.items()})

    # one parameter per thing that is masked
    mask_params = list(p for p in masked_model.mask_logits if p.requires_grad)
    # parameters for the probe (we don't use a probe)
    model_params = list(p for p in masked_model.model.parameters() if p.requires_grad)
    assert len(model_params) == 0, ("MODEL should be empty", model_params)
    trainer = torch.optim.Adam(mask_params, lr=args.lr)


    if args.zero_ablation:
        context_args = dict(ablation='zero')
    else:
        context_args = dict(ablation='resample', ablation_data=all_task_things.validation_patch_data)

    # Get canonical subgraph so we can print TPR, FPR
    canonical_circuit_subgraph = TLACDCCorrespondence.setup_from_model(masked_model.model, use_pos_embed=False)
    d_trues = set(get_true_edges())
    set_ground_truth_edges(canonical_circuit_subgraph, d_trues)

    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        masked_model.train()
        trainer.zero_grad()

        with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
            specific_metric_term = all_task_things.validation_metric(hooked_model(all_task_things.validation_data))
        regularizer_term = masked_model.regularization_loss()
        loss = specific_metric_term + regularizer_term * lambda_reg
        loss.backward()

        trainer.step()

        if epoch % print_every == 0 and args.print_stats:
            wandb.log({
                "epoch": epoch,
                "num_edges": masked_model.num_edges(),
            })
            # TODO edit this to create a corr from masked edges
            # number_of_nodes, nodes_to_mask = visualize_mask(masked_model)
            # corr, _ = iterative_correspondence_from_mask(masked_model.model, nodes_to_mask)
            # print_stats(corr, d_trues, canonical_circuit_subgraph)
            corr = edge_level_corr(masked_model)
            print_stats(corr, canonical_circuit_subgraph)
            

    wandb.log(
        {
            "regularisation_loss": regularizer_term.item(),
            "specific_metric_loss": specific_metric_term.item(),
            "total_loss": loss.item(),
        }
    )

    with torch.no_grad():
        # The loss has a lot of variance so let's just average over a few runs with the same seed
        rng_state = torch.random.get_rng_state()

        # Final training loss
        specific_metric_term = 0.0
        if args.zero_ablation:
            masked_model.do_zero_caching()
        else:
            masked_model.do_random_resample_caching(all_task_things.validation_patch_data)

        for _ in range(args.n_loss_average_runs):
            with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
                specific_metric_term += all_task_things.validation_metric(
                    hooked_model(all_task_things.validation_data)
                ).item()
        print(f"Final train/validation metric: {specific_metric_term:.4f}")

        if args.zero_ablation:
            masked_model.do_zero_caching()
        else:
            masked_model.do_random_resample_caching(all_task_things.test_patch_data)

        test_specific_metrics = {}
        for k, fn in test_metric_fns.items():
            torch.random.set_rng_state(rng_state)
            test_specific_metric_term = 0.0
            # Test loss
            for _ in range(args.n_loss_average_runs):
                with masked_model.with_fwd_hooks_and_new_cache(**context_args) as hooked_model:
                    test_specific_metric_term += fn(hooked_model(all_task_things.test_data)).item()
            test_specific_metrics[f"test_{k}"] = test_specific_metric_term

        print(f"Final test metric: {test_specific_metrics}")

        log_dict = dict(
            # number_of_nodes=number_of_nodes,
            specific_metric=specific_metric_term,
            # nodes_to_mask=nodes_to_mask,
            **test_specific_metrics,
        )
    return masked_model, log_dict


def proportion_of_binary_scores(model: MaskedTransformer) -> float:
    """How many of the scores are binary, i.e. 0 or 1
    (after going through the sigmoid with fp32 precision loss)
    """
    binary_count = 0
    total_count = 0

    for mask_name in model.mask_logits_names:
        mask = model.sample_mask(mask_name)
        for v in mask.view(-1):
            total_count += 1
            if v == 0 or v == 1:
                binary_count += 1
    return binary_count / total_count


parser = argparse.ArgumentParser("train_induction")
parser.add_argument("--wandb-name", type=str, required=False)
parser.add_argument("--wandb-project", type=str, default="subnetwork-probing")
parser.add_argument("--wandb-entity", type=str, required=True)
parser.add_argument("--wandb-group", type=str, required=True)
parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
parser.add_argument("--wandb-mode", type=str, default="online")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--loss-type", type=str, required=True)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--lambda-reg", type=float, default=100)
parser.add_argument("--zero-ablation", type=int, required=True)
parser.add_argument("--reset-subject", type=int, default=0)
parser.add_argument("--seed", type=int, default=random.randint(0, 2**31 - 1), help="Random seed (default: random)")
parser.add_argument("--num-examples", type=int, default=50)
parser.add_argument("--seq-len", type=int, default=300)
parser.add_argument("--n-loss-average-runs", type=int, default=4)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--torch-num-threads", type=int, default=0, help="How many threads to use for torch (0=all)")
parser.add_argument("--print-stats", type=int, default=1, required=False)

# %%
if __name__ == "__main__":
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    torch.manual_seed(args.seed)

    if args.task == "ioi":
        all_task_things = get_all_ioi_things(
            num_examples=args.num_examples,
            device=torch.device(args.device),
            metric_name=args.loss_type,
        )
        get_true_edges = get_ioi_true_edges
    elif args.task == "induction":
        all_task_things = get_all_induction_things(
            args.num_examples,
            args.seq_len,
            device=torch.device(args.device),
            metric=args.loss_type,
        )
        get_true_edges = get_induction_true_edges # missing??? -tkwa
    elif args.task == "tracr-reverse":
        all_task_things = get_all_tracr_things(
            task="reverse", metric_name=args.loss_type, num_examples=args.num_examples, device=torch.device(args.device)
        )
        get_true_edges = get_tracr_reverse_edges
    elif args.task == "tracr-proportion":
        all_task_things = get_all_tracr_things(
            task="proportion",
            metric_name=args.loss_type,
            num_examples=args.num_examples,
            device=torch.device(args.device),
        )
        get_true_edges = get_tracr_proportion_edges
    elif args.task == "docstring":
        all_task_things = get_all_docstring_things(
            num_examples=args.num_examples,
            seq_len=args.seq_len,
            device=torch.device(args.device),
            metric_name=args.loss_type,
            correct_incorrect_wandb=True,
        )
        get_true_edges = get_docstring_subgraph_true_edges
    elif args.task == "greaterthan":
        all_task_things = get_all_greaterthan_things(
            num_examples=args.num_examples,
            metric_name=args.loss_type,
            device=args.device,
        )
        get_true_edges = get_greaterthan_true_edges
    else:
        raise ValueError(f"Unknown task {args.task}")

    masked_model = MaskedTransformer(all_task_things.tl_model)
    masked_model = masked_model.to(args.device)

    masked_model.freeze_weights()
    print("Finding subnetwork...")
    masked_model, log_dict = train_sp(
        args=args,
        masked_model=masked_model,
        all_task_things=all_task_things,
        get_true_edges=get_true_edges,
    )

    percentage_binary = proportion_of_binary_scores(masked_model)

    # Update dict with some different things
    # log_dict["nodes_to_mask"] = list(map(str, log_dict["nodes_to_mask"]))
    # to_log_dict["number_of_edges"] = corr.count_no_edges() TODO
    log_dict["percentage_binary"] = percentage_binary

    wandb.log(log_dict)

    wandb.finish()
