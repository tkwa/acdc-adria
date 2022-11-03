# %%
from utils_circuit_discovery import get_hook_tuple
from ioi_utils import show_pp
from functools import partial
#import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import pandas as pd
import torch
import plotly.express as px
import gc
import einops
from collections import OrderedDict

from easy_transformer import EasyTransformer
from easy_transformer.experiments import get_act_hook

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
# %%
model_name = "gpt2"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']
model = EasyTransformer.from_pretrained(model_name)
if torch.cuda.is_available():
    model.to("cuda")
model.set_use_attn_result(True)

# %%

def patch_all(z, source_act, hook):
    return source_act

def patch_positions(z, source_act, hook, positions):
    if positions is None: # same as patch_all
        raise NotImplementedError("haven't implemented not specifying positions to patch")
        # return source_act
    else:
        batch = z.shape[0]
        for pos in positions:
            z[torch.arange(batch), pos] = source_act[torch.arange(batch), pos]
        return z

def path_patching(
    model: EasyTransformer, 
    orig_data, 
    new_data, 
    senders: List[Tuple], 
    receiver_hooks: List[Tuple], 
    max_layer: Union[int, None] = None, 
    position: int = 0,
    return_hooks: bool = False,
    freeze_mlps: bool = True):
    """ mlps are by default considered as just another component and so are
        by default frozen when collecting acts on receivers. 
        orig_data: string, torch.Tensor, or list of strings - any format that can be passed to the model directly
        new_data: same as orig_data
        senders: list of tuples (layer, head) for attention heads and (layer, None) for mlps
        receiver_hooks: list of tuples (hook_name, head) for attn heads and (hook_name, None) for mlps
        max_layer: layers beyond max_layer are not frozen when collecting receiver activations
        positions: default None and patch at all positions, or a tensor specifying the positions at which to patch

        NOTE: This relies on a change to the cache_some() function in EasyTransformer/hook_points.py.
    """
    if max_layer is None:
        max_layer = model.cfg.n_layers
    assert max_layer <= model.cfg.n_layers

    # save activations from orig
    orig_cache = {}
    model.cache_all(orig_cache)
    _ = model(orig_data)
    model.reset_hooks()

    # process senders
    sender_hooks = []
    for layer, head in senders:
        if head is not None: # (layer, head) for attention heads
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head))
        else: # (layer, None) for mlps
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))
    sender_hook_names = [x[0] for x in sender_hooks]

    # save activations from new for senders
    new_cache = {}
    model.cache_some(new_cache, lambda x: x in sender_hook_names)
    _ = model(new_data)
    model.reset_hooks()

    # set up receiver cache
    receiver_hook_names = [x[0] for x in receiver_hooks]
    receiver_cache = {}
    model.cache_some(receiver_cache, lambda x: x in receiver_hook_names)

    # configure hooks for freezing activations
    for layer in range(max_layer):
        # heads
        for head in range(model.cfg.n_heads):
            # if (layer, head) in senders:
            #     continue
            for hook_template in [
                'blocks.{}.attn.hook_q',
                'blocks.{}.attn.hook_k',
                'blocks.{}.attn.hook_v',
            ]:
                hook_name = hook_template.format(layer)
                hook = get_act_hook(
                    patch_all,
                    alt_act=orig_cache[hook_name],
                    idx=head,
                    dim=2,
                )
                model.add_hook(hook_name, hook)
        # mlp
        if freeze_mlps:
            hook_name = f'blocks.{layer}.hook_mlp_out'
            hook = get_act_hook(
                patch_all,
                alt_act=orig_cache[hook_name],
                idx=None,
                dim=None,
            )
            model.add_hook(hook_name, hook)
    
    # for senders, add new hook to patching in new acts
    for hook_name, head in sender_hooks:
        #assert not torch.allclose(orig_cache[hook_name], new_cache[hook_name]), (hook_name, head)
        hook = get_act_hook(
            partial(patch_positions, positions=[position]),
            alt_act=new_cache[hook_name],
            idx=head,
            dim=2 if head is not None else None,
        )
        model.add_hook(hook_name, hook)
    
    # forward pass on orig, where patch in new acts for senders and orig acts for the rest
    # and save activations on receivers
    _ = model(orig_data)
    model.reset_hooks()

    # add hooks for final forward pass on orig, where we patch in hybrid acts for receivers
    hooks = []
    for hook_name, head in receiver_hooks:
        #assert not torch.allclose(orig_cache[hook_name], receiver_cache[hook_name])
        hook = get_act_hook(
            partial(patch_positions, positions=[position]),
            alt_act=receiver_cache[hook_name],
            idx=head,
            dim=2 if head is not None else None,
        )
        hooks.append((hook_name, hook))
    
    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model

# %%
orig = "When John and Mary went to the store, John gave a bottle of milk to Mary."
new = "When John and Mary went to the store, Charlie gave a bottle of milk to Mary."
#new = "A completely different gibberish sentence blalablabladfghjkoiuytrdfg"

logit = model(orig)[0,16,5335]

# model = path_patching(
#     model, 
#     orig, 
#     new, 
#     [(8, 6)],
#     [('blocks.9.attn.hook_q', 9)],
#     12,
#     positions=[torch.tensor([16])],
# )

model = path_patching(
    model, 
    orig, 
    new, 
    [(5, 5)],
    [('blocks.8.attn.hook_v', 6)],
    12,
    positions=[torch.tensor([16])],
)

# model = path_patching(
#     model, 
#     orig, 
#     new, 
#     [(9, 9)],
#     [('blocks.11.hook_resid_post', None)],
#     12,
#     positions=[torch.tensor([16])],
# )

new_logit = model(orig)[0,16,5335]
model.reset_hooks()
print(logit, new_logit)

# %%
from ioi_dataset import (
    IOIDataset,
)

N = 1
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
)

abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)
# %%
# the circuit discovery algorithm
# 
# we want to write down the process of discovery the IOI circuit as an algorithm
# 1. start at the logits at the token position 'end', run path patching on each head and mlp.
# 2. pick threshold (probably in terms of percentage change in metric we care about?), identify components that have effect sizes above threshold
# 3. for comp in identified components:
## a. run path patching on all components upstream to it, with the q, k, or v part of comp as receiver

def path_patching_up_to(
    model: EasyTransformer, 
    layer: int,
    metric, 
    dataset,
    orig_data, 
    new_data, 
    receiver_hooks,
    position
):
    model.reset_hooks()
    attn_results = np.zeros((layer, model.cfg.n_heads))
    mlp_results = np.zeros((layer,1))
    for l in tqdm(range(layer)):
        for h in range(model.cfg.n_heads):
            model = path_patching(
                model,
                orig_data=orig_data,
                new_data=new_data,
                senders=[(l, h)],
                receiver_hooks=receiver_hooks,
                max_layer=model.cfg.n_layers,
                position=position
            )
            attn_results[l, h] = metric(model, dataset)
            model.reset_hooks()
        # mlp
        model = path_patching(
            model,
            orig_data=orig_data,
            new_data=new_data,
            senders=[(l, None)],
            receiver_hooks=receiver_hooks,
            max_layer=model.cfg.n_layers,
            position=position
        )
        mlp_results[l] = metric(model, dataset)
        model.reset_hooks()
    return attn_results, mlp_results

def logit_diff_io_s(model: EasyTransformer, dataset: IOIDataset):
    N = dataset.N
    io_logits = model(dataset.toks.long())[torch.arange(N), dataset.word_idx['end'], dataset.io_tokenIDs]
    s_logits = model(dataset.toks.long())[torch.arange(N), dataset.word_idx['end'], dataset.s_tokenIDs]
    return (io_logits - s_logits).mean().item()
    
# %%
receiver_hooks = [('blocks.11.hook_resid_post', None)]
attn_results, mlp_results = path_patching_up_to(
    model, 
    layer=12, 
    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    receiver_hooks=receiver_hooks,
    positions=[ioi_dataset.word_idx['end']])

# %%
model.reset_hooks()
default_logit_diff = logit_diff_io_s(model, ioi_dataset)
attn_results_n = (attn_results - default_logit_diff) / default_logit_diff
mlp_results_n = (mlp_results - default_logit_diff) / default_logit_diff
px.imshow(attn_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(np.expand_dims(mlp_results_n, axis=0), color_continuous_scale='RdBu', color_continuous_midpoint=0)

# %%
receiver_hooks = [('blocks.9.attn.hook_q', 9)]#, ('blocks.9.attn.hook_q', 6), ('blocks.10.attn.hook_q', 0)]
attn_results, mlp_results = path_patching_up_to(
    model, 
    layer=10, 
    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    receiver_hooks=receiver_hooks,
    positions=[ioi_dataset.word_idx['end']])
# %%
model.reset_hooks()
default_logit_diff = logit_diff_io_s(model, ioi_dataset)
attn_results_n = (attn_results - default_logit_diff) / default_logit_diff
mlp_results_n = (mlp_results - default_logit_diff) / default_logit_diff
px.imshow(attn_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(np.expand_dims(mlp_results_n, axis=0), color_continuous_scale='RdBu', color_continuous_midpoint=0)

# %%
receiver_hooks = [
    ('blocks.7.attn.hook_v', 3), 
    ('blocks.7.attn.hook_v', 9), 
    ('blocks.8.attn.hook_v', 6),
    ('blocks.8.attn.hook_v', 10)]
attn_results, mlp_results = path_patching_up_to(
    model, 
    layer=10, 

    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    receiver_hooks=receiver_hooks,
    positions=ioi_dataset.word_idx['S2'])
# %%
attn_results_n = (attn_results - default_logit_diff) / default_logit_diff
mlp_results_n = (mlp_results - default_logit_diff) / default_logit_diff
px.imshow(attn_results_n, color_continuous_scale='RdBu', color_continuous_midpoint=0).show()
px.imshow(np.expand_dims(mlp_results_n, axis=0), color_continuous_scale='RdBu', color_continuous_midpoint=0)

# %%

class Node():
    def __init__(self,
        layer: int,
        head: int,
        position: int, 
        #important: bool = False,
    ):
        self.layer = layer
        self.head = head
        self.position = position
        self.hook_name = get_hook_tuple(self.layer, self.head)
        #self.important = important
        self.children = []

    def __repr__(self):
        return f"Node({self.layer}, {self.head}, {self.position})"

class HypothesisTree():
    def __init__(self, model: EasyTransformer, metric: Callable, dataset, orig_data, new_data, threshold: int):
        self.model = model
        #self.node_stack = []
        #self.current_node = self.node_stack[-1]
        self.possible_positions = [ioi_dataset.word_idx['end']] # TODO: deal with positions
        self.node_stack = OrderedDict()
        self.populate_node_stack()
        self.current_node = self.node_stack[next(reversed(self.node_stack))] # last element
        self.metric = metric
        self.dataset = dataset
        self.orig_data = orig_data
        self.new_data = new_data
        self.threshold = threshold
        self.default_metric = self.metric(model, dataset)

    def populate_node_stack(self):
        for layer in range(self.model.cfg.n_layers):
            for head in list(range(self.model.cfg.n_heads)) + [None]: # includes None for mlp
                for pos in self.possible_positions:
                    node = Node(layer, head, pos)
                    self.node_stack[(layer, head, pos)] = node
                    #self.importance_dict[node] = False
                    #self.node_stack.append(node)
        layer = self.model.cfg.n_layers
        pos = self.possible_positions[-1] # assume the last position specified is the one that we care about in the residual stream
        resid_post = Node(layer, None, pos) 
        resid_post.hook_name = f'blocks.{layer-1}.hook_resid_post'
        self.node_stack[(layer, None, pos)] = resid_post # this represents blocks.{last}.hook_resid_post
        #self.node_stack.append(resid_post) 

    def eval(self):
        """Process current_node, then move to next current_node"""

        _, node = self.node_stack.popitem()

        # do path patching on all nodes before node
        attn_results, mlp_results = path_patching_up_to(
            model=model, 
            metric=self.metric,
            dataset=self.dataset,
            layer=node.layer,
            orig_data=self.orig_data, 
            new_data=self.new_data, 
            receiver_hooks=[(node.hook_name, node.head)],
            position=node.position
        ) 

        attn_results -= self.default_metric
        attn_results /= self.default_metric
        mlp_results -= self.default_metric
        mlp_results /= self.default_metric

        show_pp(attn_results.T)
        show_pp(mlp_results)

        # process result and mark nodes above threshold as important
        for layer in range(attn_results.shape[0]):
            for head in range(attn_results.shape[1]):
                if abs(attn_results[layer, head]) > self.threshold:
                    self.node_stack[(layer, head, node.position)].children.append(node)
            if abs(mlp_results[layer]) > self.threshold:
                self.node_stack[(layer, None, node.position)].children.append(node)

        # iterate to next node
    
        # update self.current_node
        while len(self.node_stack) > 0 and len(self.node_stack[next(reversed(self.node_stack))].children) == 0:
            self.node_stack.popitem()
        if len(self.node_stack) > 0:
            self.current_node = self.node_stack[next(reversed(self.node_stack))]
        else:
            self.current_node = None

    def show(self):
        print("pretty picture")


h = HypothesisTree(
    model, 
    metric=logit_diff_io_s, 
    dataset=ioi_dataset, 
    orig_data=ioi_dataset.toks.long(), 
    new_data=abc_dataset.toks.long(), 
    threshold=0.2)

# %%

base_hypothesis = HypothesisTree()
next_hypotheses = base_hypothesis.expand(1)
good_hypos = []
for next_hypo in next_hypotheses:
    score = next_hypo.eval() # path patching
    if score > threshold:
        good_hypos.append(next_hypo)
base_hypothesis = HypothesisTree(good_hypos)

# %%

# start at blocks.11.hook_resid_post
# results = path_patching_up_to(11)
# for h
threshold = 0.2

receiver_hooks = [('blocks.11.hook_resid_post', None)]

attn_results, mlp_results = path_patching_up_to(
    model, 
    layer=12, 
    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    receiver_hooks=receiver_hooks,
    positions=[ioi_dataset.word_idx['end']])

for layer in range(12):
    for head in range(12):
        if attn_results[layer, head] > threshold:
            receiver_hooks = [
                (f'blocks.{layer}.attn.hook_q', head),
                (f'blocks.{layer}.attn.hook_k', head),
                (f'blocks.{layer}.attn.hook_v', head)]
            new_attn_results, new_mlp_results = path_patching_up_to(
                model,
                layer=layer,
                metric=logit_diff_io_s,
                dataset=ioi_dataset,
                orig_data=ioi_dataset.toks.long(),
                new_data=abc_dataset.toks.long(),
                receiver_hooks=receiver_hooks,
                positions=[ioi_dataset.word_idx['end']]
            )
