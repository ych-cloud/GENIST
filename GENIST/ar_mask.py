import math
import torch
from typing import List, Tuple


def fixed_ar_splits(S: int, step_size: int) -> Tuple[List[int], List[int]]:
    """
    Split a sequence of length S into chunks of size `step_size` (last one may be smaller).
    Returns (split_sizes, cumsum).
    """
    assert step_size > 0 and S > 0
    splits = [step_size] * (S // step_size)
    if S % step_size != 0:
        splits.append(S % step_size)
    cumsum = [0]
    acc = 0
    for s in splits:
        acc += s
        cumsum.append(acc)
    return splits, cumsum


def sample_random_ar_splits(
    S: int,
    decay: float = 0.9,
    min_steps: int = 1,
    max_steps: int = None,
    device: torch.device = None,
    generator: torch.Generator = None,
) -> Tuple[List[int], List[int]]:
    """
    Random AR partition:
      1) sample total steps N from an exponential decay distribution over [min_steps, max_steps]
      2) uniformly draw N-1 cut points in [1, S) and sort them
      3) derive split sizes and cumsum.
    Returns python lists (split_sizes, cumsum).
    """
    S = int(S)
    assert S > 0
    min_steps = max(1, int(min_steps))
    if max_steps is None or max_steps <= 0:
        max_steps = S
    max_steps = min(int(max_steps), S)
    if min_steps > max_steps:
        min_steps = max_steps

    steps_range = torch.arange(min_steps, max_steps + 1, device=device, dtype=torch.float32)
    if decay <= 0:
        decay = 1.0
    if abs(decay - 1.0) < 1e-8:
        probs = torch.ones_like(steps_range) / steps_range.numel()
    else:
        weights = torch.pow(torch.tensor(float(decay), device=device, dtype=torch.float32), steps_range - min_steps)
        probs = weights / weights.sum()
    # multinomial expects probabilities on device
    idx = torch.multinomial(probs, num_samples=1).item()
    N = int(steps_range[int(idx)].item())
    if N <= 1:
        return [S], [0, S]

    cuts = torch.randperm(S - 1, device=device, generator=generator)[: N - 1] + 1
    cuts, _ = torch.sort(cuts)
    cuts_list = cuts.tolist()

    split_sizes = [cuts_list[0]]
    for i in range(1, len(cuts_list)):
        split_sizes.append(cuts_list[i] - cuts_list[i - 1])
    split_sizes.append(S - cuts_list[-1])

    cumsum = [0]
    acc = 0
    for s in split_sizes:
        acc += s
        cumsum.append(acc)
    return split_sizes, cumsum


def get_attn_mask(sample_len: int, cond_len: int, split_sizes: List[int], cumsum: List[int]) -> torch.Tensor:
    """
    Generalized causal mask (boolean) following CausalFusion for a 1D token sequence.
    - clean tokens: first `sample_len - split_sizes[-1]`
    - noisy tokens: last `split_sizes[-1]`
    Mask has three blocks: clean->clean, clean->noisy, noisy->noisy
    Returns a tensor of shape [1, 1, seq_len, seq_len] with True where masked.
    """
    assert sum(split_sizes) == sample_len
    visible_len = sample_len - split_sizes[-1]
    ctx_len = cond_len + visible_len
    seq_len = ctx_len + sample_len

    attn_mask = torch.ones(size=(seq_len, seq_len), dtype=torch.bool)
    attn_mask[:, :cond_len] = False

    # build `triangle` masks
    triangle1 = torch.ones(size=(visible_len, visible_len), dtype=torch.bool)
    triangle2 = torch.ones(size=(sample_len, visible_len), dtype=torch.bool)
    triangle3 = torch.ones(size=(sample_len, sample_len), dtype=torch.bool)
    for i in range(len(split_sizes) - 1):
        triangle1[cumsum[i]:cumsum[i+1], 0:cumsum[i+1]] = False
        triangle2[cumsum[i+1]:cumsum[i+2], 0:cumsum[i+1]] = False
    for i in range(len(split_sizes)):
        triangle3[cumsum[i]:cumsum[i+1], cumsum[i]:cumsum[i+1]] = False

    # paste triangles
    attn_mask[cond_len:ctx_len, cond_len:ctx_len] = triangle1
    attn_mask[ctx_len:, cond_len:ctx_len] = triangle2
    attn_mask[ctx_len:, ctx_len:] = triangle3

    return attn_mask[None, None, :, :]


def get_attn_mask_single_seq(sample_len: int, split_sizes: List[int], cumsum: List[int]) -> torch.Tensor:
    """
    Generalized causal mask for a single token sequence of length L (no extra
    cond tokens, no duplicated xn). Layout:
      - clean indices: [0:visible_len)
      - noisy indices: [visible_len:L)
    Blocks:
      - clean->clean: lower-triangular causal (allow attend to <= i)
      - clean->noisy: fully masked
      - noisy->clean: fully visible (allow attend to all clean)
      - noisy->noisy: lower-triangular causal within the noisy block
    Returns shape [1, 1, L, L] boolean mask (True = masked).
    """
    assert sum(split_sizes) == sample_len
    L = sample_len
    visible_len = L - split_sizes[-1]
    mask = torch.ones((L, L), dtype=torch.bool)
    # clean->clean causal
    for i in range(visible_len):
        mask[i, : i + 1] = False
    # clean->noisy: keep masked (True)
    # noisy rows
    for i in range(visible_len, L):
        # noisy->clean: allow all clean keys
        mask[i, :visible_len] = False
        # noisy->noisy: causal inside noisy block
        mask[i, visible_len : i + 1] = False
    return mask[None, None, :, :]


def get_attn_mask_at_step(sample_len: int, split_sizes: List[int], cumsum: List[int], step_index: int) -> torch.Tensor:
    """
    Three-zone generalized causal mask for a chosen AR step.
    Args:
      - sample_len: total tokens L
      - split_sizes, cumsum: AR partition
      - step_index: which block is the current noisy subset (0-based)
    Zones:
      clean:   [0:k)
      current: [k:k+cur)
      future:  [k+cur:L)
    Rules:
      clean->clean: lower-tri causal
      clean->current/future: masked
      current->clean: fully visible; current->current: causal; current->future: masked
      future rows: fully masked (do not affect others)
    Return: [1,1,L,L] bool mask (True=masked)
    """
    L = sample_len
    k = cumsum[step_index]
    cur = split_sizes[step_index]
    mask = torch.ones((L, L), dtype=torch.bool)
    # clean rows
    for i in range(0, k):
        mask[i, : i + 1] = False
    # current rows
    for i in range(k, k + cur):
        mask[i, :k] = False
        mask[i, k : i + 1] = False
    # future rows: keep masked except allow self to avoid all-masked rows
    for i in range(k + cur, L):
        mask[i, i] = False
    return mask[None, None, :, :]
