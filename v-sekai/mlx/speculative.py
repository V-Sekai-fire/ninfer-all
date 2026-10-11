"""Greedy MTP speculative decoding for a .ninfer model loaded with ninfer_mlx.load(draft=True).

One round: the MTP head drafts k tokens through the proposal shortlist, the target verifies the
current token plus the k drafts in one forward pass, and the longest prefix whose drafts equal the
target's argmax is kept with the target's next token. Full-attention KV is trimmed. GatedDeltaNet
state cannot be trimmed, so verification records each GDN layer's conv input and recurrence inputs
and a partial accept replays only the recurrence over the kept tokens (ninfer's ReplaySSM idea).
The MTP KV is cut back to the committed context and rebuilt from the target's hidden states, as in
docs/maintainer/qwen3_5-model.md "Prefill, decode and MTP".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

_TRACE: list | None = None


def _traced_gdn_call(self, inputs, mask=None, cache=None):
    """mlx-lm's GatedDeltaNet.__call__ plus a record of what a rollback needs to replay."""
    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = self.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    state = cache[1] if cache else None
    q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
    if _TRACE is not None and cache is not None:
        _TRACE.append((self, cache, conv_input, q, k, v, a, b, state))
    out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
                                    use_kernel=not self.training)
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


qwen3_5.GatedDeltaNet.__call__ = _traced_gdn_call


def _rollback_gdn(trace: list, keep: int) -> None:
    """Restore every traced GDN layer to its state after the first `keep` verified tokens."""
    for layer, cache, conv_input, q, k, v, a, b, state in trace:
        width = layer.conv_kernel_size - 1
        cache[0] = mx.contiguous(conv_input[:, keep:keep + width, :])
        _, cache[1] = gated_delta_update(q[:, :keep], k[:, :keep], v[:, :keep], a[:, :keep],
                                         b[:, :keep], layer.A_log, layer.dt_bias, state,
                                         use_kernel=True)


@dataclass
class Stats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    tokens: int = 0
    ttft: float = 0.0
    decode_seconds: float = 0.0
    per_round: list = field(default_factory=list)

    @property
    def acceptance(self) -> float:
        return self.accepted / max(self.drafted, 1)

    @property
    def tokens_per_round(self) -> float:
        return sum(self.per_round) / max(len(self.per_round), 1)


class MTPDecoder:
    def __init__(self, model, drafts: int = 3) -> None:
        self.model = model
        self.lm = model.language_model
        self.drafter = model.__dict__["_drafter"]
        self.drafts = drafts

    def _target(self, ids: mx.array, cache) -> tuple[mx.array, mx.array]:
        hidden = self.lm.model(ids, cache=cache)
        return hidden, self.lm.lm_head(hidden)

    def _mtp(self, tokens: mx.array, hidden: mx.array) -> mx.array:
        embedding = self.lm.model.embed_tokens(tokens)
        return self.drafter.mtp(embedding, hidden, self.mtp_cache)

    def _propose(self, mtp_hidden: mx.array) -> mx.array:
        rows = mx.argmax(self.drafter.proposal(mtp_hidden[:, -1:, :]), axis=-1)
        return self.drafter.token_ids[rows]

    def _draft(self, tokens: mx.array, hidden: mx.array) -> list[mx.array]:
        """Bridge the committed tokens into the MTP KV, then draft recursively."""
        m = self._mtp(tokens, hidden)
        drafts = [self._propose(m)]
        for _ in range(self.drafts - 1):
            m = self._mtp(drafts[-1], m[:, -1:, :])
            drafts.append(self._propose(m))
        return drafts

    def generate(self, prompt: list[int], max_tokens: int, stop: set[int] = frozenset()):
        """Yield token ids; greedy, so the stream equals plain greedy decoding of the target."""
        global _TRACE
        stats = self.stats = Stats()
        start = time.perf_counter()
        cache = make_prompt_cache(self.model)
        self.mtp_cache = KVCache()
        ids = mx.array(prompt)[None]
        hidden, logits = self._target(ids, cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        stats.ttft = time.perf_counter() - start
        produced = 1
        yield token.item()
        if token.item() in stop:
            return
        # MTP prefill: tokens shift by one, hidden states and positions do not.
        bridge_tokens = mx.concatenate([ids[:, 1:], token[:, None]], axis=1)
        bridge_hidden = hidden
        decode_start = time.perf_counter()
        while produced < max_tokens:
            committed = self.mtp_cache.offset + bridge_tokens.shape[1]
            drafts = self._draft(bridge_tokens, bridge_hidden)
            self.mtp_cache.trim(self.mtp_cache.offset - committed)
            draft = mx.concatenate(drafts, axis=0).reshape(1, -1)
            verify = mx.concatenate([token[:, None], draft], axis=1)
            _TRACE = []
            try:
                v_hidden, v_logits = self._target(verify, cache)
                target = mx.argmax(v_logits, axis=-1)
                mx.eval(target, draft)
            finally:
                trace, _TRACE = _TRACE, None
            t, d = target[0].tolist(), draft[0].tolist()
            accepted = 0
            while accepted < len(d) and d[accepted] == t[accepted]:
                accepted += 1
            keep = accepted + 1
            if keep < verify.shape[1]:
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(verify.shape[1] - keep)
                _rollback_gdn(trace, keep)
            new = d[:accepted] + [t[accepted]]
            stats.rounds += 1
            stats.drafted += len(d)
            stats.accepted += accepted
            stats.per_round.append(len(new))
            for tok in new:
                produced += 1
                yield tok
                if tok in stop or produced >= max_tokens:
                    stats.tokens = produced
                    stats.decode_seconds = time.perf_counter() - decode_start
                    return
            token = mx.array([new[-1]])
            bridge_tokens = mx.array(new)[None]
            bridge_hidden = v_hidden[:, :keep, :]
        stats.tokens = produced
        stats.decode_seconds = time.perf_counter() - decode_start
