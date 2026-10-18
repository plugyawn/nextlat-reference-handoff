"""One on-policy, gold-prefix conditional Countdown group; no replay buffer."""
from collections import Counter
import time

import torch

from search.countdown_task import CountdownTask, verify_expression


PROTOCOL = "optimizer_state_preserving_countdown_recurrent_pg_v1"
POLICY = {"temperature": 0.7, "top_k": 50, "samples": 8,
          "max_tokens": 16, "suffix_tokens": 8, "lambda_pg": 1.0,
          "updates": 128}
SCALED_POLICY = {**POLICY, "samples": 32, "updates": 4096,
                 "control_auxiliary": "first_group_only"}


def centered_advantages(rewards):
    group = rewards.numel()
    if group not in (8, 32):
        raise RuntimeError("Countdown supports only registered G8/G32 groups")
    return (rewards - rewards.mean()).detach() * (group / (group - 1))


def phase_token_budget(dataset, seed, stream_batches, remaining_updates):
    """Exact native B1x8 corpus budget, preserving the current sampler cursor."""
    from torch.utils.data import DistributedSampler
    if stream_batches < 0 or not 0 <= remaining_updates <= 4096 or len(dataset) == 0:
        raise RuntimeError("invalid Countdown policy-phase sampler clock")
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=seed,
                                 shuffle=True, drop_last=True)
    epoch, offset = divmod(stream_batches, len(sampler))
    remaining, tokens = remaining_updates * 8, 0
    while remaining:
        sampler.set_epoch(epoch)
        indices = list(sampler)[offset:offset + remaining]
        tokens += sum(int(dataset[index]["token_count"]) for index in indices)
        remaining -= len(indices)
        epoch, offset = epoch + 1, 0
    return tokens


def policy_distribution(core, states, vocab_size):
    logits = core.model.lm_head(states).float()[:, :vocab_size] / 0.7
    values, indices = logits.topk(50, dim=-1)
    return values.log_softmax(-1), indices


def conditional_group_backward(core, tokenizer, row, *, audit_gradients, lambda_pg, samples=8):
    """Sample once, replay the same actions, and accumulate PG into native grads."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    if samples not in (8, 32):
        raise RuntimeError("Countdown supports only registered G8/G32 groups")
    if (core.loss_config.anchor_noise_relative != 0
        or any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in core.modules())
        or float(getattr(core.model.config, "attention_dropout", 0)) != 0
        or any(not p.requires_grad for p in core.parameters())):
        raise RuntimeError("conditional PG requires trainable theta/psi and zero dropout/noise")
    boundary = int(row["expression_stop"]) - POLICY["suffix_tokens"]
    if boundary < int(row["expression_start"]):
        raise RuntimeError("conditional PG record is shorter than the registered suffix")
    device = core.model.lm_head.weight.device
    prefix = torch.tensor([row["input_ids"][:boundary]], device=device)
    embed = core.model.get_input_embeddings()
    anchor = core.model.model(inputs_embeds=embed(prefix), use_cache=False,
                              return_dict=True).last_hidden_state[:, -1]
    raw_eos = core.model.generation_config.eos_token_id
    eos_ids = {raw_eos} if isinstance(raw_eos, int) else set(raw_eos)
    if eos_ids != {151643, 151645} or len(tokenizer) > core.model.lm_head.weight.shape[0]:
        raise RuntimeError("conditional PG EOS/vocabulary policy differs from Countdown probe")
    actions, old_logps, masks = [], [], []
    alive = torch.ones(samples, dtype=torch.bool, device=device)
    # Sampling must not cache detached BF16 parameter casts for grad replay.
    # Replay uses the outer grad-enabled cache, sharing one head/MLP cast across
    # recurrent steps instead of retaining a fresh vocabulary matrix per step.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
        states = anchor.detach().expand(samples, -1)
        for step in range(16):
            lp, support = policy_distribution(core, states, len(tokenizer))
            choice = torch.multinomial(lp.exp(), 1)
            token = support.gather(1, choice).squeeze(1)
            actions.append(token)
            old_logps.append(lp.gather(1, choice).squeeze(1))
            masks.append(alive.clone())
            ended = torch.zeros_like(alive)
            for eos in eos_ids:
                ended |= token == eos
            alive &= ~ended
            if not bool(alive.any()) or step == 15:
                break
            states = core.dynamics(states, embed(token))
    tokens = torch.stack(actions, 1)
    mask = torch.stack(masks, 1)
    sequence_logp = torch.zeros(samples, device=device)
    states = anchor.expand(samples, -1)
    maximum_error = torch.zeros((), device=device)
    for step in range(tokens.shape[1]):
        lp, support = policy_distribution(core, states, len(tokenizer))
        matches = support == tokens[:, step, None]
        if not bool((matches.any(-1) | ~mask[:, step]).all()):
            raise RuntimeError("conditional PG action left replay top-k support")
        chosen = lp.gather(1, matches.long().argmax(-1, keepdim=True)).squeeze(1)
        error = (chosen.detach() - old_logps[step]).abs()[mask[:, step]]
        maximum_error = torch.maximum(maximum_error, error.max())
        sequence_logp = sequence_logp + torch.where(mask[:, step], chosen, 0)
        if step + 1 < tokens.shape[1]:
            states = core.dynamics(states, embed(tokens[:, step]))
    if not bool(torch.isfinite(sequence_logp).all()) or float(maximum_error) > 1e-5:
        raise RuntimeError(f"conditional PG sample/replay logp mismatch: {float(maximum_error)}")
    rewards, lengths, failures = [], [], Counter()
    task = CountdownTask(tuple(row["inputs"]), row["target"], row["expression"], 0)
    expression_prefix = row["input_ids"][row["expression_start"]:boundary]
    for branch in range(samples):
        length = int(mask[branch].sum())
        sampled = tokens[branch, :length].tolist()
        terminated = sampled[-1] in eos_ids
        text = tokenizer.decode(expression_prefix + (sampled[:-1] if terminated else sampled),
                                skip_special_tokens=False)
        result = verify_expression(text, task)
        reward = int(terminated and result["passed"])
        rewards.append(reward)
        lengths.append(length)
        if not reward:
            failures["length_cap" if not terminated else result["failure_code"]] += 1
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = centered_advantages(reward_tensor)
    loss = -(advantages * sequence_logp).sum() / (samples * 16)
    mixed = 0 < sum(rewards) < samples
    grad_metrics = {}
    if audit_gradients and mixed:
        layers = core.model.model.layers
        probes = {"interior_theta": layers[len(layers) // 2].self_attn.q_proj.weight,
                  "psi": core.dynamics.mlp[-1].weight,
                  "embedding_head": core.model.lm_head.weight}
        grads = torch.autograd.grad(loss, tuple(probes.values()), retain_graph=True)
        for name, grad in zip(probes, grads):
            norm = float(grad.detach().float().norm())
            if not torch.isfinite(grad).all() or not norm > 0:
                raise RuntimeError(f"conditional PG-only gradient invalid: {name} {norm}")
            grad_metrics[f"pg_only_grad_norm_{name}"] = norm
        del grads
    if not mixed and (bool(advantages.any()) or float(loss.detach()) != 0):
        raise RuntimeError("equal-reward group must have exactly zero PG loss")
    (lambda_pg * loss).backward()
    dense_steps = tokens.shape[1]
    backward_passes = 1 + int(bool(grad_metrics))
    return {"pg_unweighted_loss": float(loss.detach()), "pg_loss": float(loss.detach()) * lambda_pg,
            "pg_lambda": lambda_pg, "pg_gradient_applied": lambda_pg != 0 and mixed,
            "pg_rewards": rewards,
            "pg_lengths": lengths, "pg_reward_mean": sum(rewards) / samples,
            "pg_group_size": samples,
            "pg_mixed_group": mixed, "pg_equal_group_zero": not mixed,
            "pg_first_mixed_audited": bool(grad_metrics),
            "pg_sampled_positions": sum(lengths), "pg_logp_replay_max_error": float(maximum_error),
            "pg_sampling_head_positions": samples * dense_steps,
            "pg_sampling_transition_positions": samples * (dense_steps - 1),
            "pg_replay_head_positions": samples * dense_steps,
            "pg_replay_transition_positions": samples * (dense_steps - 1),
            "pg_replay_backward_head_positions": samples * dense_steps * backward_passes,
            "pg_replay_backward_transition_positions": samples * (dense_steps - 1) * backward_passes,
            "pg_exact_anchor_forward_positions": prefix.numel(),
            "pg_exact_anchor_backward_positions": prefix.numel() * backward_passes,
            "pg_backward_passes": backward_passes,
            "pg_verifier_failures": dict(failures), "pg_seconds": time.perf_counter() - started,
            "pg_problem_hash": row["problem_hash"], **grad_metrics}
