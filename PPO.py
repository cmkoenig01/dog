import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import Actor, Critic

LR                = 3e-4
GAMMA             = 0.99
LAM               = 0.95
CLIP_EPS          = 0.2
EPOCHS            = 4
BATCH_SIZE        = 512
ENTROPY_COEF      = 0.02
STEPS_PER_UPDATE  = 2048

_mse = nn.MSELoss()

# Stores experience tuples collected during a rollout before a policy update.
class RolloutBuffer:
    def __init__(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.log_probs = []
        self.values    = []
        self.dones     = []

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=256, critic_hidden_dim=512):
        self.actor  = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, critic_hidden_dim)

        # log_std is learnable so the policy can adjust exploration over time.
        # Start at -1 (std ≈ 0.37) so early actions are small and conservative.
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

        self.actor_optimizer  = optim.Adam(
            list(self.actor.parameters()) + [self.log_std], lr=LR
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)

        self.buffer = RolloutBuffer()

    # Samples an action from the current policy distribution given a state.
    @torch.no_grad()
    def get_action(self, state, deterministic=False):
        state_t  = torch.FloatTensor(state).unsqueeze(0)
        mean     = self.actor(state_t)
        std      = self.log_std.exp().clamp(0.01, 0.5)
        dist     = torch.distributions.Normal(mean, std)
        action   = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value    = self.critic(state_t)
        return (
            action.numpy().flatten(),
            log_prob.item(),
            value.item()
        )

    # Batched forward pass for N environments simultaneously.
    # Returns numpy arrays of shape (N, action_dim), (N,), (N,).
    @torch.no_grad()
    def get_actions_batch(self, states):
        states_t  = torch.FloatTensor(states)
        means     = self.actor(states_t)
        std       = self.log_std.exp().clamp(0.01, 0.5)
        dist      = torch.distributions.Normal(means, std)
        actions   = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values    = self.critic(states_t).squeeze(-1)
        return (
            actions.numpy(),
            log_probs.numpy(),
            values.numpy(),
        )

    def _gae_for_buffer(self, buf, next_value):
        advantages = []
        gae        = 0
        values     = buf.values + [next_value]
        for t in reversed(range(len(buf.rewards))):
            delta = (buf.rewards[t]
                     + GAMMA * values[t + 1] * (1 - buf.dones[t])
                     - values[t])
            gae   = delta + GAMMA * LAM * (1 - buf.dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    # Computes Generalized Advantage Estimation (GAE) over the collected rollout.
    def compute_advantages(self, next_value):
        return self._gae_for_buffer(self.buffer, next_value)

    def _ppo_mini_batch_update(self, states, actions, old_lp, old_vals, advantages, returns):
        n = len(states)
        for _ in range(EPOCHS):
            indices = torch.randperm(n)
            for start in range(0, n, BATCH_SIZE):
                idx           = indices[start:start + BATCH_SIZE]
                s, a, lp      = states[idx], actions[idx], old_lp[idx]
                adv, ret      = advantages[idx], returns[idx]
                old_v         = old_vals[idx]

                mean     = self.actor(s)
                std      = self.log_std.exp().clamp(0.01, 0.5)
                dist     = torch.distributions.Normal(mean, std)
                new_lp   = dist.log_prob(a).sum(dim=-1)
                new_vals = self.critic(s).squeeze()

                ratio       = torch.exp(new_lp - lp)
                clipped     = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                actor_loss  = (-torch.min(ratio * adv, clipped * adv).mean()
                               - ENTROPY_COEF * dist.entropy().mean())

                val_clipped = old_v + torch.clamp(new_vals - old_v, -CLIP_EPS, CLIP_EPS)
                critic_loss = torch.max(
                    _mse(new_vals, ret),
                    _mse(val_clipped, ret),
                )

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + [self.log_std], 0.5
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

    # Runs multiple epochs of PPO updates using random mini-batches from the rollout.
    # Mini-batching improves stability versus updating on the full buffer at once.
    # Entropy bonus prevents policy collapse; gradient clipping keeps updates stable.
    def update(self, next_value):
        advantages = self.compute_advantages(next_value)
        states     = torch.FloatTensor(np.array(self.buffer.states))
        actions    = torch.FloatTensor(np.array(self.buffer.actions))
        old_lp     = torch.FloatTensor(self.buffer.log_probs)
        old_vals   = torch.FloatTensor(self.buffer.values)
        advantages = torch.FloatTensor(advantages)
        returns    = advantages + old_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self._ppo_mini_batch_update(states, actions, old_lp, old_vals, advantages, returns)
        self.buffer.clear()

    # Multi-env update: each buffer gets its own GAE, then all transitions are merged
    # into one mini-batch pool so the policy sees diverse experience each update.
    def update_multi(self, buffers, next_values):
        all_states, all_actions, all_lp, all_vals, all_adv, all_ret = [], [], [], [], [], []
        for buf, nv in zip(buffers, next_values):
            old_v = torch.FloatTensor(buf.values)
            adv   = torch.FloatTensor(self._gae_for_buffer(buf, nv))
            ret   = adv + old_v
            all_states.append(torch.FloatTensor(np.array(buf.states)))
            all_actions.append(torch.FloatTensor(np.array(buf.actions)))
            all_lp.append(torch.FloatTensor(buf.log_probs))
            all_vals.append(old_v)
            all_adv.append(adv)
            all_ret.append(ret)
            buf.clear()

        states     = torch.cat(all_states)
        actions    = torch.cat(all_actions)
        old_lp     = torch.cat(all_lp)
        old_vals   = torch.cat(all_vals)
        advantages = torch.cat(all_adv)
        returns    = torch.cat(all_ret)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self._ppo_mini_batch_update(states, actions, old_lp, old_vals, advantages, returns)

    def save(self, path="checkpoint.pt", best_reward=None):
        data = {
            "actor":   self.actor.state_dict(),
            "critic":  self.critic.state_dict(),
            "log_std": self.log_std.data,
        }
        if best_reward is not None:
            data["best_reward"] = best_reward
        torch.save(data, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path="checkpoint.pt"):
        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "log_std" in checkpoint:
            self.log_std.data = checkpoint["log_std"]
        print(f"Loaded checkpoint from {path}")
