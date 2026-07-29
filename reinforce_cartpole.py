"""
REINFORCE on CartPole-v1 — vanilla policy gradient, from scratch.

    pip install torch gymnasium matplotlib
    python reinforce_cartpole.py

The whole algorithm is the `update` function. Everything else is plumbing.
Typically solves (mean return > 475 over 100 episodes) in 200-500 episodes.
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------- config
HIDDEN     = 24
LR         = 0.012
GAMMA      = 0.99
BATCH      = 5        # episodes per gradient update
MAX_EPISODES = 1500
SEED       = 0
USE_BASELINE = True   # flip to False to see what the baseline buys you

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------- policy
class Policy(nn.Module):
    """Maps a 4-d state to a distribution over {push left, push right}."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )
        # Small final layer => near-uniform initial policy => real exploration.
        nn.init.normal_(self.net[2].weight, std=0.1)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, state):
        return torch.distributions.Categorical(logits=self.net(state))


# CartPole's four observations live on very different scales; theta is ~0.05
# while x_dot can be ~2. Rescaling stops theta from being drowned out.
OBS_SCALE = torch.tensor([1 / 2.4, 0.5, 1 / 0.2095, 0.4])


def discounted_returns(rewards):
    """G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ..., computed backwards."""
    out, running = [], 0.0
    for r in reversed(rewards):
        running = r + GAMMA * running
        out.append(running)
    return torch.tensor(out[::-1], dtype=torch.float32)


def run_episode(env, policy):
    """Play one episode; return stacked log-probs and the per-step returns."""
    state, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        s = torch.as_tensor(state, dtype=torch.float32) * OBS_SCALE
        dist = policy(s)
        action = dist.sample()
        # log pi(a|s) — the quantity we differentiate. Keep the graph attached.
        log_probs.append(dist.log_prob(action))
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
    return torch.stack(log_probs), discounted_returns(rewards), len(rewards)


def update(optimizer, batch):
    """
    The policy gradient step.

        grad J = E[ A(s,a) * grad log pi(a|s) ]

    We maximise, but optimizers minimise, so the loss is the negative.
    """
    log_probs = torch.cat([lp for lp, _ in batch])
    returns   = torch.cat([g for _, g in batch])

    # Baseline: centre the returns on the batch mean. Dividing by std is
    # separate — it just keeps the gradient scale sane — so toggling
    # USE_BASELINE isolates the mean subtraction on its own.
    std = returns.std() + 1e-8
    advantages = (returns - returns.mean()) / std if USE_BASELINE else returns / std

    loss = -(log_probs * advantages).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ----------------------------------------------------------------- training
def main():
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    history, batch = [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    scatter, = ax.plot([], [], ".", ms=2.5, color="0.7", label="episode")
    trace,   = ax.plot([], [], "-", lw=1.8, color="#17458F", label="mean of last 20")
    ax.axhline(475, ls="--", lw=1, color="#2E6B4F", label="solved")
    ax.set_xlabel("episode"); ax.set_ylabel("return"); ax.set_ylim(0, 520)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("REINFORCE on CartPole-v1")

    for ep in range(1, MAX_EPISODES + 1):
        log_probs, returns, length = run_episode(env, policy)
        batch.append((log_probs, returns))
        history.append(length)

        if len(batch) == BATCH:
            update(optimizer, batch)
            batch = []

        if ep % 10 == 0:
            recent = np.mean(history[-20:])
            print(f"episode {ep:5d}   last {length:4d}   mean(20) {recent:6.1f}")

            xs = np.arange(len(history))
            ma = np.convolve(history, np.ones(20) / 20, mode="valid")
            scatter.set_data(xs, history)
            trace.set_data(np.arange(19, len(history)), ma)
            ax.set_xlim(0, max(100, len(history)))
            fig.canvas.draw(); fig.canvas.flush_events()
            plt.pause(0.001)

        if len(history) >= 100 and np.mean(history[-100:]) >= 475:
            print(f"\nSolved at episode {ep} "
                  f"(mean return {np.mean(history[-100:]):.1f} over last 100).")
            break

    env.close()
    plt.ioff(); plt.show()


if __name__ == "__main__":
    main()
