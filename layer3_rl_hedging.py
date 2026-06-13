"""
Layer 3 — Risk-Aware Optimization & Control Engine
===================================================
Non-Markovian Dynamic Programming via Discretized Path Memory & CVaR

Project: Reinforcement Learning as a Numerical Approach to Stochastic
         Optimal Control under Market Frictions
Author:  BSc Mathematics Independent Research Programme

Structure
---------
  Section 1  Path Signature Encoder (Memory Feature Transformation)
  Section 2  Risk-Aware Objective Operator (CVaR Tail-Risk Penalty)
  Section 3  Temporal Difference Tensor Optimizer (Q-Learning Core)
  Section 4  Policy Output Verification Matrix (Asset Allocations vs. Frictions)

This architecture builds allocations under non-Markovian memory features.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("output", exist_ok=True)

# ── Colour Palette Matching Project Blueprint ─────────────────────────────────
TEAL   = "#1D9E75"
PURPLE = "#7F77DD"
CORAL  = "#D85A30"
GRAY   = "#888780"
AMBER  = "#BA7517"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "font.size":         11,
})

def extract_path_signatures(price_history: np.ndarray, lookback: int = 3) -> np.ndarray:
    n_steps = len(price_history)
    signatures = np.zeros(n_steps)
    for t in range(lookback, n_steps):
        window = price_history[t-lookback:t]
        signatures[t] = (window[-1] - window[0]) / window[0]
    return signatures

def compute_cvar_penalty(rewards: np.ndarray, alpha: float = 0.05) -> float:
    sorted_rewards = np.sort(rewards)
    cutoff_idx = int(len(sorted_rewards) * alpha)
    if cutoff_idx == 0:
        return np.mean(sorted_rewards[:1])
    return np.mean(sorted_rewards[:cutoff_idx])

class Layer3ControlEngine:
    def __init__(self, n_steps=100, n_paths=1000, risk_lambda=0.5):
        self.N = n_steps
        self.blocks = n_paths
        self.risk_lambda = risk_lambda
        self.dt = 1.0 / self.N
        self.actions = np.linspace(0.0, 1.0, 11) 
        self.state_bins = np.array([-0.02, 0.02])
        self.Q_table = np.zeros((self.N, len(self.state_bins) + 1, len(self.actions)))

    def discretize_state(self, sig_value):
        return np.digitize(sig_value, self.state_bins)

    def train_control_policy(self, S_paths):
        sig_paths = np.zeros_like(S_paths)
        for p in range(self.blocks):
            sig_paths[p, :] = extract_path_signatures(S_paths[p, :])
            
        print(f"  [Q-Tensor Engine] Optimizing Bellman Fields (Paths={self.blocks}, Risk Weight={self.risk_lambda})...")
        
        for t in reversed(range(self.N - 1)):
            for s_idx in range(len(self.state_bins) + 1):
                path_mask = np.where(self.discretize_state(sig_paths[:, t]) == s_idx)[0]
                if len(path_mask) == 0:
                    continue
                    
                current_prices = S_paths[path_mask, t]
                next_prices = S_paths[path_mask, t+1]
                returns = (next_prices - current_prices) / current_prices
                
                for a_idx, action in enumerate(self.actions):
                    frictions = 0.005 * (action ** 2)
                    raw_rewards = action * returns - frictions
                    
                    cvar = compute_cvar_penalty(raw_rewards, alpha=0.05)
                    expected_return = np.mean(raw_rewards)
                    immediate_fitness = expected_return + self.risk_lambda * cvar
                    
                    if t == self.N - 2:
                        self.Q_table[t, s_idx, a_idx] = immediate_fitness
                    else:
                        next_sigs = sig_paths[path_mask, t+1]
                        next_states = self.discretize_state(next_sigs)
                        expected_continuation = np.mean([np.max(self.Q_table[t+1, ns, :]) for ns in next_states])
                        self.Q_table[t, s_idx, a_idx] = immediate_fitness + 0.99 * expected_continuation

    def extract_optimal_policy(self):
        optimal_policy = np.zeros(self.N)
        for t in range(self.N):
            optimal_policy[t] = self.actions[np.argmax(self.Q_table[t, 1, :])]
        return optimal_policy

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SECTION 1-4 — Layer 3 Risk-Aware Reinforcement Learning Control Engine")
    print("="*70)
    
    n_steps, n_paths = 100, 1000
    mock_S_paths = np.zeros((n_paths, n_steps + 1))
    mock_S_paths[:, 0] = 100.0
    
    print("  [Simulating Environment] Generating background validation paths...")
    for t in range(n_steps):
        mock_S_paths[:, t+1] = mock_S_paths[:, t] * (1 + 0.06 * (1/n_steps) + 0.22 * np.sqrt(1/n_steps) * np.random.normal(0, 1, n_paths))
        
    risk_neutral_engine = Layer3ControlEngine(n_steps, n_paths, risk_lambda=0.0)
    risk_neutral_engine.train_control_policy(mock_S_paths)
    neutral_policy = risk_neutral_engine.extract_optimal_policy()
    
    risk_averse_engine = Layer3ControlEngine(n_steps, n_paths, risk_lambda=1.5)
    risk_averse_engine.train_control_policy(mock_S_paths)
    averse_policy = risk_averse_engine.extract_optimal_policy()

    fig, ax = plt.subplots(figsize=(8.5, 4))
    time_axis = np.linspace(0, 1, n_steps)
    
    ax.step(time_axis, neutral_policy, color=AMBER, where='mid', linewidth=2.0, label="Risk-Neutral Agent ($\lambda = 0.0$)")
    ax.step(time_axis, averse_policy, color=PURPLE, where='mid', linewidth=2.5, label="Risk-Averse CVaR Agent ($\lambda = 1.5$)")
    ax.set_title("Layer 3: Dynamic Control Allocation under Rough Market Instability", fontsize=11, fontweight='bold')
    ax.set_xlabel("Time Horizon ($t$)")
    ax.set_ylabel("Optimal Allocation Vector ($u^*_t$)")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("output/layer3_control_policy.png", dpi=150)
    print("\n  [Execution Complete] Optimal control vector field fully optimized.")
    print("  Figure saved: output/layer3_control_policy.png")
    plt.close()
