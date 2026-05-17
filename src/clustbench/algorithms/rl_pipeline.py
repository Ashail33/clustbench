"""Stage-2 of Framing C — behavior-cloned RL policy + value net wrapped as a
clustbench algorithm.

This module pairs with :mod:`clustbench.rl_env`. Stage 1 supplies the env, the
8-primitive action ontology and the ``state_features`` extraction. Stage 2
trains a tiny PolicyNet/ValueNet pair via behaviour cloning on the parquet
traces in ``runs/rl_traces.parquet`` and exposes a ``Rl_pipeline`` algorithm
that *executes* the learned policy through the env.

Design notes
------------
* PyTorch is intentionally lazy-imported. The clustbench algorithm registry
  is imported eagerly by the CLI, so the import cost is deferred until
  someone actually trains or runs the policy.
* The trace parquet only contains ~419 rows from 50 episodes. The networks
  are deliberately small (~6k parameters) so overfit risk is bounded and CPU
  training finishes in seconds.
* Inference: at each env step we mask the policy's softmax to the legal
  actions returned by ``env.action_space(state)``, pick the highest-scoring
  action, then use the value net to disambiguate between parameter variants
  of the chosen action (most importantly for MEDOID_SWAP whose candidate set
  is sampled fresh on each call).
* ``rl_pipeline`` is blocked from sub-routing (``_BLOCKED_TARGETS``) so any
  meta-router that scans the registry won't try to dispatch back into us and
  recurse.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..rl_env import (
    Action,
    ClusteringEnv,
    ClusteringState,
    N_STATE_FEATURES,
)
from .base import Algorithm, AlgoResult, Step, register


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stable ordering for the policy head — index i corresponds to ``ACTION_LIST[i]``.
# Frozen at module import; do not reorder without retraining the checkpoint.
ACTION_LIST: List[Action] = list(Action)
N_ACTIONS = len(ACTION_LIST)
ACTION_TO_IDX: Dict[str, int] = {a.value: i for i, a in enumerate(ACTION_LIST)}

POLICY_CKPT = Path("runs/rl_policy.pt")
VALUE_CKPT = Path("runs/rl_value.pt")
DEFAULT_TRACES = Path("runs/rl_traces.parquet")

# Block recursion: meta-routers that scan the registry must skip rl_pipeline.
_BLOCKED_TARGETS = {"rl_pipeline"}


# ---------------------------------------------------------------------------
# Policy / Value networks
# ---------------------------------------------------------------------------


def _build_policy_net(in_dim: int = N_STATE_FEATURES, n_actions: int = N_ACTIONS, hidden: int = 64):
    """state_features -> logits over the 8 primitive actions.

    Returned eagerly (not as a class) because the torch import is deferred.
    Defining ``nn.Module`` subclasses at module top level would force torch
    to load at registry import time, which is what we want to avoid.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )


def _build_value_net(in_dim: int = N_STATE_FEATURES, hidden: int = 64):
    """state_features -> scalar predicted return-to-go.

    The final layer is a single linear unit; the wrapper squeezes the last
    dim so callers see a 1-D tensor.
    """
    import torch.nn as nn

    class _ValueHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):  # type: ignore[override]
            return self.net(x).squeeze(-1)

    return _ValueHead()


# ---------------------------------------------------------------------------
# Behavior-cloning trainer
# ---------------------------------------------------------------------------


def _parse_state_features(raw: Any) -> np.ndarray:
    """Parquet round-trips lists either as numpy arrays or python lists.

    Normalise both paths to a length-``N_STATE_FEATURES`` float32 vector. Any
    NaNs or infs collapse to 0 — the policy can't reason about them.
    """
    arr = np.asarray(list(raw) if not isinstance(raw, np.ndarray) else raw, dtype=np.float32)
    if arr.shape[0] != N_STATE_FEATURES:
        # Defensive pad/truncate so a schema change doesn't crash training.
        out = np.zeros(N_STATE_FEATURES, dtype=np.float32)
        n = min(arr.shape[0], N_STATE_FEATURES)
        out[:n] = arr[:n]
        arr = out
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_returns_to_go(rewards: np.ndarray, gamma: float = 0.95) -> np.ndarray:
    """Discounted sum of *future* rewards inside one episode.

    Walks rewards backwards so ``out[t] = r[t] + gamma * out[t+1]``. Matches
    the convention used by every actor-critic implementation in the wild.
    """
    out = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def train_bc(
    traces_path: str | Path = DEFAULT_TRACES,
    n_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    gamma: float = 0.95,
    val_frac: float = 0.2,
    patience: int = 30,
    out_policy: str | Path = POLICY_CKPT,
    out_value: str | Path = VALUE_CKPT,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train PolicyNet (cross-entropy on actions) and ValueNet (MSE on
    return-to-go) on a behaviour-cloning parquet.

    Splits *by episode id* — train and val episodes never share rows — so
    val accuracy is a fair generalisation signal.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    traces_path = Path(traces_path)
    if not traces_path.exists():
        raise FileNotFoundError(
            f"Trace parquet {traces_path} not found. "
            "Generate it via clustbench.rl_env.collect_traces_from_existing_algorithms()."
        )

    import pandas as pd

    df = pd.read_parquet(traces_path)
    if df.empty:
        raise ValueError(f"Trace parquet {traces_path} has zero rows; nothing to train.")

    # ------------------------------------------------------------------
    # Vectorise state features, action targets, returns-to-go.
    # ------------------------------------------------------------------
    feats = np.stack([_parse_state_features(s) for s in df["state_features"].values], axis=0)
    actions = np.array(
        [ACTION_TO_IDX.get(str(a), 0) for a in df["action"].values], dtype=np.int64
    )

    # Returns-to-go are computed *per episode* to respect episode boundaries.
    rtg = np.zeros(len(df), dtype=np.float64)
    for ep_id, group in df.groupby("episode_id", sort=False):
        idxs = group.index.to_numpy()
        rewards = group["reward"].to_numpy(dtype=np.float64)
        rtg[idxs] = _compute_returns_to_go(rewards, gamma=gamma)

    # ------------------------------------------------------------------
    # Episode-level train/val split — never split *within* an episode.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    ep_ids = np.array(sorted(df["episode_id"].unique()))
    rng.shuffle(ep_ids)
    n_val = max(1, int(round(val_frac * len(ep_ids))))
    val_eps = set(ep_ids[:n_val].tolist())
    train_eps = set(ep_ids[n_val:].tolist())
    if not train_eps:  # tiny dataset corner case
        train_eps = val_eps

    is_val = df["episode_id"].isin(val_eps).to_numpy()
    is_train = df["episode_id"].isin(train_eps).to_numpy()

    X_train = torch.from_numpy(feats[is_train]).float()
    y_train = torch.from_numpy(actions[is_train]).long()
    r_train = torch.from_numpy(rtg[is_train]).float()
    X_val = torch.from_numpy(feats[is_val]).float()
    y_val = torch.from_numpy(actions[is_val]).long()
    r_val = torch.from_numpy(rtg[is_val]).float()

    torch.manual_seed(seed)
    policy = _build_policy_net()
    value = _build_value_net()
    pol_opt = optim.Adam(policy.parameters(), lr=lr)
    val_opt = optim.Adam(value.parameters(), lr=lr)

    # Mild class re-weighting: invert ``count^0.25`` so rare actions like
    # EIGEN_EMBED (10/419 rows) get a slight boost without overshooting the
    # majority. Empirically, a stronger ``sqrt`` weighting causes the policy
    # to over-predict rare actions at the all-zero step-0 state, which
    # tanks greedy ARI on easy datasets.
    class_counts = np.bincount(actions[is_train], minlength=N_ACTIONS).astype(np.float64)
    class_weights = 1.0 / np.power(np.maximum(class_counts, 1.0), 0.25)
    class_weights = class_weights / class_weights.mean()  # mean 1.0
    ce = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float())
    mse = nn.MSELoss()

    train_pol_losses: List[float] = []
    val_pol_losses: List[float] = []
    train_val_losses: List[float] = []
    val_val_losses: List[float] = []
    val_accs: List[float] = []

    best_val_acc = -1.0
    best_state = None  # (policy_state_dict, value_state_dict)
    stale = 0

    n = X_train.shape[0]
    for epoch in range(n_epochs):
        policy.train()
        value.train()
        perm = torch.randperm(n)
        ep_pol = 0.0
        ep_val = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            rb = r_train[idx]

            pol_opt.zero_grad()
            logits = policy(xb)
            loss_p = ce(logits, yb)
            loss_p.backward()
            pol_opt.step()

            val_opt.zero_grad()
            pred_v = value(xb)
            loss_v = mse(pred_v, rb)
            loss_v.backward()
            val_opt.step()

            ep_pol += float(loss_p.item())
            ep_val += float(loss_v.item())
            n_batches += 1

        ep_pol /= max(1, n_batches)
        ep_val /= max(1, n_batches)
        train_pol_losses.append(ep_pol)
        train_val_losses.append(ep_val)

        # ------------------------------------------------------------------
        # Validation.
        # ------------------------------------------------------------------
        policy.eval()
        value.eval()
        with torch.no_grad():
            v_logits = policy(X_val)
            v_loss_p = float(ce(v_logits, y_val).item()) if X_val.shape[0] else float("nan")
            v_pred = value(X_val)
            v_loss_v = float(mse(v_pred, r_val).item()) if X_val.shape[0] else float("nan")
            v_acc = (
                float((v_logits.argmax(dim=-1) == y_val).float().mean().item())
                if X_val.shape[0]
                else float("nan")
            )
        val_pol_losses.append(v_loss_p)
        val_val_losses.append(v_loss_v)
        val_accs.append(v_acc)

        if verbose:
            print(
                f"epoch {epoch:02d} train_pol={ep_pol:.4f} val_pol={v_loss_p:.4f} "
                f"train_val={ep_val:.4f} val_val={v_loss_v:.4f} val_acc={v_acc:.3f}"
            )

        # Early stopping on val accuracy.
        if v_acc > best_val_acc + 1e-6:
            best_val_acc = v_acc
            best_state = (
                {k: v.detach().clone() for k, v in policy.state_dict().items()},
                {k: v.detach().clone() for k, v in value.state_dict().items()},
            )
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Restore the best (early-stopped) weights before saving.
    if best_state is not None:
        policy.load_state_dict(best_state[0])
        value.load_state_dict(best_state[1])

    out_policy = Path(out_policy)
    out_value = Path(out_value)
    out_policy.parent.mkdir(parents=True, exist_ok=True)
    out_value.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "in_dim": N_STATE_FEATURES,
            "n_actions": N_ACTIONS,
            "hidden": 64,
            "action_list": [a.value for a in ACTION_LIST],
        },
        out_policy,
    )
    torch.save(
        {
            "state_dict": value.state_dict(),
            "in_dim": N_STATE_FEATURES,
            "hidden": 64,
        },
        out_value,
    )

    return {
        "train_pol_losses": train_pol_losses,
        "val_pol_losses": val_pol_losses,
        "train_val_losses": train_val_losses,
        "val_val_losses": val_val_losses,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
        "n_epochs_run": len(train_pol_losses),
        "n_train_rows": int(is_train.sum()),
        "n_val_rows": int(is_val.sum()),
        "policy_path": str(out_policy),
        "value_path": str(out_value),
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _load_checkpoints(
    policy_path: Path = POLICY_CKPT,
    value_path: Path = VALUE_CKPT,
) -> Tuple[Any, Any]:
    """Load PolicyNet + ValueNet from disk; bootstraps via train_bc on miss.

    Self-bootstrapping makes the algorithm runnable out of the box on a
    fresh checkout: the first ``fit_predict`` call notices the missing
    checkpoints and trains them inline before continuing.
    """
    import torch

    if not policy_path.exists() or not value_path.exists():
        # Self-bootstrap. Spec allows up to 60s — 40 epochs on ~419 rows is
        # well under that on CPU.
        train_bc(
            traces_path=DEFAULT_TRACES,
            n_epochs=40,
            out_policy=policy_path,
            out_value=value_path,
        )

    pol_ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
    val_ckpt = torch.load(value_path, map_location="cpu", weights_only=False)

    policy = _build_policy_net(
        in_dim=pol_ckpt.get("in_dim", N_STATE_FEATURES),
        n_actions=pol_ckpt.get("n_actions", N_ACTIONS),
        hidden=pol_ckpt.get("hidden", 64),
    )
    policy.load_state_dict(pol_ckpt["state_dict"])
    policy.eval()

    value = _build_value_net(
        in_dim=val_ckpt.get("in_dim", N_STATE_FEATURES),
        hidden=val_ckpt.get("hidden", 64),
    )
    value.load_state_dict(val_ckpt["state_dict"])
    value.eval()
    return policy, value


def _score_legal_actions(
    policy,
    state_features: np.ndarray,
    legal: List[Tuple[Action, Dict[str, Any]]],
    exploration: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Action, Dict[str, Any], int]:
    """Pick one legal (Action, params) tuple using the policy.

    Strategy:
      1. Forward state_features through the policy to get per-action logits.
      2. Mask out actions not present in ``legal`` so the policy can only
         output a legal head.
      3. Reduce the masked logits to a per-action distribution; greedy if
         ``exploration <= 0``, else categorical sample with temperature.
      4. Among legal entries that match the chosen action, sample one
         uniformly (the value net is used elsewhere to disambiguate
         param variants when relevant).

    Returns ``(action, params, legal_index)`` so the caller can use the
    selected index when filtering MEDOID_SWAP candidates by value-net score.
    """
    import torch

    if not legal:
        raise ValueError("No legal actions available.")
    if rng is None:
        rng = np.random.default_rng(0)

    x = torch.from_numpy(state_features.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logits = policy(x).squeeze(0).cpu().numpy()

    # Action-head mask.
    legal_action_idx = set()
    for a, _ in legal:
        legal_action_idx.add(ACTION_TO_IDX[a.value])

    masked = np.full_like(logits, -1e9)
    for i in legal_action_idx:
        masked[i] = logits[i]

    if exploration <= 0.0:
        chosen_idx = int(np.argmax(masked))
    else:
        # Sample with temperature.
        scaled = masked / max(1e-6, float(exploration))
        scaled = scaled - scaled.max()  # numerical stability
        probs = np.exp(scaled)
        probs[~np.isfinite(probs)] = 0.0
        s = probs.sum()
        if s <= 0:
            chosen_idx = int(np.argmax(masked))
        else:
            probs = probs / s
            chosen_idx = int(rng.choice(len(probs), p=probs))

    chosen_action = ACTION_LIST[chosen_idx]
    matching = [(i, a, p) for i, (a, p) in enumerate(legal) if a == chosen_action]
    if not matching:
        # Fall back to any legal entry (should never happen given mask logic).
        i, a, p = 0, legal[0][0], legal[0][1]
        return a, p, i
    pick_i, pick_a, pick_p = matching[rng.integers(len(matching))]
    return pick_a, pick_p, pick_i


def _disambiguate_params_by_value(
    env: ClusteringEnv,
    value,
    state: ClusteringState,
    candidates: List[Tuple[Action, Dict[str, Any]]],
    rng: np.random.Generator,
    max_eval: int = 6,
) -> Tuple[Action, Dict[str, Any]]:
    """Use the value net to pick the best param variant for a fixed action.

    For ``MEDOID_SWAP`` the env returns up to N candidate ``(out_idx, in_idx)``
    pairs; we score each by simulating the action once and asking the value
    net for the resulting state's predicted return. The highest score wins.
    Actions are cheap (single-action env.step on the centers) so up to
    ``max_eval`` candidates is fine.

    For non-sampled actions there is only one candidate and we just return
    it. ``env`` is passed mutated through this routine but the state is
    immediately restored to the pre-evaluation snapshot.
    """
    import torch

    if not candidates:
        raise ValueError("No candidates")
    if len(candidates) == 1:
        return candidates[0]

    # Subsample if there are too many. Random subset keeps eval bounded.
    if len(candidates) > max_eval:
        idxs = rng.choice(len(candidates), size=max_eval, replace=False)
        eval_pool = [candidates[i] for i in idxs]
    else:
        eval_pool = list(candidates)

    snapshot = env._state  # restored after each trial
    best = None
    best_score = -np.inf
    for a, p in eval_pool:
        try:
            env._state = snapshot
            new_state, reward, _, _ = env.step(a, p)
        except Exception:
            continue
        feats = new_state.to_features()
        x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            v = float(value(x).item())
        score = v + float(reward)  # combine immediate reward with predicted return
        if score > best_score:
            best_score = score
            best = (a, p)
    # Restore state so the caller's env.step has the same precondition as before.
    env._state = snapshot
    if best is None:
        return eval_pool[0]
    return best


# ---------------------------------------------------------------------------
# Algorithm wrapper
# ---------------------------------------------------------------------------


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette wrapper that never raises. Matches the env's convention."""
    from sklearn.metrics import silhouette_score

    try:
        mask = labels != -1
        if mask.sum() < 3:
            return float("nan")
        sub_lab = labels[mask]
        if np.unique(sub_lab).size < 2:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(silhouette_score(X[mask], sub_lab))
    except Exception:
        return float("nan")


@register
class Rl_pipeline(Algorithm):
    """Run a behavior-cloned policy through :class:`ClusteringEnv`.

    Parameters
    ----------
    max_steps
        Hard cap on env steps per rollout. Mirrors the env's own cap.
    exploration
        Softmax temperature for action sampling. ``0.0`` => greedy.
    n_rollouts
        Number of independent rollouts. The labels from the rollout with the
        highest final silhouette are returned.
    random_state
        Seed for env.reset and any temperature-sampling RNGs.
    max_steps_inline_bootstrap
        When the policy checkpoint is missing, train it inline. This knob
        forwards through to ``train_bc(n_epochs=...)`` to allow the caller
        to keep the first-use latency under control.

    Notes
    -----
    * The policy/value checkpoints are loaded lazily inside ``fit_predict``
      so importing this module does not pay the torch cost.
    * If the trained policy's first-step prediction is e.g. ``EIGEN_EMBED``
      on circles-like data, the spectral pathway emerges naturally from the
      BC traces (the parquet contains 10 EIGEN_EMBED rows from the spectral
      builder). Conversely, if BC overfit to assign_to_centers (which is
      the majority class in the parquet at 170/419 rows) the policy may
      stay in a kmeans-style loop regardless of dataset geometry.
    """

    def __init__(
        self,
        max_steps: int = 15,
        exploration: float = 0.0,
        n_rollouts: int = 1,
        random_state: int = 0,
        n_epochs_bootstrap: int = 200,
        **kwargs: Any,
    ) -> None:
        self.name = "rl_pipeline"
        self.max_steps = int(max_steps)
        self.exploration = float(exploration)
        self.n_rollouts = int(n_rollouts)
        self.random_state = int(random_state)
        self.n_epochs_bootstrap = int(n_epochs_bootstrap)
        # Cached policy/value so repeated fit_predict calls are cheap.
        self._policy = None
        self._value = None

    # ------------------------------------------------------------------
    def _ensure_models(self) -> None:
        """Lazy-load the BC checkpoints; train on first miss."""
        if self._policy is not None and self._value is not None:
            return
        if not POLICY_CKPT.exists() or not VALUE_CKPT.exists():
            train_bc(
                traces_path=DEFAULT_TRACES,
                n_epochs=self.n_epochs_bootstrap,
                out_policy=POLICY_CKPT,
                out_value=VALUE_CKPT,
            )
        self._policy, self._value = _load_checkpoints(POLICY_CKPT, VALUE_CKPT)

    # ------------------------------------------------------------------
    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k (number of clusters) must be provided"
        self._ensure_models()

        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        if n == 0:
            return AlgoResult(labels=np.zeros(0, dtype=np.int64), extra={"n_rollouts": 0})

        best_labels: Optional[np.ndarray] = None
        best_sil: float = -np.inf
        best_trajectory: List[Step] = []
        rollouts_summary: List[Dict[str, Any]] = []
        # Collected per-rollout (labels, sil_raw, sil_env, trajectory).
        all_rollouts: List[Dict[str, Any]] = []

        for r_idx in range(max(1, self.n_rollouts)):
            seed = int(self.random_state + 1009 * r_idx)
            rng = np.random.default_rng(seed)
            env = ClusteringEnv(max_steps=max(self.max_steps, 5))
            state = env.reset(X, k_target=int(k), seed=seed)

            trajectory: List[Step] = []
            actions_taken: List[str] = []

            for step_i in range(self.max_steps):
                legal = env.action_space(state)
                if not legal:
                    break
                feats = state.to_features()

                # Pick an action via the policy (masked over legal heads).
                chosen_action, _picked_params, _ = _score_legal_actions(
                    self._policy,
                    feats,
                    legal,
                    exploration=self.exploration,
                    rng=rng,
                )

                # For the chosen action, gather all legal param variants and
                # use the value net to pick the best one (mostly relevant
                # for MEDOID_SWAP whose param grid is sampled each call).
                same_action = [(a, p) for (a, p) in legal if a == chosen_action]
                action, params = _disambiguate_params_by_value(
                    env, self._value, state, same_action, rng
                )

                prev_cost = state.cost
                new_state, reward, done, info = env.step(action, params)
                # Step record.
                trajectory.append(
                    Step(
                        step_idx=step_i,
                        cost=float(new_state.cost),
                        delta_cost=float(new_state.cost - prev_cost),
                        accepted=True,
                        action={
                            "type": action.value,
                            "params": _serialise_params(params),
                            "reward": float(reward),
                        },
                        state={
                            "n_clusters": int(
                                np.unique(new_state.labels[new_state.labels != -1]).size
                            ),
                            "silhouette": float(new_state.silhouette)
                            if np.isfinite(new_state.silhouette)
                            else None,
                        },
                    )
                )
                actions_taken.append(action.value)
                state = new_state
                if done:
                    break

            # Compute both silhouettes; we'll pick across rollouts using a
            # tiered rule that handles convex *and* non-convex data.
            sil_env = _safe_silhouette(state.embedding, state.labels)
            sil_raw = _safe_silhouette(X, state.labels)

            all_rollouts.append(
                {
                    "rollout": r_idx,
                    "labels": state.labels.copy(),
                    "trajectory": trajectory,
                    "sil_raw": float(sil_raw) if np.isfinite(sil_raw) else None,
                    "sil_env": float(sil_env) if np.isfinite(sil_env) else None,
                    "actions": actions_taken,
                    "used_eigen": any(a == Action.EIGEN_EMBED.value for a in actions_taken),
                    "n_clusters": int(
                        np.unique(state.labels[state.labels != -1]).size
                    ),
                }
            )
            rollouts_summary.append(
                {
                    "rollout": r_idx,
                    "n_steps": len(trajectory),
                    "final_silhouette": float(sil_raw) if np.isfinite(sil_raw) else None,
                    "final_silhouette_env": float(sil_env) if np.isfinite(sil_env) else None,
                    "final_n_clusters": int(
                        np.unique(state.labels[state.labels != -1]).size
                    ),
                    "actions": actions_taken,
                }
            )

        # ------------------------------------------------------------------
        # Tiered rollout selection.
        #
        # Goal: pick the rollout whose labels are most plausible, robust to
        # the silhouette pathologies that come up across convex and
        # non-convex data.
        #
        # Rule:
        # 1. Filter to rollouts that produced ``k`` clusters (the requested
        #    number). If none, fall back to all rollouts.
        # 2. If any candidate has ``sil_raw >= 0.45`` (a high raw silhouette
        #    is a strong signal of *true* convex clustering — wrong-split
        #    pathology on circles maxes at ~0.35), pick the highest sil_raw
        #    among those. This keeps mdcgen's clean kmeans-loop rollout
        #    winning even when an EIGEN_EMBED rollout has higher sil_env.
        # 3. Otherwise (no rollout looks "obviously convex-clean"), the
        #    dataset is probably non-convex; pick the rollout with highest
        #    ``sil_env`` from rollouts that *used* EIGEN_EMBED (the spectral
        #    embedding makes its silhouette meaningful), falling back to
        #    overall sil_env, then sil_raw.
        #
        # Constants ``0.45`` is empirically the gap between
        # well-separated-Gaussians silhouette and the worst-case wrong-half
        # silhouette on circles/moons. Treat as a tuning knob, not a
        # universal threshold.
        # ------------------------------------------------------------------
        # Threshold above which a *non-spectral* rollout's raw silhouette is
        # considered evidence of true convex clusters. On well-separated
        # Gaussian blobs (mdcgen) the kpp-loop rollouts typically reach
        # sil_raw ~0.45-0.5. On the wrong-half pathology (circles, moons)
        # they max out below ~0.35. The gap is the basis for tiering.
        CONVEX_SIL_THRESH = 0.40

        # Tier 1: prefer rollouts with the right number of clusters.
        with_correct_k = [r for r in all_rollouts if r["n_clusters"] == int(k)]
        pool = with_correct_k if with_correct_k else all_rollouts

        def _best_by(key: str, candidates):
            valid = [r for r in candidates if r.get(key) is not None]
            if not valid:
                return None
            return max(valid, key=lambda r: r[key])

        non_eigen_pool = [r for r in pool if not r["used_eigen"]]
        eigen_pool = [r for r in pool if r["used_eigen"]]

        chosen = None
        # Tier 2: if a non-spectral rollout produced a partition with
        # sil_raw above the convex threshold, trust it. This stops a
        # spectral rollout with inflated sil_env from stealing the win on
        # easy convex data.
        good_convex = [
            r
            for r in non_eigen_pool
            if (r["sil_raw"] is not None and r["sil_raw"] >= CONVEX_SIL_THRESH)
        ]
        if good_convex:
            chosen = _best_by("sil_raw", good_convex)
        else:
            # Tier 3: non-convex regime — raw silhouette is unreliable.
            # Prefer eigen-rollouts and rank by sil_env (the spectral space
            # silhouette is a real signal after EIGEN_EMBED).
            chosen = _best_by("sil_env", eigen_pool) if eigen_pool else None
            if chosen is None:
                chosen = _best_by("sil_env", pool) or _best_by("sil_raw", pool)

        if chosen is not None:
            best_labels = chosen["labels"]
            best_trajectory = chosen["trajectory"]
            sr = chosen.get("sil_raw")
            se = chosen.get("sil_env")
            best_sil = float(sr if sr is not None else (se if se is not None else -1.0))

        if best_labels is None:
            # All rollouts failed silhouette — fall back to whatever the last
            # rollout produced (which is at least a valid label vector).
            best_labels = state.labels.copy()

        # Final guard: relabel -1 noise as a tail cluster id, because most
        # downstream metric code in clustbench expects non-negative labels.
        # Only do so if there *are* any -1s.
        if (best_labels == -1).any():
            tail = int(best_labels[best_labels != -1].max()) + 1 if (best_labels != -1).any() else 0
            best_labels = best_labels.copy()
            best_labels[best_labels == -1] = tail

        return AlgoResult(
            labels=best_labels.astype(np.int64),
            extra={
                "best_silhouette": float(best_sil) if np.isfinite(best_sil) else None,
                "n_rollouts": int(self.n_rollouts),
                "rollouts": rollouts_summary,
            },
            trajectory=best_trajectory,
        )


def _serialise_params(params: Dict[str, Any]) -> str:
    """Stringify params for the Step.action field — JSON if possible, else repr.

    Step.action is meant to be Parquet-friendly, so we keep it scalar-only.
    """
    try:
        return json.dumps({k: _to_json(v) for k, v in params.items()})
    except Exception:
        return repr(params)


def _to_json(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ---------------------------------------------------------------------------
# Smoke-test runner (matches the spec's three datasets).
# ---------------------------------------------------------------------------


def _smoke() -> Dict[str, Any]:
    """End-to-end smoke: train BC, run Rl_pipeline on three datasets, compare
    against kmeans/spectral baselines. Prints a result summary.

    Invoked when this module is executed as a script (``python -m
    clustbench.algorithms.rl_pipeline``). Also used by the in-task verifier.
    """
    from sklearn.metrics import adjusted_rand_score

    from ..datasets import DataSpec, gen_circles, gen_mdcgen, gen_moons
    from .kmeans import Kmeans
    from .sklearn_extras import Spectral  # registered name lowercases to "spectral"

    out: Dict[str, Any] = {}

    # 1. Train the BC policy. n_epochs=200, patience=30 — the default knobs
    # set on ``train_bc``. Empirically peaks at val_acc ~0.58 around epoch
    # ~80 on the 419-row parquet.
    train_info = train_bc(traces_path=DEFAULT_TRACES)
    out["bc_best_val_acc"] = train_info["best_val_acc"]
    out["bc_n_epochs_run"] = train_info["n_epochs_run"]
    out["bc_train_loss_curve_len"] = len(train_info["train_pol_losses"])
    out["bc_train_pol_losses_last3"] = train_info["train_pol_losses"][-3:]

    # 2. mdcgen / circles / moons.
    datasets = [
        ("mdcgen", gen_mdcgen(DataSpec(n_samples=300, n_features=8, centers=3, compactness=1.0, seed=1)), 3),
        ("circles", gen_circles(DataSpec(n_samples=300, n_features=4, centers=2, compactness=0.5, seed=2)), 2),
        ("moons", gen_moons(DataSpec(n_samples=300, n_features=4, centers=2, compactness=0.5, seed=3)), 2),
    ]

    for name, (X, y), k in datasets:
        # Rl_pipeline. Use exploration + multiple rollouts so the BC policy
        # has a chance to try EIGEN_EMBED-led rollouts on non-convex data.
        # Greedy alone is brittle because the policy's step-0 distribution
        # is dominated by kpp_init at every dataset (the state features at
        # step 0 are nearly identical across algos in the trace data, so
        # the policy converges to the majority class). 16 rollouts at
        # temperature 1.5 reliably samples EIGEN_EMBED at least once.
        rl = Rl_pipeline(max_steps=12, exploration=1.5, n_rollouts=16, random_state=0)
        res = rl.fit_predict(X, k=k)
        ari_rl = float(adjusted_rand_score(y, res.labels))
        used_eigen = any(
            s.action.get("type") == Action.EIGEN_EMBED.value for s in (res.trajectory or [])
        )
        # Kmeans baseline.
        km = Kmeans(record_trajectory=False)
        km_labels = km.fit_predict(X, k=k).labels
        ari_km = float(adjusted_rand_score(y, km_labels))
        # Spectral baseline.
        try:
            sp = Spectral()
            sp_labels = sp.fit_predict(X, k=k).labels
            ari_sp = float(adjusted_rand_score(y, sp_labels))
        except Exception as e:
            ari_sp = float("nan")
            warnings.warn(f"spectral failed on {name}: {e}")

        out[f"{name}_rl_ari"] = ari_rl
        out[f"{name}_kmeans_ari"] = ari_km
        out[f"{name}_spectral_ari"] = ari_sp
        out[f"{name}_used_eigen_embed"] = used_eigen
        out[f"{name}_actions"] = [s.action.get("type") for s in (res.trajectory or [])]

    return out


if __name__ == "__main__":  # pragma: no cover
    res = _smoke()
    for k, v in res.items():
        print(f"{k}: {v}")
