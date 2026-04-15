import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


# ── Video recording ───────────────────────────────────────────────────────────

def make_env_with_video(video_dir='videos/', record_every=50):
    """Create LunarLander environment with video recording every `record_every` episodes."""
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    env = RecordVideo(env, video_dir, episode_trigger=lambda x: x % record_every == 0)
    return env


def record_episodes(num_episodes: int, out_dir: str, policy_fn) -> None:
    """
    Run `num_episodes` using `policy_fn` and save each episode as a GIF.

    Parameters
    ----------
    num_episodes : int
        Number of episodes to record.
    out_dir : str
        Directory in which to save the GIF files.
    policy_fn : callable
        A function that accepts a state (np.ndarray) and returns an integer action.
        For the random baseline use: ``policy_fn=lambda s: env.action_space.sample()``

    Hints
    -----
    - Use ``gym.make('LunarLander-v3', render_mode='rgb_array')`` so that
      ``env.render()`` returns an RGB frame (numpy array).
    - Collect frames in a list during the episode loop, then write them with
      ``imageio.mimsave(path, frames, fps=30)``.
    - Name each file ``episode_{ep+1}.gif`` inside *out_dir*.
    """
    import imageio
    os.makedirs(out_dir, exist_ok=True)
    env = gym.make('LunarLander-v3', render_mode='rgb_array')

    for ep in range(num_episodes):
        state, _ = env.reset()
        frames = []
        done = False
        total_reward = 0.0

        while not done:
            frames.append(env.render())
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        gif_path = os.path.join(out_dir, f'episode_{ep + 1}.gif')
        imageio.mimsave(gif_path, frames, fps=30)
        print(f'  Episode {ep + 1:2d}  reward = {total_reward:7.1f}  → {gif_path}')

    env.close()


# Statistics helper functions

def print_stats(stats: dict) -> None:
    """
    Print a formatted summary of episode statistics to the console.

    The *stats* dictionary is expected to contain at least:
        mean_reward, std_reward, min_reward, max_reward,
        mean_length, success_rate

    Expected output format (example)::

        ----------------------------------------
        Random Policy Statistics
        ----------------------------------------
          Mean reward  :   -123.45 ± 80.23
          Min / Max    :   -456.78 / 12.34
          Mean length  :    250.0 steps
          Success rate :      0.0%
        ----------------------------------------

    Hints
    -----
    - Use f-strings with format specs such as ``:.2f`` and ``:.1f``.
    - The success rate in *stats* is stored as a fraction (0–1); multiply by 100
      to display as a percentage.
    """
    sep = '-' * 40
    print(sep)
    print('Episode Statistics')
    print(sep)
    print(f"  Mean reward  : {stats['mean_reward']:>8.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min / Max    : {stats['min_reward']:>8.2f} / {stats['max_reward']:.2f}")
    print(f"  Mean length  : {stats['mean_length']:>8.1f} steps")
    print(f"  Success rate : {stats['success_rate'] * 100:>7.1f}%")
    print(sep)


# Plotting helper functions

def moving_average(data, window: int = 20) -> np.ndarray:
    """
    Compute a simple moving average using a uniform kernel.

    Parameters
    ----------
    data : array-like
        1-D sequence of values (e.g. per-episode rewards or losses).
    window : int
        Number of elements to average over.

    Returns
    -------
    np.ndarray
        Array of length ``len(data) - window + 1`` containing the smoothed
        values.  The first element is the mean of ``data[0:window]``.

    Hints
    -----
    - ``np.convolve(data, kernel, mode='valid')`` with a uniform kernel of
      length *window* is a one-liner solution.
    - Remember to normalise the kernel so its values sum to 1.
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def save_training_metrics_csv(metrics: dict, out_path: str = "outputs/part_b_c/training_metrics.csv", window: int = 20) -> None:
    """
    Save moving averages of training metrics to a CSV file for comparison across experiments.

    Saves moving averages for:
        - Episode rewards
        - Training loss (Huber)
        - Mean max Q-value
        - Epsilon (exploration rate)

    Parameters
    ----------
    metrics : dict
        Dictionary returned by the ``train()`` function; must contain
        ``episode_rewards``, ``avg_losses``, ``epsilons``, ``mean_q_values``.
    out_path : str
        File path (CSV) where the metrics are saved.
    window : int
        Window size for moving average computation.

    Examples
    --------
    >>> save_training_metrics_csv(metrics, out_path="outputs/part_b_c/training_metrics.csv")
    Saved training metrics → outputs/part_b_c/training_metrics.csv
    """
    import csv
    
    rewards  = metrics['episode_rewards']
    losses   = metrics['avg_losses']
    epsilons = metrics['epsilons']
    q_values = metrics['mean_q_values']
    N        = len(rewards)
    
    # Compute moving averages
    ma_rewards = moving_average(rewards, window) if len(rewards) >= window else []
    
    # Filter out NaNs for loss and Q-value before computing MA
    loss_values = [l for l in losses if not np.isnan(l)]
    ma_losses = moving_average(loss_values, window) if len(loss_values) >= window else []
    
    q_values_clean = [q for q in q_values if not np.isnan(q)]
    ma_q_values = moving_average(q_values_clean, window) if len(q_values_clean) >= window else []
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Write to CSV
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Episode', 'Reward_MA', 'Loss_MA', 'Q_Value_MA', 'Epsilon'])
        
        # Write data rows
        for ep in range(1, N + 1):
            # Reward MA: available from episode window onwards
            reward_ma = ma_rewards[ep - window] if (ep >= window and ep - window < len(ma_rewards)) else np.nan
            
            # Loss MA: need to track which episode index corresponds to non-NaN loss values
            loss_ma = np.nan
            if not np.isnan(losses[ep - 1]) and len(ma_losses) > 0:
                # Count how many non-NaN losses we've seen up to this episode
                non_nan_count = sum(1 for i in range(ep) if not np.isnan(losses[i]))
                if non_nan_count >= window:
                    loss_ma = ma_losses[non_nan_count - window]
            
            # Q-value MA: similar to loss
            q_value_ma = np.nan
            if not np.isnan(q_values[ep - 1]) and len(ma_q_values) > 0:
                non_nan_count = sum(1 for i in range(ep) if not np.isnan(q_values[i]))
                if non_nan_count >= window:
                    q_value_ma = ma_q_values[non_nan_count - window]
            
            epsilon = epsilons[ep - 1] if ep <= len(epsilons) else np.nan
            
            writer.writerow([ep, reward_ma, loss_ma, q_value_ma, epsilon])
    
    print(f'Saved training metrics CSV → {out_path}')


def compare_experiments(csv_paths: list, labels: list = None, out_path: str = "outputs/part_b_c/experiments_comparison.png") -> None:
    """
    Load multiple CSV files and create a 1×3 comparison figure of moving averages.

    Creates three subplots side-by-side showing:
        [0,0] Reward moving average comparison
        [0,1] Loss moving average comparison
        [0,2] Q-value moving average comparison

    Each subplot will display all experiments for easy comparison.

    Parameters
    ----------
    csv_paths : list
        List of paths to training_metrics.csv files to compare (up to 3 files).
    labels : list, optional
        Labels for each experiment (will use filenames if not provided).
    out_path : str
        File path (PNG) where the comparison figure is saved.

    Examples
    --------
    >>> csv_files = [
    ...     "outputs/part_b_c_exp1/training_metrics.csv",
    ...     "outputs/part_b_c_exp2/training_metrics.csv",
    ...     "outputs/part_b_c_exp3/training_metrics.csv"
    ... ]
    >>> compare_experiments(csv_files, labels=["Exp1", "Exp2", "Exp3"])
    Saved experiments comparison → outputs/part_b_c/experiments_comparison.png
    """
    import pandas as pd
    
    # Use filenames as labels if not provided
    if labels is None:
        labels = [os.path.basename(os.path.dirname(path)) for path in csv_paths]
    
    # Color palette for different experiments
    colors = ['steelblue', 'darkorange', 'crimson', 'purple', 'green']
    
    # Load CSV files
    data = {}
    for i, csv_path in enumerate(csv_paths):
        try:
            df = pd.read_csv(csv_path)
            data[labels[i]] = df
            print(f'Loaded {labels[i]}: {csv_path}')
        except FileNotFoundError:
            print(f'Warning: File not found: {csv_path}')
            continue
    
    if not data:
        print('Error: No CSV files were successfully loaded.')
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Metrics Comparison Across Experiments', fontsize=14, fontweight='bold')
    
    # [0,0] Reward MA comparison
    ax = axes[0]
    for idx, (label, df) in enumerate(data.items()):
        valid_data = df[df['Reward_MA'].notna()]
        ax.plot(valid_data['Episode'], valid_data['Reward_MA'], 
                marker='o', markersize=3, linewidth=2, label=label, 
                color=colors[idx % len(colors)], alpha=0.8)
    ax.axhline(200.0, color='green', linestyle='--', alpha=0.5, label='Solved (200)')
    ax.set_title('Reward Moving Average Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (MA-20)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # [0,1] Loss MA comparison
    ax = axes[1]
    for idx, (label, df) in enumerate(data.items()):
        valid_data = df[df['Loss_MA'].notna()]
        ax.plot(valid_data['Episode'], valid_data['Loss_MA'], 
                marker='s', markersize=3, linewidth=2, label=label, 
                color=colors[idx % len(colors)], alpha=0.8)
    ax.set_title('Loss Moving Average Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss (MA-20)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # [0,2] Q-value MA comparison
    ax = axes[2]
    for idx, (label, df) in enumerate(data.items()):
        valid_data = df[df['Q_Value_MA'].notna()]
        ax.plot(valid_data['Episode'], valid_data['Q_Value_MA'], 
                marker='^', markersize=3, linewidth=2, label=label, 
                color=colors[idx % len(colors)], alpha=0.8)
    ax.set_title('Q-Value Moving Average Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-Value (MA-20)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved experiments comparison → {out_path}')
    plt.close()


def compare_experiment_pairs(csv_path_pairs: list, labels: list = None, out_path: str = "outputs/part_b_c/experiments_pairs_comparison.png", include_epsilon: bool = False) -> None:
    """
    Load paired CSV files per experiment and create a 1×3 comparison figure.

    Each experiment is represented by two CSV files from different executions of
    the same setup. The plotted curve is the per-episode mean across the two
    runs, and the shaded area is ±1 standard deviation.

    By default, creates three subplots side-by-side showing:
        [0,0] Reward moving average comparison (mean ± std)
        [0,1] Loss moving average comparison   (mean ± std)
        [0,2] Q-value moving average comparison(mean ± std)

    If ``include_epsilon`` is True, the figure switches to a 2×2 grid showing:
        [0,0] Episode Reward
        [0,1] Training Loss
        [1,0] Epsilon Decay
        [1,1] Mean Max Q-Value

    Parameters
    ----------
    csv_path_pairs : list
        List where each element is a pair (length-2 list/tuple) of CSV paths.
        Example: [[run1.csv, run2.csv], [run1.csv, run2.csv], ...].
    labels : list, optional
        Labels for each experiment (will use folder names if not provided).
    out_path : str
        File path (PNG) where the comparison figure is saved.
    include_epsilon : bool, optional
        If True, include the epsilon decay subplot and use a 2×2 layout.
    """
    import pandas as pd

    if labels is None:
        labels = [os.path.basename(os.path.dirname(pair[0])) for pair in csv_path_pairs]

    colors = ['steelblue', 'darkorange', 'crimson', 'purple', 'green']

    # Load each pair as one experiment entry.
    paired_data = {}
    for i, pair in enumerate(csv_path_pairs):
        if len(pair) != 2:
            print(f'Warning: Skipping entry {i} because it does not contain exactly two CSV files.')
            continue

        run_a, run_b = pair
        label = labels[i] if i < len(labels) else f'Experiment {i + 1}'

        try:
            df_a = pd.read_csv(run_a)
            df_b = pd.read_csv(run_b)
            paired_data[label] = (df_a, df_b)
            print(f'Loaded {label}: {run_a} + {run_b}')
        except FileNotFoundError as exc:
            print(f'Warning: File not found for {label}: {exc.filename}')
            continue

    if not paired_data:
        print('Error: No CSV pairs were successfully loaded.')
        return

    def _pair_mean_std(df_a, df_b, metric_col: str):
        merged = pd.merge(
            df_a[['Episode', metric_col]],
            df_b[['Episode', metric_col]],
            on='Episode',
            how='outer',
            suffixes=('_a', '_b')
        ).sort_values('Episode')

        values = merged[[f'{metric_col}_a', f'{metric_col}_b']].to_numpy(dtype=float)
        counts = np.sum(~np.isnan(values), axis=1)

        # Mean/std across available runs at each episode (ignoring NaNs).
        mean_vals = np.nansum(values, axis=1) / np.where(counts == 0, 1, counts)
        std_vals = np.sqrt(np.nansum((values - mean_vals[:, None]) ** 2, axis=1) / np.where(counts == 0, 1, counts))

        mean_vals[counts == 0] = np.nan
        std_vals[counts == 0] = np.nan

        valid = ~np.isnan(mean_vals)
        episodes = merged['Episode'].to_numpy()[valid]
        return episodes, mean_vals[valid], std_vals[valid]

    def _plot_metric(ax, metric_col: str, title: str, ylabel: str, add_solved_line: bool = False):
        for idx, (label, (df_a, df_b)) in enumerate(paired_data.items()):
            ep, mean_vals, std_vals = _pair_mean_std(df_a, df_b, metric_col)
            ax.plot(ep, mean_vals, linewidth=2, label=label,
                    color=colors[idx % len(colors)], alpha=0.9)
            ax.fill_between(ep, mean_vals - std_vals, mean_vals + std_vals,
                            color=colors[idx % len(colors)], alpha=0.15)
        if add_solved_line:
            ax.axhline(200.0, color='green', linestyle='--', alpha=0.5, label='Solved (200)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if include_epsilon:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle('Training Metrics Comparison Across Experiment Pairs', fontsize=14, fontweight='bold')

        _plot_metric(axes[0, 0], 'Reward_MA', 'Episode Reward', 'Reward (MA-20)', add_solved_line=True)
        _plot_metric(axes[0, 1], 'Loss_MA', 'Training Loss', 'Loss (MA-20)')
        _plot_metric(axes[1, 0], 'Epsilon', 'Epsilon Decay', 'Epsilon')
        _plot_metric(axes[1, 1], 'Q_Value_MA', 'Mean Max Q-Value', 'Q-Value (MA-20)')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Training Metrics Comparison Across Experiment Pairs', fontsize=14, fontweight='bold')

        _plot_metric(axes[0], 'Reward_MA', 'Reward Moving Average Comparison', 'Reward (MA-20)', add_solved_line=True)
        _plot_metric(axes[1], 'Loss_MA', 'Loss Moving Average Comparison', 'Loss (MA-20)')
        _plot_metric(axes[2], 'Q_Value_MA', 'Q-Value Moving Average Comparison', 'Q-Value (MA-20)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved paired experiments comparison → {out_path}')
    plt.close()


def plot_baseline(stats: dict, out_path: str = "outputs/part_a/baseline_stats.png") -> None:
    """
    Generate a 2×2 figure summarising the random-policy baseline and save it.

    The four sub-plots should be:
        [0,0] Episode reward over time (line plot + mean horizontal line)
        [0,1] Reward distribution (histogram + mean vertical line)
        [1,0] Episode length over time (line plot + mean horizontal line)
        [1,1] Text summary box (mean, std, min, max, mean length, success rate)

    Parameters
    ----------
    stats : dict
        Dictionary returned by ``run_random_baseline()``; must contain
        ``episode_rewards``, ``episode_lengths``, ``mean_reward``,
        ``std_reward``, ``min_reward``, ``max_reward``, ``mean_length``,
        ``success_rate``.
    out_path : str
        File path (PNG) where the figure is saved.

    Hints
    -----
    - Use ``plt.subplots(2, 2, figsize=(12, 8))``.
    - For the text panel use ``axes[1,1].axis('off')`` then
      ``axes[1,1].text(...)``.  A monospace font and a ``bbox`` with
      ``boxstyle='round'`` look clean.
    - Call ``plt.tight_layout()`` before saving.
    - Use ``plt.close()`` after saving to free memory.
    """
    rewards  = stats['episode_rewards']
    lengths  = stats['episode_lengths']
    episodes = range(1, len(rewards) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Part A – Random Policy Baseline', fontsize=14)

    # [0,0] Reward over time
    axes[0, 0].plot(episodes, rewards, alpha=0.6, color='steelblue')
    axes[0, 0].axhline(stats['mean_reward'], color='red', linestyle='--',
                       label=f"Mean = {stats['mean_reward']:.1f}")
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # [0,1] Reward distribution
    axes[0, 1].hist(rewards, bins=20, color='steelblue', edgecolor='white')
    axes[0, 1].axvline(stats['mean_reward'], color='red', linestyle='--',
                       label=f"Mean = {stats['mean_reward']:.1f}")
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].set_xlabel('Total Reward')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # [1,0] Episode lengths
    axes[1, 0].plot(episodes, lengths, alpha=0.6, color='darkorange')
    axes[1, 0].axhline(stats['mean_length'], color='red', linestyle='--',
                       label=f"Mean = {stats['mean_length']:.1f}")
    axes[1, 0].set_title('Episode Length')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # [1,1] Summary text box
    summary = (
        f"Random Policy Statistics\n"
        f"{'─' * 30}\n"
        f"  Mean reward  : {stats['mean_reward']:>8.2f}\n"
        f"  Std  reward  : {stats['std_reward']:>8.2f}\n"
        f"  Min  reward  : {stats['min_reward']:>8.2f}\n"
        f"  Max  reward  : {stats['max_reward']:>8.2f}\n"
        f"  Mean length  : {stats['mean_length']:>8.1f} steps\n"
        f"  Success rate : {stats['success_rate'] * 100:>7.1f}%"
    )
    axes[1, 1].axis('off')
    axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f'Saved baseline plot → {out_path}')
    plt.close()


def plot_training_curves(metrics: dict, out_dir: str = "outputs/part_b_c", fileName_suffix: str = "") -> None:
    """
    Generate a 2×2 figure of DQN training diagnostics and save it.

    The four sub-plots should be:
        [0,0] Episode reward + moving average + solved threshold line
        [0,1] Training loss (Huber) + moving average  (skip NaN episodes)
        [1,0] Epsilon decay over episodes
        [1,1] Mean max Q-value + moving average       (skip NaN episodes)

    Parameters
    ----------
    metrics : dict
        Dictionary returned by the ``train()`` function; must contain
        ``episode_rewards``, ``avg_losses``, ``epsilons``, ``mean_q_values``,
        and ``solved_at`` (int or None).
    out_dir : str
        Directory in which to save the plot.
    fileName_suffix : str
        Suffix to append to the default file name.

    Hints
    -----
    - Use your ``moving_average()`` helper for smoothing.
    - Loss and Q-value lists may contain ``float('nan')`` for early episodes
      before the buffer is warm; filter these out before plotting.
    - If ``solved_at`` is not None, draw a vertical dashed line on the reward
      sub-plot to mark the episode where the environment was solved.
    - Save to ``os.path.join(out_dir, 'training_curves' + fileName_suffix + '.png')``.
    """
    SOLVED_THRESHOLD = 200.0
    window   = 20
    rewards  = metrics['episode_rewards']
    losses   = metrics['avg_losses']
    epsilons = metrics['epsilons']
    q_values = metrics['mean_q_values']
    solved_at = metrics.get('solved_at')
    N        = len(rewards)
    episodes = np.arange(1, N + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('DQN Training – LunarLander-v3', fontsize=14)

    # [0,0] Reward curve
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color='steelblue', label='per episode')
    if len(rewards) >= window:
        ma = moving_average(rewards, window)
        ax.plot(episodes[window - 1:], ma, color='steelblue', linewidth=2,
                label=f'MA-{window}')
    ax.axhline(SOLVED_THRESHOLD, color='green', linestyle='--', alpha=0.7,
               label=f'Solved ({SOLVED_THRESHOLD})')
    if solved_at:
        ax.axvline(solved_at, color='red', linestyle=':', alpha=0.8,
                   label=f'Solved @ ep {solved_at}')
    ax.set_title('Episode Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1] Loss curve (filter NaNs)
    ax = axes[0, 1]
    valid_loss = [(i + 1, l) for i, l in enumerate(losses) if not np.isnan(l)]
    if valid_loss:
        ep_l, l_vals = zip(*valid_loss)
        ax.plot(ep_l, l_vals, alpha=0.3, color='darkorange', label='per episode')
        if len(l_vals) >= window:
            ma_l = moving_average(l_vals, window)
            ax.plot(list(ep_l)[window - 1:], ma_l, color='darkorange',
                    linewidth=2, label=f'MA-{window}')
    ax.set_title('Training Loss (Huber)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # [1,0] Epsilon decay
    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color='purple', linewidth=2)
    ax.set_title('Epsilon Decay')
    ax.set_xlabel('Episode')
    ax.set_ylabel('ε (exploration prob.)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # [1,1] Mean Q-values (filter NaNs)
    ax = axes[1, 1]
    valid_q = [(i + 1, q) for i, q in enumerate(q_values) if not np.isnan(q)]
    if valid_q:
        ep_q, q_vals = zip(*valid_q)
        ax.plot(ep_q, q_vals, alpha=0.3, color='crimson', label='per episode')
        if len(q_vals) >= window:
            ma_q = moving_average(q_vals, window)
            ax.plot(list(ep_q)[window - 1:], ma_q, color='crimson',
                    linewidth=2, label=f'MA-{window}')
    ax.set_title('Mean Max Q-Value')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q̄')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'training_curves' + fileName_suffix + '.png')
    plt.savefig(path, dpi=150)
    print(f'Saved training curves → {path}')
    plt.close()
    
    # Save moving averages to CSV for comparison across experiments
    csv_path = os.path.join(out_dir, 'training_metrics' + fileName_suffix + '.csv')
    save_training_metrics_csv(metrics, csv_path, window=window)


# Checkpoint helper functions

def save_checkpoint(agent, episode: int, rewards: list, filename: str) -> None:
    """Save agent weights, optimiser state, and reward history to a .pt file."""
    torch.save({
        'episode': episode,
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'rewards_history': rewards,
    }, filename)


def load_checkpoint(agent, filename: str):
    """
    Load a checkpoint saved by ``save_checkpoint`` into *agent*.

    Returns
    -------
    (episode, rewards_history) : (int, list)
    """
    checkpoint = torch.load(filename)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['rewards_history']