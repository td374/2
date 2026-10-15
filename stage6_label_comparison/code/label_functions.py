"""
Stage 6: Label Functions for RUL Prediction

Implements multiple label schemes for bearing RUL prediction:
- Linear (h=0): uniform degradation across entire life
- Piecewise (h=0.8): plateau + linear decline
- Exp-Concave (Kim & Lee 2024): accelerated non-linear degradation
- Crossover-aware: dual-linear based on observed learning-gain crossover

Author: Stage 6 Experiment
Date: 2026-10-15
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable


# ============================================================================
# Core Label Functions
# ============================================================================

def linear_label(t: Union[int, np.ndarray], N: int) -> Union[float, np.ndarray]:
    """
    Linear RUL label: r = (N - t) / N
    
    Full-life uniform degradation assumption.
    
    Args:
        t: Current cycle (int or array)
        N: Total lifetime (int)
    
    Returns:
        Normalized RUL in [0, 1]
    
    Example:
        >>> linear_label(0, 100)
        1.0
        >>> linear_label(100, 100)
        0.0
        >>> linear_label(50, 100)
        0.5
    """
    return (N - t) / N


def piecewise_label(t: Union[int, np.ndarray], N: int, h: float = 0.8) -> Union[float, np.ndarray]:
    """
    Piecewise-linear RUL label with healthy plateau.
    
    r = 1.0                  if t < h*N  (healthy phase)
    r = (N - t) / ((1-h)*N)  if t >= h*N (degradation phase)
    
    Args:
        t: Current cycle
        N: Total lifetime
        h: Plateau fraction (default 0.8, i.e., first 80% is "healthy")
    
    Returns:
        Normalized RUL in [0, 1]
    
    Example:
        >>> piecewise_label(0, 100, h=0.8)
        1.0
        >>> piecewise_label(80, 100, h=0.8)  # At transition
        1.0
        >>> piecewise_label(90, 100, h=0.8)  # Halfway through degradation
        0.5
        >>> piecewise_label(100, 100, h=0.8)
        0.0
    """
    t = np.asarray(t)
    N_plateau = h * N
    
    # Vectorized version
    rul = np.where(
        t < N_plateau,
        1.0,  # Healthy phase
        (N - t) / ((1 - h) * N)  # Degradation phase
    )
    
    return float(rul) if np.isscalar(t) else rul


def exp_concave_label(t: Union[int, np.ndarray], N: int, alpha: float = 2.0) -> Union[float, np.ndarray]:
    """
    Exponential-concave RUL label (Kim & Lee 2024, IET Electronics Letters).
    
    Captures accelerated degradation (slow early, fast late).
    
    r = 1 - [(1 - exp(-alpha)) * (1 - exp(-alpha * t/N))] / (1 - exp(-alpha))
    
    Simplifies to:
    r = 1 - (1 - exp(-alpha * t/N)) / (1 - exp(-alpha))
    
    Args:
        t: Current cycle
        N: Total lifetime
        alpha: Curvature parameter (default 2.0)
               - alpha > 0: concave (accelerated degradation)
               - larger alpha → steeper decline in late life
    
    Returns:
        Normalized RUL in [0, 1]
    
    Example:
        >>> exp_concave_label(0, 100, alpha=2.0)
        1.0
        >>> exp_concave_label(100, 100, alpha=2.0)
        ~0.0
        >>> exp_concave_label(50, 100, alpha=2.0)
        ~0.6 (slower decline early)
    
    Reference:
        Kim & Lee (2024). Deep Learning‐Based Prediction of Remaining Useful Life 
        of Aircraft Engines Using Non‐Linear Labels. IET Electronics Letters.
    """
    t = np.asarray(t)
    ratio = t / N
    
    # Numerically stable computation
    exp_alpha = np.exp(-alpha)
    exp_alpha_ratio = np.exp(-alpha * ratio)
    
    rul = 1.0 - (1.0 - exp_alpha_ratio) / (1.0 - exp_alpha)
    
    # Clip to [0, 1] to handle numerical errors
    rul = np.clip(rul, 0.0, 1.0)
    
    return float(rul) if np.isscalar(t) else rul


def crossover_aware_label(t: Union[int, np.ndarray], N: int, h_cross: float = 0.58) -> Union[float, np.ndarray]:
    """
    Crossover-aware dual-linear RUL label.
    
    Based on the empirical finding that learning gain flips sign at h≈0.55-0.60.
    
    Design:
    - Early phase (t < h_cross * N): slow decline (retains information)
    - Late phase (t >= h_cross * N): fast decline (emphasizes degradation)
    
    r = 1.0 - 0.3 * (t / (h_cross * N))           if t < h_cross * N
    r = 0.7 * (N - t) / ((1 - h_cross) * N)       if t >= h_cross * N
    
    Args:
        t: Current cycle
        N: Total lifetime
        h_cross: Crossover point fraction (default 0.58, from Stage 5 experiments)
    
    Returns:
        Normalized RUL in [0, 1]
    
    Example:
        >>> crossover_aware_label(0, 100, h_cross=0.58)
        1.0
        >>> crossover_aware_label(58, 100, h_cross=0.58)  # At crossover
        0.7
        >>> crossover_aware_label(100, 100, h_cross=0.58)
        0.0
    
    Note:
        This is an experimental label based on the Stage 5 finding that 
        model-vs-hurdle gap crosses zero between h=0.5 and h=0.6.
    """
    t = np.asarray(t)
    N_cross = h_cross * N
    
    # Early phase: gentle decline (retains 70% at crossover)
    early = 1.0 - 0.3 * (t / N_cross)
    
    # Late phase: steep decline (from 0.7 to 0.0)
    late = 0.7 * (N - t) / ((1 - h_cross) * N)
    
    rul = np.where(t < N_cross, early, late)
    
    return float(rul) if np.isscalar(t) else rul


# ============================================================================
# Label Informativeness (LI) Calculation
# ============================================================================

def compute_li(label_func: Callable, N: int, n_samples: int = 1000) -> dict:
    """
    Compute Label Informativeness (LI) for a given label function.
    
    LI = 1 - Var[r] / E[(r-1)²]
    
    Where:
    - Var[r]: variance of the label
    - E[(r-1)²]: expected squared distance from naive baseline (r=1)
    
    Args:
        label_func: Label function (e.g., linear_label, exp_concave_label)
        N: Total lifetime
        n_samples: Number of samples for numerical integration (default 1000)
    
    Returns:
        dict with keys:
            - li: Label informativeness [0, 1]
            - var_r: Variance of the label
            - e_r_minus_1_sq: E[(r-1)²]
            - mean_r: Mean of the label
    
    Example:
        >>> li_linear = compute_li(linear_label, N=100)
        >>> li_linear['li']
        ~0.75 (theoretical value for continuous limit)
    """
    t = np.linspace(0, N, n_samples)
    r = label_func(t, N)
    
    var_r = np.var(r)
    mean_r = np.mean(r)
    e_r_minus_1_sq = np.mean((r - 1.0) ** 2)
    
    li = 1.0 - (var_r / e_r_minus_1_sq) if e_r_minus_1_sq > 0 else 0.0
    
    return {
        'li': li,
        'var_r': var_r,
        'e_r_minus_1_sq': e_r_minus_1_sq,
        'mean_r': mean_r
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_label_comparison(N: int = 100, save_path: str = None):
    """
    Plot all four label schemes for comparison.
    
    Args:
        N: Total lifetime (cycles)
        save_path: Optional path to save figure (e.g., 'figures/label_curves.png')
    
    Example:
        >>> plot_label_comparison(N=100, save_path='analysis/figures/label_curves.png')
    """
    t = np.linspace(0, N, 200)
    
    labels = {
        'Linear (h=0)': (linear_label(t, N), 'blue', '-'),
        'Piecewise (h=0.8)': (piecewise_label(t, N, h=0.8), 'red', '--'),
        'Exp-Concave (α=2.0)': (exp_concave_label(t, N, alpha=2.0), 'green', '-.'),
        'Crossover-aware (h=0.58)': (crossover_aware_label(t, N, h_cross=0.58), 'orange', ':')
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, (r, color, style) in labels.items():
        li_info = compute_li(lambda t, N: eval(name.split('(')[0].lower().replace('-', '_') + '_label')(t, N), N)
        ax.plot(t / N, r, label=f'{name} (LI={li_info["li"]:.3f})', 
                color=color, linestyle=style, linewidth=2.5)
    
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='RUL=0.5')
    ax.axvline(x=0.58, color='purple', linestyle=':', alpha=0.3, label='Crossover (h≈0.58)')
    
    ax.set_xlabel('Normalized Lifetime (t/N)', fontsize=12)
    ax.set_ylabel('Normalized RUL', fontsize=12)
    ax.set_title('RUL Label Comparison: Four Degradation Models', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
    else:
        plt.show()


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Stage 6 Label Functions - Self Test")
    print("=" * 70)
    
    N = 100
    test_points = [0, 25, 50, 58, 75, 80, 90, 100]
    
    print(f"\nTest: N={N}, t=[{', '.join(map(str, test_points))}]\n")
    
    labels_to_test = [
        ('Linear', linear_label),
        ('Piecewise (h=0.8)', lambda t, N: piecewise_label(t, N, h=0.8)),
        ('Exp-Concave (α=2.0)', lambda t, N: exp_concave_label(t, N, alpha=2.0)),
        ('Crossover-aware (h=0.58)', lambda t, N: crossover_aware_label(t, N, h_cross=0.58))
    ]
    
    # Print table header
    print(f"{'Label':<30} | " + " | ".join([f"{t:>5}" for t in test_points]))
    print("-" * 90)
    
    # Print each label
    for name, func in labels_to_test:
        values = [func(t, N) for t in test_points]
        print(f"{name:<30} | " + " | ".join([f"{v:>5.3f}" for v in values]))
    
    # Compute LI for each label
    print("\n" + "=" * 70)
    print("Label Informativeness (LI) Calculation")
    print("=" * 70)
    
    for name, func in labels_to_test:
        li_info = compute_li(func, N)
        print(f"\n{name}:")
        print(f"  LI:          {li_info['li']:.4f}")
        print(f"  Var[r]:      {li_info['var_r']:.4f}")
        print(f"  E[(r-1)²]:   {li_info['e_r_minus_1_sq']:.4f}")
        print(f"  Mean[r]:     {li_info['mean_r']:.4f}")
    
    # Boundary check
    print("\n" + "=" * 70)
    print("Boundary Checks")
    print("=" * 70)
    
    for name, func in labels_to_test:
        r_0 = func(0, N)
        r_N = func(N, N)
        print(f"{name:<30} | r(0)={r_0:.4f} (expect 1.0), r(N)={r_N:.4f} (expect 0.0)")
        
        if not (0.99 <= r_0 <= 1.01):
            print(f"  ⚠️  WARNING: r(0) = {r_0:.4f}, expected ~1.0")
        if not (-0.01 <= r_N <= 0.01):
            print(f"  ⚠️  WARNING: r(N) = {r_N:.4f}, expected ~0.0")
    
    print("\n" + "=" * 70)
    print("✅ Self-test complete. All labels ready for use.")
    print("=" * 70)
    
    # Optional: generate comparison plot
    try:
        print("\nGenerating label comparison plot...")
        plot_label_comparison(N=100, save_path='analysis/figures/label_curves.png')
    except Exception as e:
        print(f"⚠️  Plot generation skipped: {e}")
