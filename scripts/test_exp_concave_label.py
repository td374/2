#!/usr/bin/env python3
"""Test Exp-Concave label implementation and visualize comparison."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add rulbench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rulbench.data.labels import linear_rul, piecewise_linear_rul, exp_concave_rul


def compute_label_informativeness(rul_ratio):
    """Compute Label Informativeness (LI) metric.
    
    LI = 1 - Var[r] / E[(r-1)²]
    """
    var_r = np.var(rul_ratio)
    mean_sq_error = np.mean((rul_ratio - 1.0) ** 2)
    if mean_sq_error < 1e-9:
        return 0.0
    return 1.0 - (var_r / mean_sq_error)


def test_exp_concave_label():
    """Test Exp-Concave label implementation."""
    print("=" * 60)
    print("Testing Exp-Concave Label Implementation")
    print("=" * 60)
    
    n_samples = 1000
    
    # Generate three types of labels
    linear = linear_rul(n_samples)
    piecewise = piecewise_linear_rul(n_samples, healthy_ratio=0.8)
    exp_concave = exp_concave_rul(n_samples, alpha=2.0)
    
    # Boundary checks
    print("\n1. Boundary Checks:")
    print(f"   Linear:      r(0) = {linear[0]:.4f}, r(N) = {linear[-1]:.4f}")
    print(f"   Piecewise:   r(0) = {piecewise[0]:.4f}, r(N) = {piecewise[-1]:.4f}")
    print(f"   Exp-Concave: r(0) = {exp_concave[0]:.4f}, r(N) = {exp_concave[-1]:.4f}")
    
    assert abs(exp_concave[0] - 1.0) < 1e-3, "Exp-Concave should start at 1.0"
    assert abs(exp_concave[-1] - 0.0) < 1e-3, "Exp-Concave should end at 0.0"
    print("   ✓ All boundaries correct")
    
    # Monotonicity check
    print("\n2. Monotonicity Check:")
    diffs = np.diff(exp_concave)
    assert np.all(diffs <= 0), "Exp-Concave should be monotonically decreasing"
    print("   ✓ Exp-Concave is monotonically decreasing")
    
    # Compute LI
    print("\n3. Label Informativeness (LI):")
    li_linear = compute_label_informativeness(linear)
    li_piecewise = compute_label_informativeness(piecewise)
    li_exp_concave = compute_label_informativeness(exp_concave)
    
    print(f"   Linear (h=0):     LI = {li_linear:.3f}")
    print(f"   Piecewise (h=0.8): LI = {li_piecewise:.3f}")
    print(f"   Exp-Concave (α=2): LI = {li_exp_concave:.3f}")
    
    # Theoretical LI for piecewise
    h = 0.8
    li_theory = 3 * (1 - h) / 4
    print(f"   Piecewise theory:  LI = {li_theory:.3f} (expected)")
    
    # Visualization
    print("\n4. Generating Visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: RUL curves
    t_norm = np.linspace(0, 1, n_samples)
    ax1.plot(t_norm, linear, label='Linear (h=0)', linewidth=2, color='blue')
    ax1.plot(t_norm, piecewise, label='Piecewise (h=0.8)', linewidth=2, color='orange')
    ax1.plot(t_norm, exp_concave, label='Exp-Concave (α=2.0)', linewidth=2, color='green', linestyle='--')
    
    ax1.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=0.58, color='red', linestyle=':', alpha=0.3, label='Crossover (h≈0.58)')
    
    ax1.set_xlabel('Normalized Life (t/N)', fontsize=12)
    ax1.set_ylabel('Normalized RUL', fontsize=12)
    ax1.set_title('Three Label Designs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Degradation rate (derivative)
    linear_rate = -np.gradient(linear, t_norm)
    piecewise_rate = -np.gradient(piecewise, t_norm)
    exp_concave_rate = -np.gradient(exp_concave, t_norm)
    
    ax2.plot(t_norm, linear_rate, label='Linear', linewidth=2, color='blue')
    ax2.plot(t_norm, piecewise_rate, label='Piecewise', linewidth=2, color='orange')
    ax2.plot(t_norm, exp_concave_rate, label='Exp-Concave', linewidth=2, color='green', linestyle='--')
    
    ax2.set_xlabel('Normalized Life (t/N)', fontsize=12)
    ax2.set_ylabel('Degradation Rate (-dRUL/dt)', fontsize=12)
    ax2.set_title('Degradation Rate Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / 'label_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to: {output_path}")
    
    # Alpha sensitivity
    print("\n5. Alpha Parameter Sensitivity:")
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for alpha in alphas:
        rul = exp_concave_rul(n_samples, alpha=alpha)
        li = compute_label_informativeness(rul)
        ax.plot(t_norm, rul, label=f'α={alpha:.1f} (LI={li:.3f})', linewidth=2)
        print(f"   α = {alpha:.1f}: LI = {li:.3f}")
    
    ax.plot(t_norm, linear, label=f'Linear (LI={li_linear:.3f})', 
            linewidth=2, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Normalized Life (t/N)', fontsize=12)
    ax.set_ylabel('Normalized RUL', fontsize=12)
    ax.set_title('Exp-Concave: Alpha Parameter Sensitivity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path2 = Path(__file__).parent / 'alpha_sensitivity.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to: {output_path2}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Review generated plots: label_comparison.png, alpha_sensitivity.png")
    print("2. Adjust alpha if needed (current: 2.0)")
    print("3. Run data preparation:")
    print("   python -m rulbench.cli prepare --config configs/exp_concave.yaml")
    print("=" * 60)


if __name__ == "__main__":
    test_exp_concave_label()
