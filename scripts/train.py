"""Training entry point — run this to train and compare all 3 models."""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.trainer import train_and_evaluate


def main():
    print("=" * 60)
    print("Trading Agent — Model Training & Comparison")
    print("=" * 60)
    print()

    results = train_and_evaluate(period_days=10, profit_threshold=3.0)

    print("\n\n" + "=" * 60)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 60)

    all_results = results["all_results"]

    # Print comparison table
    header = f"{'Model':<16} {'AUC-ROC':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>6} {'P@10':>6} {'P@20':>6} {'Backtest ROI':>13} {'Sharpe':>7} {'Time(s)':>8}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        bt = r["backtest"]
        print(
            f"{r['model_name']:<16} "
            f"{r['auc_roc']:>8.4f} "
            f"{r['accuracy']:>9.4f} "
            f"{r['precision']:>10.4f} "
            f"{r['recall']:>7.4f} "
            f"{r['f1']:>6.4f} "
            f"{r['precision_at_10']:>6.3f} "
            f"{r['precision_at_20']:>6.3f} "
            f"{bt['avg_return_pct']:>12.2f}% "
            f"{bt['sharpe_ratio']:>7.3f} "
            f"{r['train_time_sec']:>8.1f}"
        )

    print()
    best = results["best_metrics"]
    print(f">>> RECOMMENDED MODEL: {best['model_name']} <<<")
    print(f"    AUC-ROC:          {best['auc_roc']:.4f}")
    print(f"    Backtest ROI:     {best['backtest']['avg_return_pct']:.2f}%")
    print(f"    Backtest Sharpe:  {best['backtest']['sharpe_ratio']:.3f}")
    print(f"    Win Rate:         {best['backtest']['positive_rate_pct']:.1f}%")
    print(f"\nModel saved to: models/best_model.pkl")


if __name__ == "__main__":
    main()
