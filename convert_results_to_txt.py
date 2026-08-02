"""
结果文件转换器：把 results 目录下所有 JSON 转成一个 TXT 汇总。
将军只需要跑一次，然后把 output.txt 发给 Mihiro 就行。

使用方法（在 CMD 里运行）：
  cd D:\HierFed-Matter\github_project
  python convert_results_to_txt.py

输出文件：D:\HierFed-Matter\github_project\results_summary.txt
"""

import os
import json
from pathlib import Path

def convert_results_to_txt():
    results_dir = Path(__file__).parent / 'results'
    output_file = Path(__file__).parent / 'results_summary.txt'

    if not results_dir.exists():
        print("❌ results 目录不存在，请先跑实验！")
        return

    lines = []
    lines.append("=" * 70)
    lines.append("HierFed-Matter-NSAC-DPBA 实验结果汇总")
    lines.append("=" * 70)

    # 找所有 JSON 文件
    json_files = sorted(results_dir.rglob('*.json'))

    if not json_files:
        print("❌ results 目录下没有 JSON 文件！")
        return

    # 先写聚合结果
    lines.append("\n" + "=" * 70)
    lines.append("【聚合结果】")
    lines.append("=" * 70)
    for jf in json_files:
        if 'aggregate' in jf.name:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            lines.append(f"\n文件: {jf.name}")
            lines.append(json.dumps(data, indent=2, ensure_ascii=False))

    # 再写每个 seed 的结果（精简版，只写关键指标）
    lines.append("\n" + "=" * 70)
    lines.append("【每个 seed 的精简结果】")
    lines.append("=" * 70)
    for jf in json_files:
        if 'aggregate' not in jf.name:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                lines.append(f"\n⚠ 读取失败: {jf.name} — {e}")
                continue

            lines.append(f"\n--- {jf.parent.name} ---")

            # Final metrics
            if 'final_metrics' in data:
                fm = data['final_metrics']
                lines.append(f"  Final: F1={fm.get('f1','?'):.4f}, "
                             f"macro-F1={fm.get('f1_macro','?'):.4f}, "
                             f"Acc={fm.get('accuracy','?'):.4f}, "
                             f"AUC={fm.get('auc_roc','?'):.4f}, "
                             f"thresh={fm.get('optimal_threshold','?'):.2f}")
                lines.append(f"  Default(0.5): F1={fm.get('f1_default','?'):.4f}, "
                             f"Acc={fm.get('accuracy_default','?'):.4f}")

            # Communication stats
            if 'communication_stats' in data:
                cs = data['communication_stats']
                lines.append(f"  Comm: speedup={cs.get('speedup','?'):.1f}×, "
                             f"reduction={cs.get('reduction_pct','?'):.1f}%")

            # Round-by-round (精简：只写每10轮)
            if 'round_metrics' in data:
                rm = data['round_metrics']
                lines.append(f"  总轮数: {len(rm)}")
                lines.append(f"  Round 1: F1={rm[0].get('f1','?'):.4f}, "
                             f"AUC={rm[0].get('auc_roc','?'):.4f}, "
                             f"CR={rm[0].get('avg_compression_ratio','?'):.4f}")
                for r in rm:
                    if r.get('round') in [10, 20, 30, 40, 50]:
                        lines.append(f"  Round {r['round']}: F1={r.get('f1','?'):.4f}, "
                                     f"AUC={r.get('auc_roc','?'):.4f}, "
                                     f"CR={r.get('avg_compression_ratio','?'):.4f}")

    # 写入 TXT
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✅ 转换完成！共处理 {len(json_files)} 个 JSON 文件")
    print(f"✅ 输出文件: {output_file}")
    print(f"   请把 results_summary.txt 发给 Mihiro～")


if __name__ == '__main__':
    convert_results_to_txt()
