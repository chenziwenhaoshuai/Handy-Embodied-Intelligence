r"""一键复现：训练 → 终验 → 导出固件头文件。

把 README 里的三步串起来，跑完直接得到一个可以烧进板子的 `policy_network.h`。

用法：
    python reproduce.py                     # 完整流程（3M 步，约 20–40 分钟）
    python reproduce.py --steps 30000       # 先冒烟测一下链路是否通
    python reproduce.py --seed 1            # 换随机种子再训一版
    python reproduce.py --skip-train        # 用已有权重，只做验证 + 导出
    python reproduce.py --weight rl/ckpt/my_run_600000_steps.zip --skip-train
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE / "rl"
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(HERE))


def pick_best_weight(tag: str) -> Path | None:
    """按评估历史挑「最早达到最高达标数」的那个 checkpoint。"""
    ckpt_dir = RL / "ckpt"
    history = ckpt_dir / f"{tag}_eval_history.json"
    if history.exists():
        try:
            entries = json.loads(history.read_text(encoding="utf-8"))
            best_ok = max(int(e["ok"]) for e in entries)
            for entry in entries:  # 已按时间顺序
                if int(entry["ok"]) == best_ok:
                    candidate = ckpt_dir / f"{tag}_{entry['steps']}_steps.zip"
                    if candidate.exists():
                        print(f"评估历史里最高 {best_ok}/4，取 {candidate.name}", flush=True)
                        return candidate
        except Exception as exc:
            print(f"读取评估历史失败：{exc}", flush=True)
    for name in (f"{tag}_best.zip", f"{tag}_last.zip"):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="EEI-RL 一键复现")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="eei_pure_rl")
    parser.add_argument("--weight", default=None, help="直接指定权重（配合 --skip-train）")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--verify-offsets", default="-10,-5,0,5,10")
    args = parser.parse_args()

    py = sys.executable

    # ---------- 1. 训练 ----------
    if not args.skip_train:
        code = run([py, "-u", str(RL / "train_pure_rl.py"),
                    "--steps", str(args.steps), "--seed", str(args.seed), "--tag", args.tag])
        if code != 0:
            print(f"训练进程返回 {code}，中止。", flush=True)
            sys.exit(code)

    # ---------- 2. 选权重 ----------
    if args.weight:
        weight = Path(args.weight)
        if not weight.is_absolute():
            weight = HERE / weight
    else:
        weight = pick_best_weight(args.tag) or (RL / "weights" / "eei_pure_rl_v1.zip")
    if not weight.exists():
        print(f"找不到权重：{weight}", flush=True)
        sys.exit(1)
    print(f"\n选用权重：{weight}", flush=True)

    # 存一份到 weights/，方便后续引用
    weights_dir = RL / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    saved = weights_dir / f"{args.tag}_v1.zip"
    if weight.resolve() != saved.resolve():
        shutil.copy2(weight, saved)
        print(f"已复制到 {saved}", flush=True)

    # ---------- 3. 终验 ----------
    if not args.skip_verify:
        run([py, "-u", str(RL / "verify_pure_rl.py"),
             "--weight", str(weight), f"--offsets={args.verify_offsets}",
             "--out", str(HERE / "verify_report.json")])

    # ---------- 4. 导出固件头文件 ----------
    header = HERE / "firmware" / "main" / "policy_network.h"
    if not args.skip_export:
        run([py, "-u", str(RL / "export_policy_header.py"), str(weight), "-o", str(header)])
        print(f"\n固件头文件已更新：{header}", flush=True)
        print("接下来：\n"
              "  copy firmware\\main\\wifi_config.example.h firmware\\main\\wifi_config.h  (填 Wi-Fi)\n"
              "  cd firmware && idf.py build && idf.py -p COM8 flash monitor", flush=True)

    print("\n完成。", flush=True)


if __name__ == "__main__":
    main()
