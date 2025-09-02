from pathlib import Path
import argparse
import nemo_run as run
from nemo.collections import llm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="results/qwen_sft/checkpoints/")
    parser.add_argument("--out", type=str, default="model/")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_path)
    sft_ckpt = next((d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.endswith("-last")), None)
    assert sft_ckpt is not None, f"No '*-last' directory found in {ckpt_dir}"

    export_ckpt = run.Partial(
        llm.export_ckpt,
        path=str(sft_ckpt),
        target="hf",
        output_path=args.out,
    )

    with run.Experiment("export-ckpt", log_level="INFO") as exp:
        exp.add(export_ckpt, executor=run.LocalExecutor(), name="export")
        exp.run(direct=True)  # <-- critical: bypasses fdl_runner
