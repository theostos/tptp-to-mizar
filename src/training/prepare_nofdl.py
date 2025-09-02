import nemo_run as run
from nemo.collections import llm
import yaml

if __name__ == "__main__":
	with open("config/training/nemo.yaml") as f:
	    cfg = yaml.safe_load(f)

	import_ckpt = run.Partial(
	    llm.import_ckpt,
	    model=llm.qwen25_32b.model(),
	    source=f"hf://{cfg['model_name']}",
	    overwrite=True,
	)

	# Option A: call the Partial directly (executes in this process)
	#import_ckpt()

	# Option B: Experiment with direct=True
	with run.Experiment("import-ckpt", log_level="INFO") as exp:
	    exp.add(import_ckpt, executor=run.LocalExecutor(), name="import")
	    exp.run(direct=True)   # executes in-process, bypassing fdl_runner