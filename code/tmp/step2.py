import os
os.environ["EXTRA_CB"] = "1"
src = open("tmp/base_trainer.py").read().replace("P = lambda *a: print(\"[P]\", *a, flush=True)", "P = lambda *a: print(\"[P]\", *a, flush=True)")
exec(compile(src, "tmp/base_trainer.py", "exec"))
