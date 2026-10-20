import flycircuit

assert "flycircuit_debug_copy" in flycircuit.__file__
from flycircuit.eval import learning_v4 as P  # noqa: E402

assert P.project_root().resolve() == P.ORIG.resolve(), (
    "run through the guarded launcher with FLYCIRCUIT_HOME=<CANONICAL_ROOT>"
)
P.prepare_manifest()
raise SystemExit(P.main(["all"]))
