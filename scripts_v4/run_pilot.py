from _paths import CANON as _CANON  # noqa: F401,E402
from _paths import ROOT as _ROOT  # noqa: F401,E402
import flycircuit

assert "flycircuit_debug_copy" in flycircuit.__file__
from flycircuit.eval import learning_v4 as P  # noqa: E402

assert P.project_root().resolve() == P.ORIG.resolve(), (
    "run through the guarded launcher with FLYCIRCUIT_HOME=$FLYCIRCUIT_HOME"
)
P.prepare_manifest()
raise SystemExit(P.main(["all"]))
