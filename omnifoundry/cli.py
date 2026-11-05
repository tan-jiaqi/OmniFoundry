"""omnifoundry CLI.

    python -m omnifoundry run     --config configs/m1.yaml --out runs/m1-001 [--stages caption,generate] [--set k=v ...]
    python -m omnifoundry funnel  runs/m1-001
    python -m omnifoundry compare runs/a runs/b
    python -m omnifoundry review  runs/m1-001          # static HTML page for manual spot checks
    python -m omnifoundry review-serve runs/m1-001    # interactive review UI (blind answer, judgments)
"""
import argparse
from pathlib import Path
import sys

import yaml

from .config import load_config


def _overrides(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = yaml.safe_load(v)
    return out


def cmd_run(args):
    from .context import Context
    from .stages import BY_NAME, ORDER
    from .stages import export

    config = load_config(args.config, _overrides(args.set))
    ctx = Context.build(config, args.out)
    ctx.snapshot(" ".join(sys.argv))
    names = args.stages.split(",") if args.stages else [s.NAME for s in ORDER] + ["export"]
    prev = None
    for stage in ORDER:
        if stage.NAME in names:
            items = stage.inputs(ctx, prev) if stage.NAME != "select" else stage.inputs(ctx)
            from .stages.common import run_stage
            run_stage(ctx, stage, items)
        prev = ctx.run_dir / stage.NAME
    if "export" in names:
        export.run(ctx)
    unknown = set(names) - set(BY_NAME) - {"export"}
    if unknown:
        print(f"warning: unknown stages ignored: {sorted(unknown)}", file=sys.stderr)


def cmd_funnel(args):
    from .analysis.funnel import funnel, render
    print(render(funnel(Path(args.run))))


def cmd_compare(args):
    from .analysis.compare import compare
    print(compare(Path(args.a), Path(args.b)))


def cmd_review(args):
    from .analysis.review_page import build
    print(build(Path(args.run), args.stage, args.n))


def cmd_review_serve(args):
    from .analysis.review_server import serve
    serve(Path(args.run), args.stage, args.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="omnifoundry", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--stages", help="comma list; default all (select,caption,generate,verify,think,export)")
    r.add_argument("--set", action="append", help="config override, dotted.key=yaml_value")
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("funnel")
    f.add_argument("run")
    f.set_defaults(fn=cmd_funnel)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=cmd_compare)
    v = sub.add_parser("review")
    v.add_argument("run")
    v.add_argument("--stage", default="think")
    v.add_argument("--n", type=int, default=20)
    v.set_defaults(fn=cmd_review)
    rs = sub.add_parser("review-serve", help="interactive review UI on 127.0.0.1")
    rs.add_argument("run")
    rs.add_argument("--stage", default="think", help="which stage's out.jsonl to review (think or verify)")
    rs.add_argument("--port", type=int, default=8765)
    rs.set_defaults(fn=cmd_review_serve)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
