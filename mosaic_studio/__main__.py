"""Command line:

    python -m mosaic_studio app [picture] [--projects DIR] [--port 7860]
        opens the web app in your browser

    python -m mosaic_studio run picture.jpg [--name NAME] [--board-width 30]
                                            [--tile 0.4] [--grout 0.04] [--model vit_l]
        does everything automatically; results in mosaic_projects/NAME/
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser(prog="mosaic_studio")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("app", help="open the web app")
    a.add_argument("picture", nargs="?")
    a.add_argument("--projects", default="mosaic_projects")
    a.add_argument("--port", type=int, default=7860)
    r = sub.add_parser("run", help="make a mosaic automatically")
    r.add_argument("picture")
    r.add_argument("--name")
    r.add_argument("--projects", default="mosaic_projects")
    r.add_argument("--board-width", type=float, default=30.0)
    r.add_argument("--tile", type=float, default=0.4)
    r.add_argument("--grout", type=float, default=0.04)
    r.add_argument("--model", default="vit_l", choices=["vit_l", "vit_b"])
    args = ap.parse_args()
    if args.cmd == "app":
        from .app import main as app_main
        app_main(args.picture, args.projects, args.port)
    else:
        from .pipeline import run_all
        from .project import Project
        name = args.name or os.path.splitext(os.path.basename(args.picture))[0]
        p = Project(os.path.join(args.projects, name))
        p.cfg.update(board_width_cm=args.board_width, tile_cm=args.tile, grout_cm=args.grout,
                     sam_model=args.model)
        run_all(p, args.picture)
        print("Results in", os.path.abspath(p.folder))


if __name__ == "__main__":
    main()
