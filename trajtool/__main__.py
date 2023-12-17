import argparse
import sys

from .superpose import superpose
from .tfile import TFile


def get_args():
    option = str.lower(sys.argv[1])
    if option == "superpose":
        ap = argparse.ArgumentParser(prog="trajtool superpose")
        ap.add_argument("--out", type=str, required=True)
    else:
        assert False
    ap.add_argument("--input", nargs="*", required=True)
    ap.add_argument("--gsizes", nargs="*", required=True)
    ap.add_argument("--msizes", nargs="*", required=True)
    args = ap.parse_args(sys.argv[2:])
    return args


def mainfunc():
    args = get_args()
    if args.option == "superpose":
        tf = TFile(args)
        superpose(tf)
        tf.write_traj()


if __name__ == "__main__":
    mainfunc()
