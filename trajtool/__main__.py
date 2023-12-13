import argparse
import sys


def get_args():
    option = str.lower(sys.argv[1])
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*", required=True)
    ap.add_argument("--gsizes", nargs="*", required=True)
    ap.add_argument("--msizes", nargs="*", required=True)
    if option == "superpose":
        ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args(sys.argv[2:])
    args.option = option
    return args


def mainfunc():
    args = get_args()


if __name__ == "__main__":
    mainfunc()
