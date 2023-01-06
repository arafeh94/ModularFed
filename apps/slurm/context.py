import argparse
import hashlib


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, help='epochs count')
    parser.add_argument('-b', '--batch', type=int, help='batch count')
    parser.add_argument('-r', '--round', type=int, help='number of rounds')
    parser.add_argument('-cr', '--clients_ratio', type=int, help='selected client percentage for fl')
    parser.add_argument('-lr', '--learn_rate', type=float, help='learn rate')
    parser.add_argument('-t', '--tag', type=str, help='tag to save the results')
    return parser.parse_args()


def hashed(parsed_args):
    return hashlib.md5(str(vars(parsed_args)).encode()).hexdigest()
