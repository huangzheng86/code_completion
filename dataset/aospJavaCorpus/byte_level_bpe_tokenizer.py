# -*- coding: utf-8 -*-

"""
@author: HuangZheng
@file: byte_level_bpe_tokenizer
@date: 2021/6/13 8:34
"""
import os
import argparse
from pathlib import Path
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class BPEToken(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([NFKC()])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def bpe_train(self, paths, vocab_size=50000):
        trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True, initial_alphabet=ByteLevel.alphabet(),
                             special_tokens=[
                                 "<s>",
                                 "</s>",
                                 "<pad>",
                                 "<|UNKNOWN|>"
                             ])
        self.tokenizer.train(trainer, paths)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)


def bpe_tokenizer(args):
    paths = [str(x) for x in Path(args.input_dir).glob("**/*.txt")]
    print(paths)

    # Initialize a tokenizer
    tokenizer = BPEToken()

    # Customize training
    tokenizer.bpe_train(paths, vocab_size=args.vocab_size)

    # Save files to disk
    save_path = os.path.join(".", args.output_dir)
    tokenizer.save_tokenizer(save_path)


def main():
    parser = argparse.ArgumentParser()
    # 训练数据所在目录
    parser.add_argument("--input_dir", default="token_completion", type=str, help="The input data path")
    parser.add_argument("--output_dir", default="vocab_and_merges", type=str,
                        help="The vocab.json and merges.txt directory.")
    parser.add_argument("--vocab_size", default=50527, type=int, help="The vocab size.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    bpe_tokenizer(args)


if __name__ == '__main__':
    main()