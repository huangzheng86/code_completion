# -*- coding: utf-8 -*-

"""
@author: HuangZheng
@file: preprocess
@date: 2021/6/8 22:28
"""
import os
import argparse
import random
import javalang


def list_all_files(args, suffix=".java"):
    if not os.path.exists(args.file_dir):
        os.makedirs(args.file_dir)

    wf = open(os.path.join(args.file_dir, "all.txt"), "w")
    for root, dirs, files in os.walk(args.base_dir):
        for f in files:
            if f.endswith(suffix):
                print(os.path.join(args.base_dir, f))
                wf.write(os.path.join(root.replace(args.base_dir + os.sep, ""), f) + "\n")
    wf.close()


def write_to_file(args, file_type, file_list):
    wf = open(os.path.join(args.file_dir, f"{file_type}.txt"), "w")
    for f in file_list:
        wf.write(f)
    wf.close()


def split_data(args):
    file_list = open(os.path.join(args.file_dir, "all.txt"), "r").readlines()
    random.shuffle(file_list)

    train_list = file_list[:int(len(file_list) * 0.8)]
    test_list = file_list[int(len(file_list) * 0.8):int(len(file_list) * 0.95)]
    dev_list = file_list[int(len(file_list) * 0.95):]

    write_to_file(args, "train", train_list)
    write_to_file(args, "test", test_list)
    write_to_file(args, "dev", dev_list)


def java_tokenize(args, file_name, file_type):
    file_paths = open(os.path.join(args.file_dir, file_name)).readlines()
    # print(file_paths)
    wf = open(os.path.join(args.output_dir, f"{file_type}.txt"), "w", encoding="utf-8")
    for ct, path in enumerate(file_paths):
        print(ct, path)
        try:
            code = open(os.path.join(args.base_dir, path.strip())).read()
            print("code: \n", code)

            token_gen = javalang.tokenizer.tokenize(code)
            out_tokens = []
            for token in token_gen:
                tokval = " ".join(token.value.split())
                print(tokval)
                if len(tokval) > 100:
                    print("len(tokval) > 100: ", tokval)
                    continue
                out_tokens.append(tokval.strip())
        except Exception as e:
            print("Error: ", str(e))
            out_tokens = []

        out_tokens = ["<s>"] + out_tokens + ["</s>"]
        out = " ".join(out_tokens)
        out = out.encode("utf-8", "ignore").decode("utf-8")
        wf.write(out + "\n")

        if ct != 0 and ct % 1000 == 0:
            print(f"{file_type}: {ct} are done")

    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="aospCorpus", type=str, help="The download data path.")
    parser.add_argument("--file_dir", default="code_file", type=str, help="The split file.")
    parser.add_argument("--output_dir", default="token_completion", type=str, help="The output directory.")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # list_all_files(args)

    # split_data(args)

    java_tokenize(args, file_name="train.txt", file_type="train")
    java_tokenize(args, file_name="test.txt", file_type="test")
    java_tokenize(args, file_name="dev.txt", file_type="dev")


if __name__ == '__main__':
    main()
