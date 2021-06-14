# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import os
import pickle
import gc
import json
import torch
from torch.utils.data import Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_langs_%s" % args.langs+"_blocksize_%d" % block_size)
        if os.path.exists(cached_file) and not args.overwrite_cache:
            if file_type == 'train':
                logger.info("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)
        else:
            self.inputs = []
            if args.langs == 'all':
                langs = os.listdir(args.data_dir)
            else:
                langs = [args.langs]

            data=[]
            for lang in langs:
                datafile = os.path.join(args.data_dir, lang, file_type+'.pkl')
                if file_type == 'train':
                    logger.warning("Creating features from dataset file at %s", datafile)
                dataset = pickle.load(open(datafile, 'rb'))
                data.extend(['<s> '+' '.join(x['function'].split())+' </s>' for idx,x in enumerate(dataset) if idx%world_size==local_rank])

            # random.shuffle(data)
            data = data
            length = len(data)
            logger.info("Data size: %d" % length)

            input_ids = []
            for idx, x in enumerate(data):
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass

                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.info("Load %d" % percent)
            del data
            gc.collect()

            length = len(input_ids)
            for i in range(0, length-block_size, block_size):
                self.inputs.append(input_ids[i : i + block_size])

            del input_ids
            gc.collect()

            if file_type == 'train':
                logger.info("Training %d token, %d samples" % (length, len(self.inputs)))
                logger.info("Saving features into cached file %s", cached_file)

            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])


# 继承自Dataset基类， 与Dataloader是一对， 用来将数据包装成Dataset类， 模型输入必须是Dataset对象
# 只需要给出初始化__init__, 长度函数__len__, 取元素函数__getitem__即可。
# 返回一个样本的所有特征的字典，这一个个样本实在Dataloader里拼接后再输入到模型中的。
class FinetuneDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d" % block_size)
        if os.path.exists(cached_file) and not args.overwrite_cache:
            # 如果cached_file存在，且不能被覆盖，就直接从已经存在的cached_filed中加载数据
            if file_type == 'train':
                logger.info("Loading features from cached file %s", cached_file)
            # 使用pickle模块从文件中重构python对象
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)
        else:
            # 从文件（如： train.txt）读取数据并封装成可输入模型的训练数据
            self.inputs = []

            datafile = os.path.join(args.data_dir, f"{file_type}.txt")
            if file_type == 'train':
                logger.warning("Creating features from dataset file at %s", datafile)

            with open(datafile) as f:
                data = f.readlines()

            # 数据集长度
            length = len(data)
            logger.info("Data size: %d" % length)

            input_ids = []
            for idx, x in enumerate(data):
                # 保证每一行数据以<s>开头， 以</s>结尾
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    # 数据编码，文本转词向量
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    # 对于编码失败的数据， 可以丢弃
                    pass

                # 每10%打印一次数据加载进度（0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%）
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.info("Load %d" % percent)

            # 删除临时存储的数据，回收垃圾
            del data
            gc.collect()

            # 组织数据
            length = len(input_ids)
            logger.info(f"tokens: {length}")

            # 将数据切分成多个samples, 每个samples大小是block_size
            # 按bloxk_size(如bloxk_size=1024)截断数据段形成样本数据，每个样本大小是block_size
            for i in range(0, length-block_size, block_size):
                self.inputs.append(input_ids[i: i + block_size])

            # 删除临时数据，回收垃圾
            del input_ids
            gc.collect()

            if file_type == 'train':
                logger.info("Training %d token, %d samples" % (length, len(self.inputs)))
                logger.info("Saving features into cached file %s", cached_file)

            # 生成一次样本数据时间比较长，所以使用pickle模块保存到文件，下次直接读取就可以了
            # 使用pickle模块保存python对象文件
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 提供数据集的大小
    def __len__(self):
        return len(self.inputs)

    # 提供整数索引
    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])


class EvalDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d" % block_size)
        if os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)
        else:
            self.inputs = []

            datafile = os.path.join(args.data_dir, f"{file_type}.txt")
            with open(datafile) as f:
                data = f.readlines()

            length = len(data)  # 数据文件行数
            logger.info("Data size: %d" % length)
            input_ids = []
            for idx, x in enumerate(data):
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass

                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.info("load %d" % percent)

            del data
            gc.collect()

            logger.info(f"tokens: {len(input_ids)}")
            self.split(input_ids, tokenizer, logger, block_size=block_size)

            del input_ids
            gc.collect()

            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def split(self, input_ids, tokenizer, logger, block_size=1024):
        sample = []
        i = 0
        while i < len(input_ids):
            sample = input_ids[i: i+block_size]
            if len(sample) == block_size:
                for j in range(block_size):
                    if tokenizer.convert_ids_to_tokens(sample[block_size-1-j])[0] == '\u0120':
                        break
                    if sample[block_size-1-j] in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id]:
                        if sample[block_size-1-j] != tokenizer.bos_token_id:
                            j -= 1
                        break
                if j == block_size-1:
                    print(tokenizer.decode(sample))
                    exit()
                sample = sample[: block_size-1-j]
            # print(len(sample))
            i += len(sample)
            pad_len = block_size-len(sample)
            sample += [tokenizer.pad_token_id]*pad_len
            self.inputs.append(sample)

            if len(self.inputs) % 10000 == 0:
                logger.info(f"{len(self.inputs)} samples")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])
        

class LineDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='test', block_size=924):
        datafile = os.path.join(args.data_dir, f"{file_type}.json")
        with open(datafile) as f:
            datas = f.readlines()

        length = len(datas)
        logger.info("Data size: %d" % length)
        self.inputs = []
        self.gts = []
        for data in datas:
            data = json.loads(data.strip())
            self.inputs.append(tokenizer.encode(data["input"])[-block_size:])
            self.gts.append(data["gt"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), self.gts[item]
