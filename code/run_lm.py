# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from dataset import TextDataset, FinetuneDataset, EvalDataset
# from beam import Beam

logger = logging.getLogger(__name__)


# 设置随机种子，其意义在于能够让我们代码的运行结果复现，每次随机数都一样，一般都是以下四步
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# 数据加载
def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.use_pretrain:
        dataset = FinetuneDataset(tokenizer, args, logger, file_type='dev' if evaluate else 'train', block_size=args.block_size)
    else:
        dataset = TextDataset(tokenizer, args, logger, file_type='dev' if evaluate else 'train', block_size=args.block_size)
    return dataset         


def update_config(args, config):
    # config.n_positions = config.n_ctx = args.block_size
    config.vocab_size = args.vocab_size


# 训练过程
def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # tensorboard日志信息收集相关，模型可视化
    tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tb_writer = SummaryWriter(tensorboard_dir)
    
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # 训练数据采样器， RandomSampler无放回地随机采样样本元素
    train_sampler = RandomSampler(train_dataset)
    # 训练数据加载器，组合一个数据集和一个采样器
    # 使用DataLoader这个类来更加快捷的对数据进行操作
    # DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小)， shuffle(是否进行shuffle操作)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)
    total_examples = len(train_dataset)
    batch_size = args.batch_size * args.gradient_accumulation_steps

    t_total = 1
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1

    # num_train_epochs训练轮数
    if args.num_train_epochs > 0:
        t_total = total_examples // batch_size * args.num_train_epochs
        # max_steps 训练总步数（训练轮数 * 每一轮训练的步数）
        args.max_steps = t_total

    # 将模型加载搭到CPU或GPU
    model.to(args.device)

    # 定义优化器，给各个参数设置权重衰减weight_decay
    # 'bias', 'LayerNorm.weight'都不设置权重衰减：'weight_decay': 0.0
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # model.named_parameters()返回一个迭代器，包含参数名称和参数值，所以下面n是参数名称，p是参数值
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # AdamW优化算法中的参数
    # optimizer_grouped_parameters所有可学习的参数，包括不需要权重衰减的'bias', 'LayerNorm.weight'
    # learning_rate学习率
    # adam_epsilon一个非常小的数，防止优化过程中除以0， 通常取1e-8
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 学习率预热
    # 由于刚开始训练时，模型的权重weights是随机初始化的，此时若选择一个较大的学习率，可能带来模型的不稳定（振荡）
    # 选择warmup预热学习率的方式，可以使得开始训练的几个epoches或者一些steps内学习率较小，在预热的小学习率下，模型可以慢慢趋于稳定
    # 等模型相对稳定后再选择先设置的的学习率进行训练，使得模型收敛速度变得更快，模型效果更佳
    # 在预热期间，学习率从0线性增加到优化器中的初始lr
    # 在预热阶段之后创建一个schedule, 使其学习率从优化器中的初始lr线性降低到0
    # warmup_steps：warmup步长阈值，即train_steps < warmup_steps, 使用预热学习率，否则使用预设置值学习率
    # train_steps：训练了的步长数
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    # if os.path.exists(scheduler_last):
    #     scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        logger.info(f"Loading optimizer from {optimizer_last}")
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))

    # CPU训练，默认为fp32, 设置成fp16可以加快训练速度（如果硬件支持的话）
    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # 多GPU和分布式训练，只要硬件支持即可
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    # 训练过程
    # 1. 日志打印，清楚模型在做什么
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_examples)
    logger.info("  Num epoch = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    # 设置训练参数（超参数）
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb = 0.0, 0.0, 0.0, global_step
    # model.resize_token_embeddings(len(tokenizer))
    # 梯度清零
    # 由于pytorch是动态计算图，当我们使用loss.backward()和optimizer.step()进行梯度下降更新参数的时候，梯度并不会自动清零，所以要手动清零
    # 说明pytorch的一个特点是每一步都是独立功能的操作，因此也就有需要梯度清零的说法
    # 如若不显示的进行model.zero_grad()或者optimizer.zero_grad()（效果一样），loss.backward()的时候就会累加梯度
    model.zero_grad()

    # 设置随机种子
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    # 迭代循环，训练的主要部分
    # 每个epoch和每个step形成双重循环
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        for step, batch in enumerate(train_dataloader):
            # 将数据从train_dataloader中读出，每次读取的样本数是batch_size个
            inputs, labels = (batch, batch)

            # 将数据加载到CPU或GPU
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # 启用batch normalization和drop out
            model.train()
            outputs = model(inputs, labels=labels)

            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 反向传播，计算梯度
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # optimizer.step()是执行一次优化步骤，通过梯度下降方法来更新参数的值
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if global_step % args.logging_steps == 0:
                    logger.info("  steps: %s  ppl: %s", global_step, round(avg_loss, 5))

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # tensorboard可视化相关
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)

                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            logger.info("  %s = %s", key, round(value,4))                    
                        output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step, round(results['perplexity'],4)))
                    else:
                        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    # _rotate_checkpoints(args, checkpoint_prefix)
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    model_to_save.save_pretrained(last_output_dir)
                    tokenizer.save_pretrained(last_output_dir)
                    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                    with open(idx_file, 'w', encoding='utf-8') as idxf:
                        idxf.write(str(0) + '\n')

                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, 'step_file.txt')
                    with open(step_file, 'w', encoding='utf-8') as stepf:
                        stepf.write(str(global_step) + '\n')
                    
            # 如果设置了最大步数，且循环达到了最大步数，则提前停止
            if 0 < args.max_steps < global_step:
                break
        if 0 < args.max_steps < global_step:
            break

    tb_writer.close()

    # 返回全局的步数，以及平均的损失
    return global_step, tr_loss / global_step


# 评估过程
# 评估过程与训练过程类似，也包含了加载数据，评估数据，输出结果
# 嵌入到训练过程中的评估过程
def evaluate(args, model, tokenizer, prefix="", eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
       
    for batch in eval_dataloader:
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": float(perplexity)
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        #logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            #logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


# 纯评估过程
def eval_acc(args, model, tokenizer, file_type='test'):
    """
    Evaluate token level code completion on accuracy.
    """
    eval_dataset = EvalDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.to(args.device)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    model.eval()

    correct = 0.0
    total = 0

    total_pred = []
    total_gt = []

    for step, batch in enumerate(eval_dataloader):
        inputs = batch.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)
            pred_scores = outputs[0]
            pred_ids = pred_scores.argmax(-1)    # 取每一行最大值

        all_pred = []
        all_gt = []
        prev_pred = None
        for pred, gt in zip(pred_ids, inputs):    # gt表示一个batch中的一个输入样本
            pred = pred.cpu().tolist()
            gt = gt.cpu().tolist()

            for i, y in enumerate(gt):
                if i == 0:    # 一个输入样本的首个词
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                        now_gt = [y]
                        now_pred = [0] if prev_pred is None else [prev_pred]
                        all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                        all_gt.append(tokenizer.decode(now_gt).strip())
                        now_gt = []
                        now_pred = []
                    else:
                        now_gt = [y]
                        now_pred = [0] if prev_pred is None else [prev_pred]
                else:
                    if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append("<SPACE>")
                            all_gt.append(tokenizer.decode(now_gt).strip())
                            now_gt = []
                            now_pred = []
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append("<SPACE>")
                            all_gt.append(tokenizer.decode(now_gt).strip())
                        now_gt = [y]
                        now_pred = [pred[i-1]]
                        try:
                            all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                        except IndexError:
                            all_pred.append("<SPACE>")
                        all_gt.append(tokenizer.decode(now_gt).strip())
                        now_gt = []
                        now_pred = []
                        continue
                    now_gt.append(y)
                    now_pred.append(pred[i-1])
        assert len(all_pred) == len(all_gt)

        total_pred.extend(all_pred)
        total_gt.extend(all_gt)

        for x, y in zip(all_pred, all_gt):
            if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                total += 1
                if x == y:
                    correct += 1
        
        if step % args.logging_steps == 0:
            logger.info(f"{step} are done!")
            logger.info(f"{total}, {correct/total}")

    # pickle.dump(total_pred, open(os.path.join(args.output_dir, "preds.pkl"), "wb"))
    # pickle.dump(total_gt, open(os.path.join(args.output_dir, "gts.pkl"), "wb"))

    saved_file = os.path.join(args.output_dir, "predictions.txt")
    total_samples = post_process(args, total_pred, total_gt, open(os.path.join(args.data_dir, f"{file_type}.txt")).readlines(), saved_file)
    logger.info(f"Eval on {total_samples}, saved at {saved_file}")
    
    return total, correct


def post_process(args, preds, gts, true_gts, saved_file):
    wf = open(saved_file, "w")

    cnt = 0
    new_gt = []
    new_pred = []
    for i, (pred,gt) in enumerate(zip(preds,gts)):
        if gt in ["", "<pad>"]:
            continue
        new_gt.append(gt)
        new_pred.append(pred.replace(" ", ""))
        if gt == "</s>":
            gt_str = " ".join(new_gt)
            pred_str = " ".join(new_pred)
            assert gt_str == true_gts[cnt].strip(), f"{cnt} sample gt_str != true_gt"
            wf.write(pred_str+"\n")
            cnt += 1
            new_gt = []
            new_pred = []
    
    return cnt


# 程序入口
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--langs", default=None, type=str, required=True,
                        help="Languages to train, if all, train all languages in data_dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    # 预训练模型所在目录
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_dir", type=str,
                        help="config name. Required when training from scratch")
    parser.add_argument("--tokenizer_dir", type=str,
                        help="Pre-trained tokenizer dir. Required when training from scratch")
    # 使用的GPU编号
    parser.add_argument("--gpu_no", type=int, default=-1, help="the no. of GPU")
    # 随机种子
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    # 是否基于预训练模型的数据开始训练
    parser.add_argument('--use_pretrain', action='store_true', help="use different dataset")
    # 每个sample大小是block_size
    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    # output_dir是否可以覆盖
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    # cached_file是否可以覆盖，用于处理Dataset
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")

    ## 优化器相关参数
    # 每个GPU每个batch的大小
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int, help="Batch size per GPU/CPU for evaluation.")
    # 学习率
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    # 权重衰减
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    # AdamW优化算法中的参数，是一个非常小的数，防止在优化过程中除以0， 通常取1e-8
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # 训练轮数
    parser.add_argument("--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    # 学习率预热，步长阈值
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # 跑多少步输出一次日志
    parser.add_argument('--logging_steps', type=int, default=1000, help="Log every X updates steps.")
    # 跑多少步保存一次模型
    parser.add_argument('--save_steps', type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--load_name", type=str, default="pretrained", help="Load pretrained model name")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    # 训练
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    # 评估
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    # 日志文件
    parser.add_argument('--log_file', type=str, default='')
    args = parser.parse_args()

    # args.output_dir = os.path.join(args.output_dir, args.dataset)
    # do_train训练时， 如果output_dir已经存在且不能覆盖，则抛出异常
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # 使用GPU or CPU训练
    if args.gpu_no == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.gpu_no)
        args.device = torch.device("cuda", args.gpu_no)
        args.n_gpu = 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.info("Process device: %s, n_gpu: %s, gpu: %s", args.device, args.n_gpu, args.gpu_no)

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)

    # Set seed
    set_seed(args)

    args.start_epoch = 0
    args.start_step = 0

    # 设置断开后可继续训练
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if args.do_train and os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.pretrain_dir = os.path.join(checkpoint_last)
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        # epoch
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1
        # step
        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} steps".format(checkpoint_last, args.start_step))

    # Load pre-trained model
    pretrained = args.pretrain_dir
    if pretrained:
        # 加载分词器
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained, do_lower_case=False, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>')
        # 加载模型
        model = GPT2LMHeadModel.from_pretrained(pretrained)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>')
        args.vocab_size = len(tokenizer)
        config = GPT2Config.from_pretrained(args.config_dir)
        model = GPT2LMHeadModel(config)
        model.resize_token_embeddings(len(tokenizer))

    # 模型参数
    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")
    logger.info("Training/evaluation parameters %s", args)

    # 训练
    # Training
    if args.do_train:
        # 预处理训练数据，将数据变为可以输入给模型的存储形式
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Only works on single GPU
    if args.do_eval:
        # dev_total, dev_cr = eval_acc(args, model, tokenizer, 'dev')
        # logger.info(f"Dev total tokens: {dev_total}, accuracy: {dev_cr/dev_total}")
        test_total, test_cr = eval_acc(args, model, tokenizer, 'test')
        logger.info(f"Test total tokens: {test_total}, accuracy: {test_cr/test_total}")


if __name__ == "__main__":
    main()
