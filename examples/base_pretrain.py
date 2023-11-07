# base_pretrain
"""
预训练
    目前针对80w 的 wiki 预训练;
"""

import os
import sys
sys.setrecursionlimit(10000)

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import distance
from rouge import Rouge
import pickle

import torch
from torch.utils.data import DataLoader

from transformers import (
    HfArgumentParser,
    AutoProcessor
)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import kp_setup
import logging

from libs.datasets.base_pretrain_dataset import BasePretrainDataset
from libs import utils
from libs.datasets.dataset_utils import custom_token_split
from libs.models.MyCustomPix2StructForConditionalGeneration import MyCustomPix2StructForConditionalGeneration

# 任务数据路径
data_dir = '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki_500k_long'
# 来自官方的预训练好的路径, 简单做了扩充
model_path = "/home/ana/data4/models/pix2struct-base-raw-enlarge-vocab-special-mean"

train_file = os.path.join(data_dir, "train", "metadata.jsonl")
validation_file = os.path.join(data_dir, "validation", "metadata.jsonl")
test_file = validation_file

# 输出路径
output_dir = "/home/ana/data4/output_models/MyMLLM/p2s_pretrain/stage0_longwiki_80w"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=model_path,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(default="text-generation",
                                     metadata={"help": "The name of the task (ner, pos...)."})
    language: Optional[str] = field(
        default='zh', metadata={"help": "The dataset in xfund to use"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=train_file,
        metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=validation_file,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=test_file,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=data_dir)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    # 这个实现里应该没有这个
    # second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    # 这部分数值貌似model_dir 有保存的
    # imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})
    decoder_max_length: int = field(default=2048, metadata={"help": "生成文本长度"})
    decoder_max_input_length: int = field(default=1024, metadata={"help": "最大 prompt 长度"})


@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    output_dir: Optional[str] = field(
        default=output_dir,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )


compute_counter = 0
def compute_metrics(pred, num_examples=20):
    global compute_counter
    rouge_obj = Rouge()
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # dump for debug
    try:
        utils.make_sure_dir_there(training_args.output_dir)
        with open(os.path.join(training_args.output_dir, f'eval_token_out.{compute_counter}.txt'), 'wb') as f:
            pickle.dump([labels_ids, pred_ids], f)
    except Exception as e:
        logging.error(e)

    # pred trainer 里面 默认 pad -100,(对pred)
    # 截断再对比
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    # pred_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    for i, pred_id in enumerate(pred_ids):
        i_sep_index = np.where(pred_id == processor.tokenizer.sep_token_id)[0][0]
        pred_ids[i][:i_sep_index + 1] = processor.tokenizer.pad_token_id

    decoded_pred_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    decoded_label_strs = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=False)

    # 去掉特殊的 pad 和 </s>
    cleaned_decoded_label_strs = [
        s.replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.eos_token, "") for s in
        decoded_label_strs]
    cleaned_decoded_pred_strs = [s.replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.eos_token, "")
                                 for s in decoded_pred_strs]
    # 防止空串
    label_strs = ["#" if s.strip() == "" else s for s in cleaned_decoded_label_strs]
    pred_strs = ["#" if s.strip() == "" else s for s in cleaned_decoded_pred_strs]

    # 空格分隔是为了配合后续 rouge 等计算
    sep_label_strs = [" ".join(custom_token_split(x)) for x in label_strs]
    sep_pred_strs = [" ".join(custom_token_split(x)) for x in pred_strs]

    sum_distance_ratio = 0
    for i_pred, i_label in zip(pred_strs, label_strs):
        if i_pred == '':
            logging.error(f"should not be empty i_pred:[f{i_pred}], i_label:[f{i_label}]")
        ned = distance.levenshtein(i_pred, i_label, normalized=True)
        sum_distance_ratio += ned

    # sep_pred_str = [" ".join(x) for x in pred_str]
    # sep_label_str = [" ".join(x) for x in label_str]

    levenshtein_distance_ratio = sum_distance_ratio / len(pred_strs)

    #  rouge一句不知道为啥有个bug， 观察
    try:
        with open(os.path.join(training_args.output_dir, f'eval_out.{compute_counter}.txt'), 'w') as f:
            for i_pred, i_label in zip(pred_strs, label_strs):
                line = "\n".join([i_pred, i_label])
                f.write(line + "\n\n")
    except Exception as e:
        logging.error(e)

    rouge_scores = {}
    try:
        with utils.utils_timer("calculating rouge"):
            rouge_scores = rouge_obj.get_scores(sep_pred_strs, sep_label_strs, avg=True)
    except Exception as e:
        logging.error(e)

    bleu_score_corpus = 0
    try:
        with utils.utils_timer("calculating bleu"):
            bleu_score_corpus = corpus_bleu([[s] for s in sep_label_strs], sep_pred_strs)
    except Exception as e:
        logging.error(e)

    logging.info(f"EVAL examples: {num_examples}")
    for i_pred, i_label in zip(pred_strs, label_strs):
        logging.info(f"pred:{i_pred} | label:{i_label}")
        num_examples -= 1
        if num_examples == 0:
            break

    scores = dict()
    scores['NED'] = round(levenshtein_distance_ratio, 4)
    scores['BLEU'] = round(bleu_score_corpus, 4)

    for r_l, r_scores in rouge_scores.items():
        for k, v in r_scores.items():
            scores[f"{r_l}_{k}"] = round(v, 4)
    compute_counter += 1
    return scores


def collator(batch, max_length=2048):
    """
        train / eval /test
        max_length for input
    """
    # 当前实现就是需要有prompt
    tokenizer = processor.tokenizer
    new_batch = {"flattened_patches": [], "attention_mask": []}
    # prompt_and_labels = [item['prompt'] + processor.tokenizer.sep_token_id + item["labels"] for item in batch]

    # 计算 label ，构建 decoder_input_id
    # prompts = []
    prompt_and_labels = []
    for i, item in enumerate(batch):
        i_prompt_and_labels = item['prompt'] + tokenizer.sep_token + item["labels"]
        # shift right by hand
        prompt_and_labels.append(i_prompt_and_labels)
        # prompts.append(item['prompt'])

    # decoder_input
    text_input_ids = processor(text=prompt_and_labels, padding=True, return_tensors="pt", add_special_tokens=True,
                               max_length=max_length, truncation=True)

    # decoder_input_ids, 应该仅训练的时候用到
    decoder_input_ids = model._shift_right(text_input_ids.input_ids)
    # decoder_attention_mask = decoder_input_ids.ne(tokenizer.pad_token_id).float()
    # 因为 shift_right 了,  前面的pad 要被attention到
    # decoder_attention_mask[:, 0] = 1
    new_batch["decoder_input_ids"] = decoder_input_ids
    # new_batch["decoder_attention_mask"] = decoder_attention_mask

    # label, 前面要 -100 加持
    new_batch["labels"] = text_input_ids.input_ids.clone()
    # 找到sep id, 然后 -100
    # 每行都仅有一个 sep
    for i, label in enumerate(new_batch["labels"]):
        # sep_token_id_indexes = int(torch.where(label == tokenizer.sep_token_id)[0][0])
        sep_token_indexes = torch.where(label == tokenizer.sep_token_id)[0].data.numpy()
        i_sep_token_index = 0
        if len(sep_token_indexes) == 0:
            # 有时候会超长.... 这真太难了...
            logging.error(f"不应该找不到 sep_token_id 的位置 !! 请检查预训练任务的长度. {label}, {prompt_and_labels[i]}")
            raise ValueError()
        else:
            i_sep_token_index = int(sep_token_indexes[0])
        label[:i_sep_token_index + 1] = -100

        new_batch["labels"][i] = label

    # for generate utils
    if batch[0]['prompt'] != '':
        prompts = [item['prompt'] + tokenizer.sep_token for item in batch]
        prompts_inputs = processor(text=prompts, padding=True, return_tensors="pt", add_special_tokens=False,
                                   max_length=max_length, truncation=True)
        new_batch['input_ids'] = prompts_inputs['input_ids']

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch


def test_collator(batch, max_length=1024):
    """ 适 用于 train和eval
        todo: 要不要处理好label， 使得prompt部分loss 是-100 ?
    """
    # 当前实现就是需要有prompt
    tokenizer = processor.tokenizer
    new_batch = {"flattened_patches": [], "attention_mask": []}
    # prompt_and_labels = [item['prompt'] + processor.tokenizer.sep_token_id + item["labels"] for item in batch]

    # 计算 label ，构建 decoder_input_id
    #prompts = []
    prompt_and_labels = []
    oris = []
    for i, item in enumerate(batch):
        i_prompt_and_labels = item['prompt'] + tokenizer.sep_token + item["labels"]
        # shift right by hand
        prompt_and_labels.append(i_prompt_and_labels)
        # prompts.append(item['prompt'])
        oris.append(item['ori'])

    # decoder_input
    text_input_ids = processor(text=prompt_and_labels, padding=True, return_tensors="pt", add_special_tokens=True,
                               max_length=max_length, truncation=True)

    # for generate utils
    if batch[0]['prompt'] != '':
        prompts = [item['prompt'] + tokenizer.sep_token for item in batch]
        prompts_inputs = processor(text=prompts, padding=True, return_tensors="pt", add_special_tokens=False,
                                   max_length=max_length, truncation=True)
        new_batch['input_ids'] = prompts_inputs['input_ids']

    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
    new_batch["ori"] = oris

    return new_batch


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    logging.info(f'load processor from {model_args.model_name_or_path}')

    train_dataset, eval_dataset, test_dataset = None, None, None
    train_dataset = BasePretrainDataset(data_args, data_args.train_file, processor, 'train', max_records=None,
                                          image_dir='images', load_cache=True)
    logging.info(f"train_dataset size: {len(train_dataset)}")
    eval_dataset = BasePretrainDataset(data_args, data_args.validation_file, processor, 'eval', max_records=200,
                                        image_dir='images', load_cache=False)
    logging.info(f"eval_dataset size: {len(eval_dataset)}")
    test_dataset = eval_dataset

    # eval_dataset = BaseInsructionDataset(data_args, data_args.validation_file, processor, 'test', max_records=50,
    #                                      prompt='盖章单位是:')
    # logging.info(f"test_dataset size: {len(test_dataset)}")
    # # padding = "max_length" if data_args.pad_to_max_length else False

    # if training_args.do_train:
    if True:
        # # 构造model
        # 输入上针对LayoutLM
        model = MyCustomPix2StructForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        logging.info(f"model loaded from {model_args.model_name_or_path}")

        # training_args.max_steps = 4800
        # training_args.gradient_accumulation_steps = 16
        # training_args.predict_with_generate = True
        # training_args.evaluation_strategy = "steps"
        # training_args.per_device_train_batch_size = 1
        # training_args.per_device_eval_batch_size = 1
        # training_args.logging_steps = 10
        # # training_args.save_steps = 10
        # # training_args.eval_steps = 10
        # # training_args.eval_steps = 1
        # # training_args.fp16 = True
        # training_args.overwrite_output_dir = True
        # training_args.generation_max_length = 2048
        # training_args.generation_num_beams = 1

        # 回头
        model.config.max_length = 2048
        model.config.num_beams = 1

        print(training_args)

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=processor.tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )
        # trainer.evaluate()
        # trainer.train(resume_from_checkpoint=True)
        trainer.train()

    # test
    # if training_args.do_predict:
    if False:
        model = MyCustomPix2StructForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        model.to('cuda')
        logging.info("model loaded")
        model.config.max_length = 2048
        model.config.num_beams = 1
        max_new_tokens = 1024
        num_beams = 1

        all_output = []

        test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=test_collator)
        # 这部分要研究下了... 要支持prompt
        for batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader)):
            logging.info("batch")
            # ori_batch = batch
            labels = batch.pop('labels')
            # batch.pop('decoder_input_ids')
            # prompts = batch.pop('prompt')
            ori_records = [json.loads(x) for x in batch.pop('ori')]
            for k in batch:
                batch[k] = batch[k].to(model.device)
            # print(batch)
            generated_ids = model.generate(**batch, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)
            output = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for pred, i_ori_rec in zip(output, range(len(ori_records))):
                ori_rec = ori_records[i_ori_rec]
                ori_rec['pred'] = pred
            all_output += ori_records
            logging.info("batch Done")

        print(len(output), output)
        print(all_output)
        pred_output_file = os.path.join(training_args.output_dir, 'pred_out.txt')
        try:
            with open(pred_output_file, 'w') as f:
                json.dump(all_output, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logging.error(e)
