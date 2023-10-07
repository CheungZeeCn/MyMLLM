"""
用 pix2struct 做wiki语料预训练 看效果
    完成 cn 版本的基础模型
    # 语言模型预训练, OCR 能力训练
    # 1. 图像输入
    # 2. decoder中放入指令:
        a. 训练的时候:
            label 中 prompt 部分要改为 -100;
            同时 decoder_input_ids 改为:  <pad> 向右偏移的prompt+原来的label （参考模型的_shift_right()）

        b. 预测的时候:
            加入一个 input_ids, transformer 的 model.generate 会自动将其作为前缀.
            todo:
            如果要计算metric, 这里也应该只截取后半部分, 这部分怎么截取 ? 怎么知道哪些是new token?
"""
import kp_setup
import logging
import datasets
import os
import sys
sys.setrecursionlimit(10000)
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import numpy as np

import torch
from transformers import BertTokenizerFast

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoProcessor,
    Pix2StructForConditionalGeneration,
    DataCollator
)

from transformers import Pix2StructProcessor

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from libs.processors.processing_layoutlmv3 import LayoutLMv3Processor
from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# custom models
import libs.models.custom_unilm_layoutlmv3
from libs.models.MyEncoderDecoderModelv4 import MyEncoderDecoderModelv4
from libs.models.MyCustomPix2StructForConditionalGeneration import MyCustomPix2StructForConditionalGeneration


from libs.datasets.wiki_pretrain import wiki_pretrain_dataset
from libs.datasets.dataset_utils import custom_token_split
from libs import utils

import distance
from rouge import Rouge
import pickle


# model_path = "/home/ana/data4/models/pix2struct-base"
# model_path = "/home/ana/data4/models/pix2struct-base-enlarge-vocab-special"
model_path = "/home/ana/data4/models/pix2struct-base-raw-enlarge-vocab-special"

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
        default="/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/train/metadata.jsonl",
        metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default="/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/validation/metadata.jsonl",
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default="/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/test/metadata.jsonl",
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
    data_dir: Optional[str] = field(default="/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/")
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    # 这个实现里应该没有这个
    # second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    # 这部分数值貌似model_dir 有保存的
    # imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})
    decoder_max_length: int = field(default=1024, metadata={"help": "生成文本长度"})
    decoder_max_input_length: int = field(default=1024, metadata={"help": "最大 prompt 长度"})


@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    output_dir: Optional[str] = field(
        default="/home/ana/data4/output_models/MyMLLM/100k_p2s_wiki_pretrain_2tasks",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


##################  metrics ###############
# rouge = datasets.load_metric(os.path.join(kp_setup.lib_dir, 'rouge.py'))
rouge_obj = Rouge()
compute_counter = 0
def compute_metrics(pred):
    global compute_counter
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    #dump for debug
    try:
        utils.make_sure_dir_there(training_args.output_dir)
        with open(os.path.join(training_args.output_dir, f'eval_token_out.{compute_counter}.txt'), 'wb') as f:
            pickle.dump([labels_ids, pred_ids], f)
    except Exception as e:
        logging.error(e)

    # pred trainer 里面 默认 pad -100,(对pred)
    # 截断再对比
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    #pred_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    for i, pred_id in enumerate(pred_ids):
        i_sep_index = np.where(pred_id == processor.tokenizer.sep_token_id)[0][0]
        pred_ids[i][:i_sep_index+1] = processor.tokenizer.pad_token_id

    decoded_pred_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    decoded_label_strs = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=False)

    # 去掉特殊的 pad 和 </s>
    cleaned_decoded_label_strs = [s.replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.eos_token, "") for s in decoded_label_strs]
    cleaned_decoded_pred_strs = [s.replace(processor.tokenizer.pad_token, "").replace(processor.tokenizer.eos_token, "") for s in decoded_pred_strs]
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
                line = "\t#####\t".join([i_pred, i_label])
                f.write(line+"\n")
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

    num_examples = 5
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

def collator(batch, max_length=1024):
    """ 适 用于 train和eval
        todo: 要不要处理好label， 使得prompt部分loss 是-100 ?
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
    #decoder_attention_mask[:, 0] = 1
    new_batch["decoder_input_ids"] = decoder_input_ids
    #new_batch["decoder_attention_mask"] = decoder_attention_mask

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


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    logging.info(f"tokenizer's vocab size = {len(processor.tokenizer.vocab)}")

    data_args.train_file = "/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/train/metadata.jsonl"
    data_args.validation_file = "/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/validation/metadata.jsonl"
    training_args.output_dir = "/home/ana/data4/output_models/MyMLLM/100k_p2s_wiki_pretrain_2tasks"

    train_dataset, eval_dataset, test_dataset = None, None, None
    train_dataset = wiki_pretrain_dataset(data_args, os.path.dirname(data_args.train_file), processor, 'train', max_records=10000000)
    logging.info(f"train_dataset size: {len(train_dataset)}")
    eval_dataset = wiki_pretrain_dataset(data_args, os.path.dirname(data_args.validation_file), processor, 'eval', max_records=100)
    logging.info(f"eval_dataset size: {len(eval_dataset)}")
    #padding = "max_length" if data_args.pad_to_max_length else False

    if False:
        # OK: evaluate 部分 测试部分 可以这样做
        model = Pix2StructForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 回头
        model.config.max_length = 2048
        model.config.num_beams = 1
        model_inputs = tokenizer(["请问头衔是?:"],  add_special_tokens=False, return_tensors="pt")

        print(tokenizer.decode(model_inputs['input_ids'][0]) )

        for batch in DataLoader(eval_dataset, batch_size=1):
            batch['input_ids'] =  model_inputs['input_ids']
            print(batch)
            generated_ids = model.generate(**batch, max_new_tokens=128)
            output = tokenizer.batch_decode(generated_ids,)[0]
            break
        print(output)
        sys.exit(0)

    if True:
        # # 构造model
        # 输入上针对LayoutLM
        model = MyCustomPix2StructForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

        training_args.max_steps = 32000
        training_args.gradient_accumulation_steps = 64
        training_args.predict_with_generate = True
        training_args.evaluation_strategy = "steps"
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        training_args.logging_steps = 10
        training_args.save_steps = 1600
        # training_args.eval_steps = 1600
        training_args.eval_steps = 1600
        # training_args.fp16 = True
        training_args.overwrite_output_dir = True
        training_args.generation_max_length = 2048
        training_args.generation_num_beams = 1
        # training_args.remove_unused_columns = False
        training_args.num_train_epochs = 20
        # training_args.set_lr_scheduler(name="cosine", warmup_ratio=0.05)
        training_args.learning_rate = 1e-4

        # 回头
        model.config.max_length = 2048
        model.config.num_beams= 1

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
        trainer.evaluate()
        # trainer.train(resume_from_checkpoint=True)
        # trainer.train()