"""
用 pix2struct 做wiki语料预训练 看效果
    完成 cn 版本的基础模型
    # 语言模型预训练, OCR 能力训练
    # 1. 只有图像输入, 进行文本生成预测, 生成所有文本  ##
    todo
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

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from libs.processors.processing_layoutlmv3 import LayoutLMv3Processor
from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# custom models
import libs.models.custom_unilm_layoutlmv3
from libs.models.MyEncoderDecoderModelv4 import MyEncoderDecoderModelv4

from libs.datasets.wiki_warmup import wiki_warmup_dataset
from libs import utils

import distance
from rouge import Rouge
import pickle


# model_path = "/home/ana/data4/models/pix2struct-base"
model_path = "/home/ana/data4/models/pix2struct-base-enlarge-vocab"


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
    decoder_max_length: int = field(default=512, metadata={"help": "生成文本长度"})


@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    output_dir: Optional[str] = field(
        default="/home/ana/data4/output_models/MyMLLM/100k_p2s_prerain_wo_ocr_lvocab",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


##################  metrics ###############
# rouge = datasets.load_metric(os.path.join(kp_setup.lib_dir, 'rouge.py'))
rouge_obj = Rouge()


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    #dump for debug
    try:
        utils.make_sure_dir_there(training_args.output_dir)
        with open(os.path.join(training_args.output_dir, 'eval_token_out.txt'), 'wb') as f:
            pickle.dump([labels_ids, pred_ids], f)
    except Exception as e:
        logging.error(e)

    # pred trainer 里面 默认 pad -100,(对pred)
    # todo
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    sep_pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    sep_label_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    #  bert tokenizer decode 会带空格，要注意
    if False:
        label_str = ["#" if s.strip() == "" else "".join(s.split(" ")) for s in sep_label_str]
        pred_str = ["#" if s.strip() == "" else "".join(s.split(" ")) for s in sep_pred_str]
    else:
        label_str = ["#" if s.strip() == "" else s for s in sep_label_str]
        pred_str = ["#" if s.strip() == "" else s for s in sep_pred_str]
        sep_pred_str = [" ".join(x) for x in pred_str]
        sep_label_str = [" ".join(x) for x in label_str]

    sum_distance_ratio = 0
    for i_pred, i_label in zip(pred_str, label_str):
        ned = distance.levenshtein(i_pred, i_label, normalized=True)
        sum_distance_ratio += ned

    # sep_pred_str = [" ".join(x) for x in pred_str]
    # sep_label_str = [" ".join(x) for x in label_str]

    levenshtein_distance_ratio = sum_distance_ratio / len(pred_str)

    #  rouge一句不知道为啥有个bug， 观察
    try:
        with open(os.path.join(training_args.output_dir, 'eval_out.txt'), 'w') as f:
            for i_pred, i_label in zip(pred_str, label_str):
                line = "\t#####\t".join([i_pred, i_label])
                f.write(line+"\n")
    except Exception as e:
        logging.error(e)
    #
    rouge_scores = rouge_obj.get_scores(sep_pred_str, sep_label_str, avg=True)
    bleu_score_corpus = corpus_bleu([[s] for s in sep_label_str], sep_pred_str)


    num_examples = 5
    logging.info(f"EVAL examples: {num_examples}")
    for i_pred, i_label in zip(pred_str, label_str):
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
    return scores


def collator(batch):
    new_batch = {"flattened_patches": [], "attention_mask": []}
    texts = [item["labels"] for item in batch]

    text_inputs = processor(text=texts, padding=True, return_tensors="pt", add_special_tokens=True,
                            max_length=1024, truncation=True)

    new_batch["labels"] = text_inputs.input_ids

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

    data_args.train_file = "/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/train/metadata.jsonl"
    data_args.validation_file = "/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/validation/metadata.jsonl"
    # training_args.output_dir = "/home/ana/data4/output_models/MyMLLM/100k_p2s_prerain_wo_ocr_lvocab"

    train_dataset, eval_dataset, test_dataset = None, None, None
    train_dataset = wiki_warmup_dataset(data_args, os.path.dirname(data_args.train_file), processor, 'train')
    logging.info(f"train_dataset size: {len(train_dataset)}")
    eval_dataset = wiki_warmup_dataset(data_args, os.path.dirname(data_args.validation_file), processor, 'eval', max_records=200)
    logging.info(f"eval_dataset size: {len(eval_dataset)}")
    #padding = "max_length" if data_args.pad_to_max_length else False

    if True:
        # # 构造model
        # 输入上针对LayoutLM
        model = Pix2StructForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

        training_args.max_steps = 100000
        training_args.gradient_accumulation_steps = 32
        training_args.predict_with_generate = True
        training_args.evaluation_strategy = "steps"
        training_args.per_device_train_batch_size = 2
        training_args.per_device_eval_batch_size = 2
        training_args.logging_steps = 10
        training_args.save_steps = 1600
        training_args.eval_steps = 1600
        # training_args.eval_steps = 1
        training_args.fp16 = True
        training_args.overwrite_output_dir = True
        training_args.generation_max_length = 1024
        training_args.generation_num_beams = 1

        # 回头
        model.config.max_length = 1024
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
        trainer.train()

