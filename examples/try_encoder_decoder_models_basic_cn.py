"""
    完成 cn 版本的基础模型
    加载unilm版本 的 layoutlmv3
    decoder 暂时先用 chinese roberta
    # 试训练 xfund
"""
import kp_setup
import logging
import datasets
import os
import sys
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
)

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from libs.processors.processing_layoutlmv3 import LayoutLMv3Processor
from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader

# custom models
import libs.models.custom_unilm_layoutlmv3
from libs.models.MyEncoderDecoderModelv4 import MyEncoderDecoderModelv4

from libs.datasets.unilm_layoutlmv3.data_collator import DataCollatorForSeq2Seq
from libs.datasets.unilm_layoutlmv3.xfund_for_gen import xfund_dataset, XFund_label2ids

encoder_model_path = "/home/ana/data2/models/custom-layoutlmv3-base-chinese"
decoder_model_path = "/home/ana/data2/models/chinese-roberta-wwm-ext"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=encoder_model_path,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    encoder_model_name_or_path: str = field(
        default=encoder_model_path,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    decoder_model_name_or_path: str = field(
        default=decoder_model_path,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
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
        default="/home/ana/data2/datasets/XFUND/zh.train.json",
        metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default="/home/ana/data2/datasets/XFUND/zh.val.json",
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default="/home/ana/data2/datasets/XFUND/zh.val.json",
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
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
    # label_all_tokens: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
    #                 "one (in which case the other tokens will have a padding index)."
    #     },
    # )
    # return_entity_level_metrics: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    # )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default="/home/ana/data2/datasets/XFUND")
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    # 这个实现里应该没有这个
    # second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    # 这部分数值貌似model_dir 有保存的
    # imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})


@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    output_dir: Optional[str] = field(
        default="/home/ana/data4/output_models/MyMLLM/tmp_cn_709",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


##################  metrics ###############
rouge = datasets.load_metric(os.path.join(kp_setup.lib_dir, 'rouge.py'))


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = decoder_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = decoder_tokenizer.pad_token_id
    label_str = decoder_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##############################################
    # 先处理好数据集
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        encoder_model_path,
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
        use_fast=True,
        add_prefix_space=True
    )
    decoder_tokenizer = BertTokenizerFast.from_pretrained(decoder_model_path)

    train_dataset, eval_dataset, test_dataset = None, None, None
    train_dataset = xfund_dataset(data_args, encoder_tokenizer, decoder_tokenizer, 'train')
    eval_dataset = xfund_dataset(data_args, encoder_tokenizer, decoder_tokenizer, 'eval')
    padding = "max_length" if data_args.pad_to_max_length else False

    if False:
        # # 构造model
        # 输入上针对LayoutLM
        model = MyEncoderDecoderModelv4.from_encoder_decoder_pretrained(
            encoder_model_path, decoder_model_path
        )

        model.config.decoder_start_token_id = decoder_tokenizer.cls_token_id
        model.config.eos_token_id = decoder_tokenizer.sep_token_id
        model.config.pad_token_id = decoder_tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        model.config.max_length = 64
        model.config.min_length = 5
        model.config.no_repeat_ngram_size = 3
        model.config.early_stopping = True
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

        training_args.max_steps = 2000
        training_args.gradient_accumulation_steps = 32
        training_args.predict_with_generate = True
        training_args.evaluation_strategy = "steps"
        training_args.per_device_train_batch_size = 8
        training_args.per_device_eval_batch_size = 2
        training_args.logging_steps = 5
        training_args.save_steps = 10
        training_args.eval_steps = 1
        training_args.fp16 = True
        training_args.overwrite_output_dir = True

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            model=model,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            padding=padding,
            max_length=512,
            decoder_max_length=model.config.max_length,
        )


        print(training_args)

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=decoder_tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        trainer.train()

    # 尝试使用
    if True:
        model_path = '/home/ana/data4/output_models/MyMLLM/tmp_cn_709/checkpoint-100'
        model = MyEncoderDecoderModelv4.from_pretrained(
            model_path
        )

        model.config.decoder_start_token_id = decoder_tokenizer.cls_token_id
        model.config.eos_token_id = decoder_tokenizer.sep_token_id
        model.config.pad_token_id = decoder_tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        model.config.max_length = 64
        model.config.min_length = 5
        model.config.no_repeat_ngram_size = 3
        model.config.early_stopping = True
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            model=model,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            padding=padding,
            max_length=512,
            decoder_max_length=model.config.max_length,
        )

        train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)
        i = 0
        for batch in train_loader:
            encoded_labels = batch['labels'].cpu().numpy()
            del batch['labels']
            out = model.generate(**batch)
            preds = decoder_tokenizer.batch_decode(out, skip_special_tokens=True)
            encoded_labels[encoded_labels == -100] = 0
            labels = decoder_tokenizer.batch_decode(encoded_labels, skip_special_tokens=True)
            for pred, label in zip(preds, labels):
                print("-" * 100)
                print(f"Label: {label}".rstrip("[PAD]"))
                print(f"Pred: {pred}".rstrip("[PAD]"))
            i += 1
            if i >= 5:
                break

        print("""-------------- EVAL -----------------""")
        i = 0
        eval_loader = DataLoader(eval_dataset, batch_size=2, collate_fn=data_collator)
        for batch in eval_loader:
            encoded_labels = batch['labels'].cpu().numpy()
            del batch['labels']
            out = model.generate(**batch)
            preds = decoder_tokenizer.batch_decode(out, skip_special_tokens=True)
            encoded_labels[encoded_labels == -100] = 0
            labels = decoder_tokenizer.batch_decode(encoded_labels, skip_special_tokens=True)
            for pred, label in zip(preds, labels):
                print("-" * 100)
                print(f"Label: {label}".rstrip("[PAD]"))
                print(f"Pred: {pred}".rstrip("[PAD]"))

            i += 1
            if i >= 5:
                break




