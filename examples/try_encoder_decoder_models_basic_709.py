"""
    用hf 来组装 encoder  decoder 试水;
    尝试使用 FUNSD 任务来做效果验证
    1. 研究processor的复用
    2. 需要做对应的数据集改动
    3. metrics 改变一下方便对齐

    + 引用自定义的layoutlmv3模型，加载时候亦然;
    + 709 / 512 的attn; 需细扣此部分transformers attn的实现细节

    完成 en版本的基础模型
"""
import kp_setup
import logging
import datasets
import datasets
import os
import sys
from pathlib import Path

# from transformers import EncoderDecoderModel
from transformers import BertTokenizerFast
from transformers import AutoProcessor, AutoModel, AutoConfig
from datasets.features import ClassLabel
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
import torch
from libs.processors.processing_layoutlmv3 import LayoutLMv3Processor
from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader

# 主要是注册 模型 __init__
import libs.models.custom_layoutlmv3

from libs.models.MyEncoderDecoderModelv3 import MyEncoderDecoderModelv3


encoder_model_path = "/home/ana/data4/models/custom-layoutlmv3-base"
decoder_model_path = "/home/ana/data4/models/bert-base-uncased"


processor = LayoutLMv3Processor.from_pretrained(encoder_model_path, apply_ocr=False)
decoder_tokenizer = BertTokenizerFast.from_pretrained(decoder_model_path)


#####################  DATASET  #######
datasets.config.DOWNLOADED_DATASETS_PATH = Path("/home/ana/data4/datasets/transformers_datasets")
dataset = datasets.load_dataset("nielsr/funsd-layoutlmv3")


features = dataset["train"].features
column_names = dataset["train"].column_names
image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "bboxes"
label_column_name = "ner_tags"

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    # No need to convert the labels since they are already ints.
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
else:
    label_list = get_label_list(dataset["train"][label_column_name])
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
num_labels = len(label_list)

def prepare_examples(examples):
    encoder_max_length = 512
    decoder_max_length = 128
    print("HIHIHI")

    images = examples[image_column_name]
    words = examples[text_column_name]
    boxes = examples[boxes_column_name]
    word_labels = examples[label_column_name]
    # 仅挑出 headers , 然后讲对应的token 连起来

    title_labels = []
    # 每一个case
    for i_word_labels, i_words in zip(word_labels, words):
        i_all_labels = []
        tmp_labels = []
        for label, word in zip(i_word_labels, i_words):
            if label in (1, 2):
                tmp_labels.append(word)
            elif len(tmp_labels) != 0:
                i_all_labels.append(" ".join(tmp_labels))
                tmp_labels = []
        if len(tmp_labels) != 0:
            i_all_labels.append(" ".join(tmp_labels))
            tmp_labels = []
        title_labels.append("#".join(i_all_labels))
        # print("==== _word_labels", i_word_labels)
        # print("==== i_words", i_words)
        # print("==== i_titles", "#".join(i_all_labels))
    encoding = processor(images, words, boxes=boxes,
                       truncation=True, padding="max_length")
    batch_labels = decoder_tokenizer(title_labels, padding="max_length", truncation=True, max_length=decoder_max_length).input_ids.copy()
    batch_labels = [[-100 if token == decoder_tokenizer.pad_token_id else token for token in labels] for labels in
                       batch_labels]
    encoding['labels'] = batch_labels
    return encoding

features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
})

train_dataset = dataset["train"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)
eval_dataset = dataset["test"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)

# train_dataset= train_dataset.select(range(32))
# eval_dataset= eval_dataset.select(range(5))

train_dataset.set_format("torch")
eval_dataset.set_format("torch")

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
    if False:
        # # 构造model
        # 输入上针对LayoutLM
        model = MyEncoderDecoderModelv3.from_encoder_decoder_pretrained(
            encoder_model_path, decoder_model_path
        )
        model.config.decoder_start_token_id = decoder_tokenizer.cls_token_id
        model.config.eos_token_id = decoder_tokenizer.sep_token_id
        model.config.pad_token_id = decoder_tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        model.config.max_length = 32
        model.config.min_length = 5
        model.config.no_repeat_ngram_size = 3
        model.config.early_stopping = True
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

        training_args = Seq2SeqTrainingArguments(
            max_steps=2000,
            gradient_accumulation_steps=16,
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            output_dir="./tmp_709",
            logging_steps=5,
            save_steps=10,
            eval_steps=20,
            # logging_steps=1000,
            # save_steps=500,
            # eval_steps=7500,
            # warmup_steps=2000,
            # save_total_limit=3,
        )

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=decoder_tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator
        )
        trainer.train()

    sys.exit(0)


    model = MyEncoderDecoderModelv3.from_pretrained('/home/ana/data4/projects/MyMLLM/examples/tmp_709/checkpoint-1000')
    # exp = train_dataset

    # :w
    # model.generate(**exp)

    #def generate_pred(examples):
    #    # cut off at BERT max length 512
    #    encoder_max_length = 512
    #    decoder_max_length = 128

    #    images = examples[image_column_name]
    #    words = examples[text_column_name]
    #    boxes = examples[boxes_column_name]
    #    word_labels = examples[label_column_name]
    #    # 仅挑出 headers , 然后讲对应的token 连起来

    #    title_labels = []
    #    # 每一个case
    #    for i_word_labels, i_words in zip(word_labels, words):
    #        i_all_labels = []
    #        tmp_labels = []
    #        for label, word in zip(i_word_labels, i_words):
    #            if label in (1, 2):
    #                tmp_labels.append(word)
    #            elif len(tmp_labels) != 0:
    #                i_all_labels.append(" ".join(tmp_labels))
    #                tmp_labels = []
    #        if len(tmp_labels) != 0:
    #            i_all_labels.append(" ".join(tmp_labels))
    #            tmp_labels = []
    #        title_labels.append("#".join(i_all_labels))
    #        # print("==== _word_labels", i_word_labels)
    #        # print("==== i_words", i_words)
    #        # print("==== i_titles", "#".join(i_all_labels))
    #    examples['labels'] = title_labels
    #    encoding = processor(images, words, boxes=boxes,
    #                         truncation=True, padding="max_length")
    #    # batch_labels = decoder_tokenizer(title_labels, padding="max_length", truncation=True,
    #    #                                  max_length=decoder_max_length).input_ids.copy()
    #    # batch_labels = [[-100 if token == decoder_tokenizer.pad_token_id else token for token in labels] for labels in
    #    #                 batch_labels]

    #    # encoding['labels'] = batch_labels
    #    for k in encoding:
    #        if k != 'pixel_values':
    #            encoding[k] = torch.LongTensor(encoding[k])
    #        else:
    #            encoding[k] = torch.FloatTensor(encoding[k])
    #    outputs = model.generate(**encoding)

    #    output_str = decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    #    examples["pred"] = output_str

    #    return examples

    # result_dataset = dataset["test"].select(range(5)).map(generate_pred,
    #                         batch_size=2,
    #                         batched=True
    #             # remove_columns = column_names,
    # )
    # print(result_dataset)

    if True:
        train_loader = DataLoader(train_dataset.select(range(5)), batch_size=2)
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

        print("""-------------- EVAL -----------------""")
        eval_loader = DataLoader(eval_dataset.select(range(5)), batch_size=2)
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
