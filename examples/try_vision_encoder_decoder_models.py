"""
    用hf 来组装 encoder  decoder 试水;
    尝试使用 FUNSD 任务来做效果验证
    1. 研究processor的复用
    2. 需要做对应的数据集改动
    3. metrics 改变一下方便对齐
"""
import kp_setup
import logging
import datasets
import datasets
from pathlib import Path

from transformers import VisionEncoderDecoderModel, EncoderDecoderModel

from transformers import GPT2TokenizerFast, VisionEncoderDecoderModel
from transformers import AutoProcessor
from datasets.features import ClassLabel
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
import torch
from libs.processors.processing_layoutlmv3 import LayoutLMv3Processor
from libs.models.MyEncoderDecoderModel import MyEncoderDecoderModel

from  transformers.models.layoutlmv3 import LayoutLMv3Model



encoder_model_path = "/home/ana/data4/models/layoutlmv3-base"
decoder_model_path = "/home/ana/data4/models/gpt2"
processor = LayoutLMv3Processor.from_pretrained(encoder_model_path, apply_ocr=False)
decoder_tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model_path)

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
    labels = decoder_tokenizer(title_labels, padding="max_length").input_ids
    encoding['labels'] = labels
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
train_dataset.set_format("torch")


if __name__ == '__main__':
    # 构造model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "/home/ana/data4/models/layoutlmv3-base", "/home/ana/data4/models/gpt2"
    )

    # 怎么构造一个样例?
    for rec in train_dataset.select([0, 1, 2]):
        loss = model(**rec).loss
