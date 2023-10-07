"""
    wiki预训练任务组装
    0. mask 一部分 50%
    1. 没有文本输入
"""

import os
import json

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from libs.datasets.dataset_utils import load_pp_ocr_json, load_json_lines_into_list, order_by_tbyx_coord, box_norm
from libs.datasets.image_utils import get_image_wh
import logging
from tqdm import tqdm


from .image_utils import Compose, RandomResizedCropAndInterpolationWithTwoPic


class wiki_multitask_dataset(Dataset):
    def __init__(
            self,
            args,
            file_dir,
            tokenizer,
            decoder_tokenizer,
            mode,
            max_records=None,
    ):
        self.args = args
        self.mode = mode
        self.cur_la = args.language
        self.tokenizer = tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.file_dir = file_dir
        self.max_records = max_records

        self.common_transform = Compose([
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, interpolation=args.train_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor((0.5, 0.5, 0.5)),
                std=torch.tensor((0.5, 0.5, 0.5)))
        ])

        # file_name = os.path.join(args.data_dir, "{}.{}.json".format(self.cur_la, 'train' if mode == 'train' else 'val'))
        file_name = os.path.join(file_dir, "metadata.jsonl")

        self.task1_prompt_cmd = "预训练:填空"
        self.task1_prompt_cmd_token_ids = self.tokenizer(self.task1_prompt_cmd, truncation=False, add_special_tokens=False,
                                                         return_attention_mask=False)['input_ids'].copy() + [
                                              self.tokenizer.sep_token_id]
        self.task2_prompt_cmd = "预训练:识别"
        self.task2_prompt_cmd_token_ids = self.tokenizer(self.task2_prompt_cmd, truncation=False, add_special_tokens=False,
                                                         return_attention_mask=False)['input_ids'].copy() + [
                                              self.tokenizer.sep_token_id]
        logging.info(f"加载数据集: {file_name}, max_records: {self.max_records}")
        self.feature = self.load_data(file_name)

    def get_segment_ids(self, bboxs):
        """
            第几个bbox -> segment_id
            0, 1, 2, 3, ...
        """
        segment_ids = []
        for i in range(len(bboxs)):
            if i == 0:
                segment_ids.append(0)
            else:
                if bboxs[i - 1] == bboxs[i]:
                    segment_ids.append(segment_ids[-1])
                else:
                    segment_ids.append(segment_ids[-1] + 1)
        return segment_ids

    def get_position_ids(self, segment_ids):
        """
            同一个bbox 内的位置信息
            2 开始计算,
            2, 3, 4, ...
        """
        position_ids = []
        for i in range(len(segment_ids)):
            if i == 0:
                position_ids.append(2)
            else:
                if segment_ids[i] == segment_ids[i - 1]:
                    position_ids.append(position_ids[-1] + 1)
                else:
                    position_ids.append(2)
        return position_ids

    def load_data(
            self,
            data_file,
    ):
        """
            json 文件的数据预处理
        """
        data_list = load_json_lines_into_list(data_file)
        ocr_dir = os.path.join(self.file_dir, 'ocr')
        ocr_data = load_pp_ocr_json(ocr_dir, suffix='.ocr.json', with_len4_coord=True)
        logging.info("ocr data loaded")

        # re-org data format
        total_data = {"id": [], "lines": [], "bboxes": [], "ner_tags": [], "image_path": [], "text_labels": []}
        # bbox 级别的数据分组处理
        for i, rec in tqdm(enumerate(data_list), total=len(data_list)):
            if self.max_records is not None and i > self.max_records:
                break
            i_file_id = rec["file_name"]
            i_image_path = os.path.join(self.file_dir, i_file_id)
            i_doc_text_labels = json.loads(rec["ground_truth"])["gt_parse"]["text_sequence"]
            assert os.path.exists(i_image_path)
            #assert i_file_id in ocr_data, f"{i_file_id} not in ocr_data"

            if not i_file_id in ocr_data:
                logging.warning(f"{i_file_id} not in ocr_data, ignore")
                continue

            width, height = get_image_wh(i_image_path)

            i_ocr_data = ocr_data[i_file_id]
            # 要不要排个序?
            sorted_i_coords = order_by_tbyx_coord(i_ocr_data.keys())
            i_texts = [i_ocr_data[i_coord] for i_coord in sorted_i_coords]
            i_coords = [box_norm(coord, width=width, height=height) for coord in sorted_i_coords]

            total_data['id'].append(i_file_id)
            total_data['lines'].append(i_texts)
            total_data['bboxes'].append(i_coords)
            total_data['image_path'].append(i_image_path)
            total_data['text_labels'].append(i_doc_text_labels)

        logging.info(f"task records loaded, proc {len(data_list)} , got {len(total_data['id'])}")

        # tokenize text and get bbox/label
        # 转换为token_id
        total_input_ids, total_bboxs, total_label_ids = [], [], []
        for i in range(len(total_data['lines'])):
            cur_doc_input_ids, cur_doc_bboxs, cur_doc_labels = [], [], []
            # todo: 明确下transformer encoder decoder 架构下 这部分应该怎么处理; 目前实现没有添加 special token
            cur_doc_text_labels = \
                self.decoder_tokenizer(total_data['text_labels'][i], truncation='longest_first',
                                       add_special_tokens=False,
                                       return_attention_mask=False).input_ids.copy()
            # 因为没有限制一定要pad 到最长，所以这里问题应该不大
            cur_doc_text_labels = \
                [-100 if token == self.decoder_tokenizer.pad_token_id else token for token in cur_doc_text_labels]
            for j in range(len(total_data['lines'][i])):
                cur_input_ids = self.tokenizer(total_data['lines'][i][j], truncation=False, add_special_tokens=False,
                                               return_attention_mask=False)['input_ids'].copy()

                if len(cur_input_ids) == 0: continue

                # cur_label = total_data['ner_tags'][i][j].upper()
                # if cur_label == 'OTHER':
                #     cur_labels = ["O"] * len(cur_input_ids)
                #     for k in range(len(cur_labels)):
                #         cur_labels[k] = self.label2ids[cur_labels[k]]
                # else:
                #     cur_labels = [cur_label] * len(cur_input_ids)
                #     cur_labels[0] = self.label2ids['B-' + cur_labels[0]]
                #     for k in range(1, len(cur_labels)):
                #         cur_labels[k] = self.label2ids['I-' + cur_labels[k]]
                # assert len(cur_input_ids) == len([total_data['bboxes'][i][j]] * len(cur_input_ids)) == len(cur_labels)
                cur_doc_input_ids += cur_input_ids
                cur_doc_bboxs += [total_data['bboxes'][i][j]] * len(cur_input_ids)
                # cur_doc_labels += cur_labels
            assert len(cur_doc_input_ids) == len(cur_doc_bboxs)
            # == len(cur_doc_labels)
            assert len(cur_doc_input_ids) > 0

            total_input_ids.append(cur_doc_input_ids)
            total_bboxs.append(cur_doc_bboxs)
            # total_label_ids.append(cur_doc_labels)
            total_label_ids.append(cur_doc_text_labels)
        assert len(total_input_ids) == len(total_bboxs) == len(total_label_ids)

        # split text to several slices because of over-length
        # 在这里直接截断即可
        input_ids, bboxs, labels = [], [], []
        segment_ids, position_ids = [], []
        task_types = []
        image_path = []
        for i in range(len(total_input_ids)):
            start = 0
            cur_iter = 0
            while start < len(total_input_ids[i]):
                end = min(start + 510, len(total_input_ids[i]))
                # 先做个单任务 只有 命令的 任务, 看能否识别出文字
                #
                if True:
                    task2_input_ids = [self.tokenizer.cls_token_id] + self.task2_prompt_cmd_token_ids \
                                      + [self.tokenizer.sep_token_id]
                    input_ids.append(task2_input_ids)
                    bboxs.append([[0, 0, 0, 0]] * (len(task2_input_ids)-1) + [[1000, 1000, 1000, 1000]])
                    labels.append(total_label_ids[i])
                    cur_segment_ids = self.get_segment_ids(bboxs[-1])
                    cur_position_ids = self.get_position_ids(cur_segment_ids)
                    segment_ids.append(cur_segment_ids)
                    position_ids.append(cur_position_ids)
                    image_path.append(total_data['image_path'][i])
                    task_types.append(2)

                if True and False:
                    # task_type 1:  需要在collator 做mask
                    task1_input_ids = [self.tokenizer.cls_token_id] + self.task1_prompt_cmd_token_ids + \
                                      total_input_ids[i][start: end] + [self.tokenizer.sep_token_id]
                    prefix_len = len(self.task1_prompt_cmd_token_ids) + 1

                    input_ids.append(task1_input_ids)
                    bboxs.append([[0, 0, 0, 0]] * prefix_len + total_bboxs[i][start: end] + [[1000, 1000, 1000, 1000]])

                    # labels.append([-100] + total_label_ids[i][start: end] + [-100])
                    labels.append(total_label_ids[i])

                    cur_segment_ids = self.get_segment_ids(bboxs[-1])
                    cur_position_ids = self.get_position_ids(cur_segment_ids)
                    segment_ids.append(cur_segment_ids)
                    position_ids.append(cur_position_ids)
                    image_path.append(total_data['image_path'][i])
                    task_types.append(1)

                start = end
                cur_iter += 1
                # 仅要前510 长
                break

        assert len(input_ids) == len(bboxs) == len(labels) == len(segment_ids) == len(position_ids)
        assert len(segment_ids) == len(image_path)
        logging.info(f"dataset size loaded: {len(input_ids)}")

        res = {
            'input_ids': input_ids,
            'bbox': bboxs,
            'labels': labels,
            'segment_ids': segment_ids,
            'position_ids': position_ids,
            'image_path': image_path,
            'task_types': task_types,
        }
        return res

    def __len__(self):
        return len(self.feature['input_ids'])

    def __getitem__(self, index):
        input_ids = self.feature["input_ids"][index]

        # attention_mask = self.feature["attention_mask"][index]
        attention_mask = [1] * len(input_ids)
        labels = self.feature["labels"][index]
        bbox = self.feature["bbox"][index]
        segment_ids = self.feature['segment_ids'][index]
        position_ids = self.feature['position_ids'][index]

        img = pil_loader(self.feature['image_path'][index])
        for_patches, _ = self.common_transform(img, augmentation=False)
        patch = self.patch_transform(for_patches)
        task_type = self.feature['task_types'][index]

        # assert len(input_ids) == len(attention_mask) == len(labels) == len(bbox) == len(segment_ids)
        assert len(input_ids) == len(attention_mask) == len(bbox) == len(segment_ids)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "bbox": bbox,
            "segment_ids": segment_ids,
            "position_ids": position_ids,
            "images": patch,
            "task_type": task_type,
        }
        return res


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
