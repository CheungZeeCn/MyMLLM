"""
    wiki预训练任务组装
    拼凑给pix2struct 模型
    只需要传入任务文件和对应的processor，在get item 的时候 会对任务对应的文件做处理

    updated: 加入对 prompt 的支持, 会区分是在train 环节 还是 eval 环节
"""
import os
import json

from torch.utils.data.dataset import Dataset
from libs.datasets.dataset_utils import load_json_lines_into_list
import logging
from tqdm import tqdm
from PIL import Image


class wiki_warmup_dataset(Dataset):
    def __init__(
            self,
            args,
            file_dir,
            processor,
            mode,
            max_records=None,
            max_patches=1024,
            prompt=""
    ):
        self.args = args
        self.mode = mode
        self.processor = processor
        self.file_dir = file_dir
        self.max_records = max_records
        self.max_patches = max_patches
        self.prompt = prompt

        # file_name = os.path.join(args.data_dir, "{}.{}.json".format(self.cur_la, 'train' if mode == 'train' else 'val'))
        file_name = os.path.join(file_dir, "metadata.jsonl")
        logging.info(f"加载数据集: {file_name}")
        self.data = self.load_task(file_name)

    def load_task(
            self,
            data_file,
    ):
        """
            json 文件的数据预处理
        """
        data_list = load_json_lines_into_list(data_file)
        # re-org data format
        ret_data = []
        # bbox 级别的数据分组处理
        for i, rec in tqdm(enumerate(data_list), total=len(data_list)):
            if self.max_records is not None and i > self.max_records:
                break
            i_file_id = rec["file_name"]
            i_image_path = os.path.join(self.file_dir, i_file_id)
            i_doc_text_labels = json.loads(rec["ground_truth"])["gt_parse"]["text_sequence"]
            assert os.path.exists(i_image_path)
            # assert i_file_id in ocr_data, f"{i_file_id} not in ocr_data"
            i_data = {}

            i_data['id'] = i_file_id
            # i_data['image'] = pil_loader(i_image_path)
            i_data['image'] = i_image_path
            i_data['labels'] = i_doc_text_labels

            ret_data.append(i_data)
        logging.info(f"task records loaded, all {len(data_list)}, got {len(ret_data)}")
        return ret_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # on the run
        image = pil_loader(item['image'])
        # encoding = self.processor(images=item["image"], return_tensors="pt", add_special_tokens=True,
        #                           max_patches=self.max_patches)
        encoding = self.processor(images=image, return_tensors="pt", add_special_tokens=True,
                                  max_patches=self.max_patches)
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding['prompt'] = self.prompt
        encoding["labels"] = self.prompt + item["labels"]
        return encoding


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
