"""
    预训练任务 基类数据集
    带prompt的预训练任务

    有以下特性:
        设定长度
        支持多种任务, 可以扩展;
            1. 类似donut的ocr, 识别文字
            2. 类似udop的 layout任务
        对于train:
            对每一个样例，都可以用一定的比例生成不同任务的record, 实现在get里面
        对于 eval 和 test,
            任务的生成的固定的, 针对一个record, 会固定生成所有任务类型的记录
"""
import json
import logging
import os
import random
from collections import Counter

from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import pickle

from libs.datasets.dataset_utils import load_json_lines_into_list
from libs.datasets.image_utils import pil_loader

from libs.datasets.dataset_utils import load_single_pp_ocr_json, load_json_lines_into_list, \
    get_word_layouts, filter_bboxes_by_ratio, collect_mask_spans, load_pa_ocr_in_dict
from libs.datasets.image_utils import get_image_size


class BasePretrainDataset(Dataset):
    def __init__(
            self,
            args,
            file_name,
            processor,
            # 是在训练还是在预测 ?
            mode,
            max_records=None,
            max_patches=2048,
            image_dir='./',
            load_cache=True,
            data_dir=None,
            ocr_format='auto',
            task_weights=(0.5, 0.5),
            cache_tag=None,
            **kwargs
    ):
        self.args = args
        # train eval test
        self.mode = mode
        self.processor = processor
        self.max_records = max_records
        self.max_patches = max_patches
        self.load_cache = load_cache
        self.ocr_format = ocr_format
        self.cache_tag = cache_tag

        if data_dir is None:
            data_dir = os.path.dirname(file_name)
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_dir)

        self.file_name = file_name
        logging.info(f"将加载数据集: {file_name}")

        # 默认预训练任务集合
        self.tasks = [['按顺序识别图片中的文字: ', task_weights[0]],
                      ['根据上下文预测文字和位置: ', task_weights[1]],
                      ]
        self.task_weights = task_weights
        self.task_keys = list(range(len(self.tasks)))
        self.dev_task_counter = Counter()

        self.ocr_data = {}

        self.load()

    def load(self):
        num = 'all'
        if self.max_records is not None:
            num = f'{self.max_records}'
        # load_cache or load raw_file
        if self.cache_tag is None:
            cache_file = self.file_name + f'.{self.__class__.__name__}.' + self.mode + '.'+ num + '.pickle'
        else:
            cache_file = self.file_name + f'.{self.__class__.__name__}.' + self.cache_tag + '.' + self.mode + '.'+ num + '.pickle'
        if self.load_cache is True:
            if not os.path.exists(cache_file):
                logging.warning(f"cache file f{cache_file} do not exists! call load_task()")
                self.data = self.load_task(self.file_name)
                self.dump_cache_file(self.data, cache_file)
                logging.info(f'cache file is dumped to: {cache_file}')
            else:
                logging.info(f"load cache file {cache_file}")
                self.data = self.load_cache_file(cache_file)
        else:
            self.data = self.load_task(self.file_name)
            self.dump_cache_file(self.data, cache_file)
            logging.info(f'cache file is dumped to: {cache_file}')

    def load_cache_file(self, cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def dump_cache_file(self, data, cache_file):
        with open(cache_file, 'wb') as f:
            ret = pickle.dump(data, f)
        return ret

    def gen_task_label(self, task):
        return task['label']

    def gen_task_prompt(self, task):
        return self.prompt

    def gen_task_image_path(self, task):
        return os.path.join(self.image_dir, task['file_name'])

    def get_ocr_path(self, task):
        ocr_dir = os.path.join(self.data_dir, 'ocr')

        def ocr_file_exists(suffix):
            return os.path.exists(os.path.join(ocr_dir, task['file_name'] + suffix))

        ocr_format = self.ocr_format
        suffix = '.ocr.json'
        if ocr_format == 'json':
            suffix = '.ocr.json'
            if not ocr_file_exists(suffix):
                suffix = '.json'
            if not ocr_file_exists(suffix):
                suffix = ''
            if not ocr_file_exists(suffix):
                return None, None, None
        elif ocr_format == 'txt':
            suffix = '.ocr.txt'
            if not ocr_file_exists(suffix):
                suffix = '.txt'
            if not ocr_file_exists(suffix):
                suffix = ''
            if not ocr_file_exists(suffix):
                return None, None, None
        else:  # auto
            ocr_format = 'json'
            suffix = '.ocr.json'
            if not ocr_file_exists(suffix):
                suffix = '.json'
            if not ocr_file_exists(suffix):
                ocr_format = 'txt'
                suffix = '.ocr.txt'
            if not ocr_file_exists(suffix):
                suffix = '.txt'
            if not ocr_file_exists(suffix):
                return None, None, None
        return os.path.join(ocr_dir, task['file_name'] + suffix), ocr_format, suffix

    def load_single_ocr_data(self, file_path, ocr_format='json'):
        if ocr_format == 'json':  # pp json
            try:
                ocr_data = load_single_pp_ocr_json(file_path, with_len4_coord=True)
            except Exception as e:
                logging.error(f"load_single_pp_ocr_json: {file_path} error, {e}", exc_info=True)
                return None
            return ocr_data
        elif ocr_format == 'txt':  # pa txt
            try:
                ocr_data = load_pa_ocr_in_dict(file_path, text_only=True, with_len4_coord=True)
            except Exception as e:
                logging.error(f"load_pa_ocr_in_dict: {file_path} error, {e}", exc_info=True)
                return None
            return ocr_data
        raise NotImplemented()

    @staticmethod
    def gen_layout_texts_tasks(file_id, bboxes, text, size, ratio=0.30, max_prompt_length=1024, max_label_length=1024):
        """
            首先过滤出多次出现的bbox(主要是简化了, 可以不做对齐过程)
            随机mask掉30%的内容，然后针对每个bbox里面的每个span，进行在layout text 任务子类型内按顺序选择
            todo: 支持 wwm ? bbox和label text 对齐
        """
        task_counters = {
            'layout': 0,
            'text': 0,
            'text_layout': 0
        }
        g_task_types = list(task_counters.keys())
        g_task_type_index = 0
        width, height = size

        def choose_a_type():
            nonlocal g_task_type_index
            i_type = g_task_types[g_task_type_index % len(g_task_types)]
            g_task_type_index += 1
            return i_type

        def choose_a_type_random():
            i_type = random.choices(g_task_types, k=1)[0]
            return i_type

        def joined_bbox_text(sorted_coord, candidate_bboxes):
            joined_bbox_text = ""
            for coord in sorted_coord:
                candidate_bboxes[coord] = candidate_bboxes[coord] + [len(joined_bbox_text), len(joined_bbox_text) + len(
                    candidate_bboxes[coord][0]), [0] * len(candidate_bboxes[coord][0])]
                joined_bbox_text += candidate_bboxes[coord][0]
            return joined_bbox_text, candidate_bboxes

        def do_mask(text, joined_bbox_text, sorted_coord, candidate_bboxes):
            """
                text:  整个文档的文字原文
                joined_bbox_text: 被候选的bbox内的文本的join

                return: task_text, label_text, candidate_bboxes, filtered_bboxes

                todo: 如果要做wwm， 就可以升级这里
            """
            mask_len = int(len(text) * 0.3)
            indexes = list(range(len(joined_bbox_text)))
            random.shuffle(indexes)
            mask_indexes = sorted(indexes[:mask_len])
            # update candidate_bboxes 里面的内容，设置flag 标志位
            for mask_index in mask_indexes:
                for coord in sorted_coord:
                    i_begin = candidate_bboxes[coord][3]
                    i_end = candidate_bboxes[coord][4]
                    if mask_index >= i_begin and mask_index < i_end:
                        # 找到了
                        candidate_bboxes[coord][5][mask_index - i_begin] = 1
            return candidate_bboxes

        def gen_task_texts(candidate_bboxes, sorted_coords, width, height):
            # 开始生成
            # 会更新所有的 candidate_bboxes
            for coord in sorted_coords:
                mask_flags = candidate_bboxes[coord][5]
                ori_text = candidate_bboxes[coord][0]
                word_layouts = get_word_layouts(coord, ori_text)
                # 检查有多少个连续片段
                mask_spans = collect_mask_spans(mask_flags)
                bbox_task_text = ""
                bbox_label_text = ""
                last_text_index = 0
                mask_span = None
                # 对bbox内的每个span
                for mask_span in mask_spans:
                    task_type = choose_a_type_random()
                    # print(task_type)
                    if mask_span[0] != last_text_index:
                        bbox_task_text += ori_text[last_text_index:mask_span[0]]
                        last_text_index = mask_span[1]

                    # span_layout
                    span_layouts = word_layouts[mask_span[0]:mask_span[1]]
                    span_layout = span_layouts[0]['coord'][:2] + span_layouts[-1]['coord'][2:]
                    span_loc = [int(span_layout[0] / width * 500), int(span_layout[1] / height * 500),
                                int(span_layout[2] / width * 500), int(span_layout[3] / height * 500)]
                    # 任务部分
                    task_type_index = task_counters[task_type]
                    if task_type_index >= 100:
                        logging.info(f"task_type: {task_type} > 100 aleady, ignore this one")
                        bbox_task_text += ori_text[mask_span[0]:mask_span[1]]
                        last_text_index = mask_span[1]
                        continue
                    if task_type == 'layout':
                        # task
                        task_text = f"<layout_{task_type_index}>{ori_text[mask_span[0]:mask_span[1]]}</layout_{task_type_index}>"
                        bbox_task_text += task_text
                        # label
                        task_label_text = f"<layout_{task_type_index}><loc_{span_loc[0]}><loc_{span_loc[1]}><loc_{span_loc[2]}><loc_{span_loc[3]}>"
                        bbox_label_text += task_label_text
                    elif task_type == 'text':
                        # task
                        task_text = f"<text_{task_type_index}><loc_{span_loc[0]}><loc_{span_loc[1]}><loc_{span_loc[2]}><loc_{span_loc[3]}></text_{task_type_index}>"
                        bbox_task_text += task_text
                        # label
                        task_label_text = f"<text_{task_type_index}>{ori_text[mask_span[0]:mask_span[1]]}"
                        bbox_label_text += task_label_text
                    elif task_type == 'text_layout':
                        # task
                        task_text = f"<text_layout_{task_type_index}>"
                        bbox_task_text += task_text
                        # label
                        task_label_text = f"<text_layout_{task_type_index}>{ori_text[mask_span[0]:mask_span[1]]}<loc_{span_loc[0]}><loc_{span_loc[1]}><loc_{span_loc[2]}><loc_{span_loc[3]}>"
                        bbox_label_text += task_label_text
                    else:
                        last_text_index = mask_span[1]
                        continue
                    last_text_index = mask_span[1]
                    task_counters[task_type] += 1
                    # 补充添加后面部分

                # 如果这个bbox 没有span
                if mask_span is None:
                    bbox_task_text += ori_text
                elif mask_span[1] != len(ori_text):
                    bbox_task_text += ori_text[mask_span[1]:]

                candidate_bboxes[coord].append(bbox_task_text)
                candidate_bboxes[coord].append(bbox_label_text)
            return candidate_bboxes

        # 过滤bbox
        candidate_bboxes, filtered_bboxes = filter_bboxes_by_ratio(bboxes, text, ratio=ratio)
        # 不符合当前需求， 只能退化成一个最简答的问题了:
        if len(candidate_bboxes) == 0:
            logging.warning(f"{file_id}无符合要求的bbox，退化为整个大框下按顺序识别文本")
            prompt = "根据上下文预测文字和位置: " + "<text_0><loc_0><loc_0><loc_500><loc_500></text_0>"
            label = "<text_0>" + text[:max_label_length+max_prompt_length-len(prompt)-3]
            return prompt, label, 1

        # 将bbox coord 按照在原文中出现顺序排序
        sorted_coords = sorted(candidate_bboxes.keys(), key=lambda x: candidate_bboxes[x][1])
        # 计算, joined_bbox_text， 并update candidate_bboxes
        joined_bbox_text, candidate_bboxes = joined_bbox_text(sorted_coords, candidate_bboxes)
        # 开始要做mask
        candidate_bboxes = do_mask(text, joined_bbox_text, sorted_coords, candidate_bboxes)
        # 生成 文本
        candidate_bboxes = gen_task_texts(candidate_bboxes, sorted_coords, width, height)

        # 组装最后的prompt 和 label:
        prompt = "根据上下文预测文字和位置: "
        label = ""

        last_text_index = 0
        ori_end = None
        break_flag = False
        for coord in sorted_coords:
            ori_begin = candidate_bboxes[coord][1]
            ori_end = candidate_bboxes[coord][2]
            task_text = candidate_bboxes[coord][6]
            label_text = candidate_bboxes[coord][7]
            if ori_begin != last_text_index:
                prompt += text[last_text_index:ori_begin]

            if len(prompt + task_text) >= max_prompt_length or len(label + label_text) >= max_label_length:
                break

            prompt += task_text
            label += label_text
            last_text_index = ori_end

        # if ori_end is None:
        #     print(ori_end)
        if ori_end != len(text) and break_flag is not True:
            prompt += text[ori_end:]
        prompt = prompt[:max_prompt_length]
        label = label[:max_label_length]
        return prompt, label, 0

    def gen_task_record(self, i, task):
        '''
            生成rec 异常返回 None
        '''
        # 图片
        i_image_path = self.gen_task_image_path(task)
        assert os.path.exists(i_image_path), f"{i_image_path} is not exists, please check it"

        # OCR
        # ocr 路径:
        i_ocr_path, i_ocr_format, i_ocr_suffix = self.get_ocr_path(task)
        if i_ocr_path is None:
            logging.warning(f"NO OCR record for {task['file_name']}, ignore!")
            return None

        # ocr 内容: 酌情而定，有可能会爆内存, 要先测试一下, 如果爆内存就放到get item 去做
        i_ocr_data = self.load_single_ocr_data(i_ocr_path, ocr_format=i_ocr_format)
        if i_ocr_path is None:
            logging.warning(f"ERROR OCR record for {task['file_name']}, ignore!")
            return None
        self.ocr_data[task['file_name']] = i_ocr_data

        # 任务数据
        if self.mode == 'train':
            # 先保存数据就够了，训练任务是实时生成的
            i_data = {}
            i_data['id'] = task["file_name"]
            i_data['image'] = i_image_path
            i_data['text'] = json.loads(task["ground_truth"])["gt_parse"]["text_sequence"]
            i_data['ocr_path'] = i_ocr_path
            i_data['ocr_data'] = i_ocr_data
            i_data['ori'] = json.dumps(task)
            return i_data
        else:
            # 多种任务都生成一次
            i_data_list = []
            # 这部分先手写
            ### donut style
            i_data = {}
            task_type = 0
            # donut style
            i_data['id'] = task["file_name"]
            i_data['image'] = i_image_path
            i_doc_text_labels = json.loads(task["ground_truth"])["gt_parse"]["text_sequence"]
            i_prompt, i_labels = self.dynamic_gen_record_prompt_and_labels_task0(i_doc_text_labels)
            # i_data['prompt'] = self.tasks[self.task_keys[task_type]][0]
            i_data['prompt'] = i_prompt
            # i_can_add_len = self.args.decoder_max_length - len(i_data['prompt'])
            # i_data['labels'] = i_doc_text_labels[:i_can_add_len]
            i_data['labels'] = i_labels
            i_data['task_type'] = task_type
            i_data['ori'] = json.dumps(task)
            i_data_list.append(i_data)

            ### udop style layout
            i_data = {}
            task_type = 1
            i_doc_text_labels = json.loads(task["ground_truth"])["gt_parse"]["text_sequence"]
            i_data['id'] = f'{task["file_name"]}_{task_type}'
            i_data['image'] = i_image_path
            # ocr
            i_ocr_data = self.ocr_data[task['file_name']]
            #
            # image_size = get_image_size(i_image_path)
            # i_prompt, i_label, i_status = self.gen_layout_texts_tasks(task['file_name'], i_ocr_data, i_doc_text_labels, image_size,
            #                                                     self.args.decoder_max_input_length,
            #                                                     self.args.decoder_max_length)
            i_prompt, i_labels, i_status = self.dynamic_gen_record_prompt_and_labels_task1(task['file_name'],
                                                                                           i_doc_text_labels,
                                                                                           i_ocr_data, i_image_path)
            self.dev_task_counter[f"1_{i_status}"] += 1
            # i_data['prompt'] = self.tasks[self.task_keys[task_type]][0]
            # 简单用字符串做一个cut
            i_data['prompt'] = i_prompt
            i_data['labels'] = i_labels
            i_data['task_type'] = task_type
            i_data['ori'] = json.dumps(task)
            i_data_list.append(i_data)
            return i_data_list

    def decide_data_format(self, file_name: str):
        if file_name.endswith('.txt'):
            return 'txt'
        elif file_name.endswith('.json'):
            return 'json'
        elif file_name.endswith('.jsonl'):
            return 'jsonl'
        else:
            raise NotImplemented("UNKNOWN data format")

    def load_task(
            self,
            data_file,
    ):
        """
            基础文件加载
                然后生成任务:
                    ori: 原始任务描述
                    image: 文件路径
                    prompt: 调用gen_task_prompt 生成prompt
                    labels: 调用gen_task_label 生成label
            这里默认实现是盖章任务
        """
        """
            默认是 jsonlines 文件的数据预处理
        """
        data_list = load_json_lines_into_list(data_file)
        # re-org data format
        ret_data = []
        for i, rec in tqdm(enumerate(data_list), total=len(data_list)):
            if self.max_records is not None and i >= self.max_records:
                break
            # 静态生成
            i_data = self.gen_task_record(i, rec)
            if i_data is None or len(i_data) == 0:
                continue
            if isinstance(i_data, list):
                ret_data += i_data
            else:
                ret_data.append(i_data)
        if self.mode in ('dev', 'eval'):
            logging.info(f"self.dev_task_counter: {self.dev_task_counter}")
        logging.info(
            f"task records {data_file} loaded in mode {self.mode},  processed {i + 1}/{len(data_list)}, got {len(ret_data)} records")
        return ret_data

    def choice_task(self):
        randomList = random.choices(self.task_keys, weights=self.task_weights, k=1)
        return randomList[0]

    def dynamic_gen_record_prompt_and_labels_task0(self, text):
        task_type = 0
        i_doc_text_labels = text
        # donut style
        prompt = self.tasks[self.task_keys[task_type]][0]
        # 简单用字符串做一个cut
        i_can_add_len = self.args.decoder_max_length - len(prompt)
        labels = i_doc_text_labels[:i_can_add_len]
        return prompt, labels

    def dynamic_gen_record_prompt_and_labels_task1(self, task_id, text, ocr_data, image_path):
        image_size = get_image_size(image_path)
        i_prompt, i_label, i_status = self.gen_layout_texts_tasks(task_id, ocr_data, text, image_size,
                                                                  self.args.decoder_max_input_length,
                                                                  self.args.decoder_max_length)
        return i_prompt, i_label, i_status

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 有些要动态加载
        item = self.data[idx]
        # on the run
        image = pil_loader(item['image'])
        # encoding = self.processor(images=item["image"], return_tensors="pt", add_special_tokens=True,
        #                           max_patches=self.max_patches)
        encoding = self.processor(images=image, return_tensors="pt", add_special_tokens=True,
                                  max_patches=self.max_patches)
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        if self.mode == 'train':
            # 动态生成任务
            i_task = self.choice_task()
            if i_task == 0:
                i_prompt, i_labels = self.dynamic_gen_record_prompt_and_labels_task0(item['text'])
            elif i_task == 1:
                if 'ocr_data' in item:
                    i_ocr_data = item['ocr_data']
                elif item['id'] in self.ocr_data:
                    i_ocr_data = self.ocr_data[item['id']]
                else:
                    i_ocr_path, i_ocr_format, i_ocr_suffix = self.get_ocr_path({'file_name': item['id']})
                    i_ocr_data = self.load_single_ocr_data(i_ocr_path, ocr_format=i_ocr_format)
                    if i_ocr_data is None:
                        logging.warning(f"OCR data should not be None for file {i_ocr_path}, switch to task_type 0;")

                if i_ocr_data is None:
                    i_prompt, i_labels = self.dynamic_gen_record_prompt_and_labels_task0(item['text'])
                    i_task = 0
                else:
                    i_prompt, i_labels, i_status = self.dynamic_gen_record_prompt_and_labels_task1(item['id'],
                                                                                                   item['text'],
                                                                                        i_ocr_data, item['image'])
            encoding["prompt"] = i_prompt
            encoding["labels"] = i_labels
            encoding["task_type"] = i_task
        else:
            encoding["prompt"] = item["prompt"]
            if self.mode != 'test':
                encoding["labels"] = item["labels"]
            encoding["task_type"] = item["task_type"]

        encoding['id'] = item['id']
        encoding["ori"] = item["ori"]
        return encoding


if __name__ == '__main__':
    pass
