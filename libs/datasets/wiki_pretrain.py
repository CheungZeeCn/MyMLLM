"""
    wiki预训练任务组装:
        1. 支持 prompt
            train:
                label => label
                prompt => prompt
            infer:
        2. 支持多种任务(可以多合一)
            任务指示:
                给上下文问位置:
                    输入:
                        识别位置. 今天天气不错，挺<layout_0>风和日丽</layout_0>的, 我们下午没有课, 这<layout_1>的确是挺好</layout_1>的
                    输出:
                        #(x1,y1,x2,y2) 左上，右下
                        <layout_0><loc_0><loc30><loc_20><loc_31><layout_1><loc_0><loc40><loc_20><loc_50>

                给位置问内容:
                    输入:
                        识别内容. 今天天气不错，挺<text_0><loc30><loc_20><loc_31></text_0>的, 我们下午没有课, 这<text_1><loc_0><loc40><loc_20><loc_50></text_1>的
                    输出:
                        # 文字内容
                        <text_0>风和日丽<text_1>的确是挺好

                给上下文，问内容和位置
                    输入:
                        识别位置和内. 容今天天气不错，挺<text_layout_0>的, 我们下午没有课, 这<text_layout_1>的
                    输出:
                        #(x1,y1,x2,y2) 左上，右下
                        <text_layout_0>风和日丽<loc_0><loc30><loc_20><loc_31><text_layout_1>的确是挺好<loc_0><loc40><loc_20><loc_50>
        3. 文字的mask与生成
            a. 无mask 的生成 '按顺序识别图片中的文字: ' 类似OCR
            b. 有mask的生成: todo, 有LM的训练
            c. 随机15%的patch? todo 有LM的训练，且这部分可以恢复patch，加一个decoder
"""
import os
import json

from torch.utils.data.dataset import Dataset
from libs.datasets.dataset_utils import load_single_pp_ocr_json, load_pp_ocr_json, load_json_lines_into_list, \
    order_by_tbyx_coord, \
    box_norm, get_word_layouts, filter_bboxes_by_ratio, collect_mask_spans
from libs.datasets.image_utils import get_image_wh, get_image_size

import pickle
import logging
from tqdm import tqdm
from PIL import Image
import random
from libs import utils


class wiki_pretrain_dataset(Dataset):
    def __init__(
            self,
            args,
            file_dir,
            processor,
            mode,
            max_records=None,
            max_patches=2048,
            load_cache=True,
    ):
        self.args = args
        self.mode = mode
        self.processor = processor
        self.file_dir = file_dir
        self.max_records = max_records
        self.max_patches = max_patches
        self.load_cache = load_cache

        # file_name = os.path.join(args.data_dir, "{}.{}.json".format(self.cur_la, 'train' if mode == 'train' else 'val'))
        self.tasks = [['按顺序识别图片中的文字: ', 0.5],
                      ['根据上下文预测文字和位置: ', 0.5],
                      ]
        self.task_keys = list(range(len(self.tasks)))

        self.task_weights = [task[1] for task in self.tasks]

        file_name = os.path.join(file_dir, "metadata.jsonl")
        logging.info(f"加载数据集: {file_name}")
        cache_file = file_name + '.pickle'
        if self.load_cache is True:
            if not os.path.exists(cache_file):
                logging.warning(f"cache file {cache_file} does not exist!")
                self.data = self.load_task(file_name)
                self.dump_cache_file(self.data, cache_file)
            else:
                logging.warning(f"load cache file {cache_file}")
                self.data = self.load_cache(cache_file)
        else:
            self.data = self.load_task(file_name)
            self.dump_cache_file(self.data, cache_file)

    def load_cache_file(self, cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def dump_cache_file(self, data, cache_file):
        with open(cache_file, 'wb') as f:
            ret = pickle.dump(data, f)
        return ret

    def choice_task(self):
        randomList = random.choices(self.task_keys, weights=self.task_weights, k=1)
        return randomList[0]

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
            d_all_the_same_y_flag = True
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
                    task_type = choose_a_type()
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
                        print(f"task_type: {task_type} > 100, ignore this one")
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
                        # debug
                        if span_loc[1] != span_loc[3]:
                            d_all_the_same_y_flag = False
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
                        # debug
                        if span_loc[1] != span_loc[3]:
                            d_all_the_same_y_flag = False
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
            # if d_all_the_same_y_flag is True:
            #     logging.warning(f"{file_id} tasks are all in the same y")
            # else:
            #     logging.warning(f"{file_id} tasks are NOT all in the same y")

            return candidate_bboxes

        # 过滤bbox
        candidate_bboxes, filtered_bboxes = filter_bboxes_by_ratio(bboxes, text, ratio=ratio)
        # 不符合当前需求， 只能退化成一个最简答的问题了:
        if len(candidate_bboxes) == 0:
            logging.warning(f"{file_id}无符合要求的bbox，退化为整个大框下按顺序识别文本")
            prompt = "根据上下文预测文字和位置: " + "<text_0><loc_0><loc_0><loc_500><loc_500></text_0>"
            label = "<text_0>" + text
            return prompt, label

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
        return prompt, label

    def gen_task_record(self, i_task, rec, rec_ocr_data=None):
        i_file_id = rec["file_name"]
        i_image_path = os.path.join(self.file_dir, i_file_id)
        assert os.path.exists(i_image_path)
        if i_task == 0:  # 按顺序识别
            i_doc_text_labels = json.loads(rec["ground_truth"])["gt_parse"]["text_sequence"]
            i_data = {}
            i_data['id'] = i_file_id
            # i_data['image'] = pil_loader(i_image_path)
            i_data['image'] = i_image_path
            i_data['prompt'] = self.tasks[i_task][0]
            # 简单用字符串做一个cut
            i_can_add_len = self.args.decoder_max_length - len(i_data['prompt'])
            i_data['labels'] = i_doc_text_labels[:i_can_add_len]
            i_data['task_type'] = i_task
            return i_data
        elif i_task == 1:
            i_ori_labels = json.loads(rec["ground_truth"])["gt_parse"]["text_sequence"]
            i_data = {}
            i_data['id'] = i_file_id
            # i_data['image'] = pil_loader(i_image_path)
            i_data['image'] = i_image_path
            image_size = get_image_size(i_image_path)
            # 过滤出bbox中文本完全在text中的bbox，排序，然后按照比例来选取目标
            prompt, label = self.gen_layout_texts_tasks(i_file_id, rec_ocr_data, i_ori_labels, image_size,
                                                        self.args.decoder_max_input_length,
                                                        self.args.decoder_max_length)
            i_data['prompt'] = prompt
            i_data['labels'] = label
            i_data['task_type'] = i_task
            return i_data

        raise NotImplementedError()

    def load_single_ocr_data(self, file_id, suffix=".ocr.json"):
        ocr_dir = os.path.join(self.file_dir, 'ocr')
        file_path = os.path.join(ocr_dir, f"{file_id}{suffix}")
        return load_single_pp_ocr_json(file_path, with_len4_coord=True)

    def load_task(
            self,
            data_file,
    ):
        """
            ocr data
        """
        ocr_dir = os.path.join(self.file_dir, 'ocr')

        #  方便debug 改为单个文件加载
        # with utils.utils_timer("load pp ocr json"):
        #     ocr_data = load_pp_ocr_json(ocr_dir, suffix='.ocr.json', with_len4_coord=True)
        # logging.info("ocr data loaded")
        ocr_data = {}

        """
            json 文件的数据预处理
        """
        data_list = load_json_lines_into_list(data_file)
        # re-org data format
        ret_data = []
        # bbox 级别的数据分组处理
        for i, rec in tqdm(enumerate(data_list), total=len(data_list)):
            if self.max_records is not None and i >= self.max_records:
                break
            i_file_id = rec["file_name"]
            i_image_path = os.path.join(self.file_dir, i_file_id)
            assert os.path.exists(i_image_path)
            if i_file_id not in ocr_data:
                ocr_data[i_file_id] = self.load_single_ocr_data(i_file_id)
            # assert i_file_id in ocr_data, f"{i_file_id} not in ocr_data"

            if not i_file_id in ocr_data:
                logging.warning(f"{i_file_id} not in ocr_data, ignore")
                continue

            # assert i_file_id in ocr_data, f"{i_file_id} not in ocr_data"
            i_task = self.choice_task()
            i_data = self.gen_task_record(i_task, rec, ocr_data[i_file_id])
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
        encoding['prompt'] = item["prompt"]
        encoding["labels"] = item["labels"]
        encoding["task_type"] = item["task_type"]
        return encoding


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == '__main__':
    pass
