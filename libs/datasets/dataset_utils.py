# dataset_utils.py
"""
一些相对常用的数据集函数
        by zhangz@20230330
"""

import os
import json
import itertools
import string
from copy import deepcopy
from collections import defaultdict
from PIL import Image
import jsonlines
import re
import random
import logging
import torch
from typing import Any


def box_norm(box, width, height, norm_to=1000):
    def clip(min_num, num, max_num):
        return min(max(num, min_num), max_num)

    x0, y0, x1, y1 = box
    x0 = clip(0, int((x0 / width) * norm_to), norm_to)
    y0 = clip(0, int((y0 / height) * norm_to), norm_to)
    x1 = clip(0, int((x1 / width) * norm_to), norm_to)
    y1 = clip(0, int((y1 / height) * norm_to), norm_to)
    # 有些图片旋转的比较厉害 就没有办法咯
    if x1 < x0:
        x1 = x0 + 1
    if y1 < y0:
        y1 = y0 + 1
    assert x1 >= x0 and x0 >= 0, f"{x1}, {x0}"
    assert y1 >= y0 and y0 >= 0, f"{y1}, {y0}"
    return [x0, y0, x1, y1]


def get_image_wh_by_ocr(data):
    x_values = []
    y_values = []
    for kv_pair in data["KV-Pairs"]:
        for value in kv_pair['Values']:
            x_values += [coord_v for i, coord_v in enumerate(value['Coord']) if i % 2 == 0]
            y_values += [coord_v for i, coord_v in enumerate(value['Coord']) if i % 2 == 1]
    for value in data["Backgrounds"]:
        x_values += [coord_v for i, coord_v in enumerate(value['Coords']) if i % 2 == 0]
        y_values += [coord_v for i, coord_v in enumerate(value['Coords']) if i % 2 == 1]
    for key, values in data["Keys"].items():
        for value in values:
            x_values += [coord_v for i, coord_v in enumerate(value['Coords']) if i % 2 == 0]
            y_values += [coord_v for i, coord_v in enumerate(value['Coords']) if i % 2 == 1]
    min_x = max(0, min(x_values))
    min_y = max(0, min(y_values))
    max_x = max(x_values)
    max_y = max(y_values)
    return min_x, min_y, max_x, max_y


def get_image_wh_by_image(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
    return 0, 0, img.size[0], img.size[1]


def fix_coord_error(coord):
    """
        简单的位置修正, 固定到合理的矩形坐标上, 副作用是会改动coord原来的数值
    """
    coord = list(coord)
    x_values = []
    y_values = []
    x_values += [coord_v for i, coord_v in enumerate(coord) if i % 2 == 0]
    y_values += [coord_v for i, coord_v in enumerate(coord) if i % 2 == 1]
    min_x = max(0, min(x_values))
    min_y = max(0, min(y_values))
    max_x = max(max(x_values), min_x + 1)
    max_y = max(max(y_values), min_y + 1)
    coord = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
    coord = tuple(coord)
    return coord


def load_pp_ocr_data(dir_path, suffix=".txt"):
    ret_data = defaultdict(dict)
    for file in os.listdir(dir_path):
        if file.endswith(suffix):
            file_id = file.replace(suffix, "")
            with open(os.path.join(dir_path, file)) as f:
                data = f.read().split("\n")
                for line in data:
                    line = line.strip()
                    if line == "":
                        continue
                    line_obj = json.loads(line)
                    coords = tuple(itertools.chain(*line_obj[0]))
                    coords = fix_coord_error(coords)
                    text = line_obj[1][0]
                    ret_data[file_id][coords] = text
    return ret_data


def load_pp_ocr_json(dir_path, suffix=".json", with_len4_coord=False):
    ret_data = defaultdict(dict)
    for file in os.listdir(dir_path):
        if file.endswith(suffix):
            file_id = file.replace(suffix, "")
            file_path = os.path.join(dir_path, file)
            ret_data[file_id] = load_single_pp_ocr_json(file_path, with_len4_coord=with_len4_coord)
    return ret_data


def load_single_pp_ocr_json(file_path, with_len4_coord=False):
    ret_data = {}
    with open(os.path.join(file_path)) as f:
        data = json.load(f)
        for line_obj in data[0]:
            coords = tuple(itertools.chain(*line_obj[0]))
            coords = fix_coord_error(coords)
            text = line_obj[1][0]
            if with_len4_coord is True:
                coords = tuple(pick_2_points(coords))
            ret_data[coords] = text
    return ret_data


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_punc(uchar):
    if uchar in string.punctuation:
        return True
    else:
        return False


def char_type(ch):
    if is_chinese(ch):
        return 'zh'
    elif is_number(ch):
        return 'number'
    elif is_alphabet(ch):
        return 'en'
    elif is_punc(ch):
        return 'en_punc'
    else:
        return 'default'


def get_word_layouts(bbox, text):
    """
        用最简单的方式来估算文字所在位置
        仅接收4 元组 bbox
    """
    c_type_len = {
        "zh": 2,
        "number": 1,
        "en": 1,
        "en_punc": 1,
        "default": 2
    }

    total_len = 0
    for c in text:
        c_type = char_type(c)
        total_len += c_type_len[c_type]

    uni_len = (bbox[2] - bbox[0]) / total_len

    result = []
    now_len = 0
    for c in text:
        c_type = char_type(c)
        c_len = c_type_len[c_type]

        x0 = int(bbox[0] + (now_len * uni_len))
        x1 = int(bbox[0] + ((now_len + c_len) * uni_len))

        result.append({"word": c,
                       "coord": [x0, bbox[1], x1, bbox[3]]
                       })
        now_len += c_len
    return result



def to_4_points(flat_coord):
    return (flat_coord[0:2], flat_coord[2:4], flat_coord[4:6], flat_coord[6:8])


def pick_2_points(flat_coord):
    return flat_coord[0:2] + flat_coord[4:6]


def order_by_tbyx_coord(coords, th=20):
    """
	ocr_info: a list of dict, which contains bbox information([x1, y1, x2, y2])
	th: threshold of the position threshold
	"""
    res = sorted(coords, key=lambda r: (r[1], r[0]))  # sort using y1 first and then x1
    for i in range(len(res) - 1):
        for j in range(i, 0, -1):
            # restore the order using the
            if abs(res[j + 1][1] - res[j][1]) < th and \
                    (res[j + 1][0] < res[j][0]):
                tmp = deepcopy(res[j])
                res[j] = deepcopy(res[j + 1])
                res[j + 1] = deepcopy(tmp)
            else:
                break
    return res


def c_offset_to_t_offset(offsets_mapping, x):
    """
        char offset -> token offset
        todo: 二分可提速
    :param offsets_mapping:
    :param x:
    :return:
    """
    for i, i_c_offset in enumerate(offsets_mapping):
        if x >= i_c_offset[0] and x < i_c_offset[1]:
            return i
    raise ValueError(f"x should < len(offset_mappings), but {x} >= {len(offsets_mapping)}")


# 20230815
def load_json_lines_into_list(file, encoding='utf-8'):
    ret = []
    with open(file, "r", encoding=encoding) as f:
        for item in jsonlines.Reader(f):
            ret.append(item)
    return ret

def load_json(file, encoding='utf-8'):
    with open(file, 'r', encoding=encoding) as f:
        ret = json.load(f)
    return ret

# 20230914
def gen_mask_in_bboxes_by_ratio(bboxes, text, ratio=0.30):
    """
        1. 过滤出仅有一次出现的bbox， 作为候选
    """
    pass


def filter_bboxes_by_ratio(bboxes, text, ratio=0.30):
    """
        1. 过滤出仅有一次出现的bbox， 作为候选bbox， 并对bbox出现的位置进行记录和排序
        2. 对排序好的bbox内容动态mask，并映射回bbox中，接着开始按顺序生成和拼接prompt 和label
    """
    prompt, label = "", ""
    candidate_bboxes = {}
    filtered_bboxes = {}
    for coord, bbox_text in bboxes.items():
        if text.count(bbox_text) != 1:
            filtered_bboxes[coord] = bbox_text
            continue
        candidate_bboxes[coord] = [bbox_text, text.index(bbox_text), text.index(bbox_text) + len(bbox_text)]
    return candidate_bboxes, filtered_bboxes


def get_bbox_center_coord(four_coords):
    return ((four_coords[0] + four_coords[2]) / 2, (four_coords[1] + four_coords[3]) / 2)


def collect_mask_spans(mask_flags):
    spans = []
    i = 0
    while i < len(mask_flags):
        i_label = mask_flags[i]
        # begin
        if i_label != 0:
            j = i + 1
            for j in range(i + 1, len(mask_flags)):
                j_label = mask_flags[j]
                if i_label == j_label:
                    j += 1
                    continue
                else:
                    break
            spans.append([i, j])
            i = j
            continue
        else:
            # next
            i += 1
    return spans


def custom_token_split(input, pattern=r"</?.+?>"):
    last_index = 0
    tokenized = []
    for m in re.finditer(pattern, input):
        m_i, m_j = m.span()
        if m_i != last_index:
            tokenized += list(input[last_index:m_i])
        tokenized.append(input[m_i:m_j])
        last_index = m_j
    if last_index == 0:
        tokenized += list(input)
    elif m_j != len(input):
        tokenized += list(input[m_j:])
    return tokenized


def load_pa_ocr_data_in_dict(dir_path):
    ret_data = defaultdict(dict)
    for file in os.listdir(dir_path):
        if file.endswith(".txt"):
            file_id = file.replace(".txt", "")
            fn = os.path.join(dir_path, file)
            ret_data[file_id] = load_pa_ocr_in_dict(fn)
    return ret_data


def load_pa_ocr_in_dict(fn, text_only=False, with_len4_coord=False):
    bboxes = {}
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            sp_line = line.split("\t")
            if len(sp_line) < 10:
                continue
            bbox = {}
            for i in range(8):
                sp_line[i] = int(float(sp_line[i]))
            coords = fix_coord_error(sp_line[:8])
            if with_len4_coord is True:
                coords = pick_2_points(coords)
            bbox['coord'] = coords
            bbox['score'] = round(float(sp_line[8].strip()), 5)
            bbox['text'] = sp_line[9].strip()
            bbox['length'] = len(bbox['text'])
            if len(bbox['text']) == 0:
                logging.warning("0 == bbox['length'], ignore: '{}'".format(line))
                continue
            if text_only is True:
                bboxes[bbox['coord']] = bbox['text']
            else:
                bboxes[bbox['coord']] = bbox
    return bboxes


def load_pa_ocr_with_char_x_axis(fn):
    bboxes = []
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            sp_line = line.split("\t")
            if len(sp_line) <= 10:
                continue
            bbox = {}
            coords = fix_coord_error(sp_line[:8])
            bbox['coord'] = coords
            bbox['score'] = round(float(sp_line[8].strip()), 5)
            bbox['text'] = sp_line[9].strip()
            bbox['length'] = len(bbox['text'])
            bbox['char_x_axis'] = [round(float(x_coord), 5) for x_coord in sp_line[10:]]
            if len(bbox['char_x_axis']) != bbox['length']:
                logging.warning("len(bbox['char_x_axis']) != bbox['length'], ignore: '{}'".format(line))
                continue
            bboxes.append(bbox)
    return bboxes


def load_pa_ocr_data(dir_path):
    """
        加载成字典的格式返回
    :param dir_path:
    :return:
    """
    ret_data = defaultdict(dict)
    for file in os.listdir(dir_path):
        if file.endswith(".txt"):
            file_id = file.replace(".txt", "")
            with open(os.path.join(dir_path, file)) as f:
                data = f.read().split("\n")
                for line in data[1:]:
                    line = line.strip()
                    if line == "":
                        continue
                    ls = line.split("\t")
                    if len(ls) <= 10:
                        continue
                    coords = ls[0:2] + ls[7:8] + ls[4:6] + ls[2:4]
                    coords = fix_coord_error(coords)
                    text = ls[9].strip()
                    if len(text) == 0:
                        continue
                    ret_data[file_id][coords] = text
    return ret_data


def collect_dict_keys(rec, vocab):
    """
        过滤数据, 将字典的key都找出来，作为特殊的token
    """
    to_add_keys = set()
    if isinstance(rec, list):
        for i_rec in rec:
            to_add_keys |= collect_dict_keys(i_rec, vocab)
    elif isinstance(rec, dict):
        for k, v in rec.items():
            if f"<{k}>" not in vocab:
                to_add_keys.add(f"<s_{k}>")
            if f"</{k}>" not in vocab:
                to_add_keys.add(f"</s_{k}>")
            to_add_keys |= collect_dict_keys(v, vocab)
    else:
        pass
    return to_add_keys

def add_special_keys(model, tokenizer, keys):
    to_add_special_tokens = []
    for k in sorted(keys):
        to_add_special_tokens.append(k)
    added_num = tokenizer.add_special_tokens({"additional_special_tokens": to_add_special_tokens})

    if added_num != 0:
        try:
            origin_emb_len = len(model.decoder.model.decoder.embed_tokens.weight)
            added_len = len(tokenizer.vocab) - origin_emb_len
            model.decoder.resize_token_embeddings(len(tokenizer.vocab))
            logging.info(
                f"origin_emb_len is {origin_emb_len}, added_len is {added_len}, now model emb shape {model.decoder.model.decoder.embed_tokens.weight.shape}")
        except Exception as e:
            origin_emb_len = len(model.decoder.embed_tokens.weight)
            added_len = len(tokenizer.vocab) - origin_emb_len
            model.resize_token_embeddings(len(tokenizer.vocab))
            logging.info(
                f"origin_emb_len is {origin_emb_len}, added_len is {added_len}, now model emb shape {model.decoder.embed_tokens.weight.shape}")
    else:
        pass
    return model, tokenizer, added_num

"""
对比donut的原生实现 这是个简化版 有些复杂结构不一定合适
"""
def json2token(obj: Any, sort_key: bool = True, sep_token="<_sep_>"):
    """
        编码解码
    """
    if isinstance(obj, list):
        return sep_token + sep_token.join([ json2token(v, sort_key) for v in obj]) + sep_token
    elif isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda x: x[0]) if sort_key else obj.items()
        return "".join([fr"<s_{k}>" + json2token(v) + fr"</s_{k}>" for k, v in items])
    return str(obj)

def token2json(tokens, sep_token="<_sep_>"):
    # dict?
    if tokens.startswith(sep_token):
        return [token2json(sub_tokens) for sub_tokens in tokens.split(sep_token) if sub_tokens != '']
    elif re.match(r"<s_(.+?)>", tokens):
        ret_dict = {}
        while True:
            dict_m = re.match(r"<s_(.+)>(.*?)</s_\1>", tokens)
            if dict_m is None:
                break
            k, v = dict_m.groups()
            offset = dict_m.span()[1]
            ret_dict[k] = token2json(v)
            tokens = tokens[offset:]
        return ret_dict
    else:
        return tokens


def custom_json2token(obj):
    """
        很浅的递归就可以搞定
    """
    if isinstance(obj, list):
        return "".join([ "<li>"+custom_json2token(v)+"</li>" for v in obj])
    elif isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda x: x[0])
        return "".join([fr"<{k}>" + custom_json2token(v) + fr"</{k}>" for k, v in items])
    return str(obj)


def custom_token2json(tokens):
    if tokens.startswith("<li>"):
        return [custom_token2json(match) for match in re.findall(r"<li>(.*?)</li>", tokens)]
    elif re.match(r"<(.+?)>", tokens):
        ret_dict = {}
        while True:
            dict_m = re.match(r"<(.+)>(.*?)</\1>", tokens)
            if dict_m is None:
                break
            k, v = dict_m.groups()
            offset = dict_m.span()[1]
            ret_dict[k] = custom_token2json(v)
            tokens = tokens[offset:]
        return ret_dict
    else:
        return tokens

