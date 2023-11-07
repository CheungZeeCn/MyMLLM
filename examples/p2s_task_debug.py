import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import re

from libs.datasets import image_utils
from libs.draw_utils import pil_draw_line, pil_draw_box
from libs.datasets import dataset_utils
import math
import logging


import kp_setup



#model_path = '/home/ana/data4/models/pix2struct-base-raw-enlarge-vocab-special-mean'
# p = Pix2StructProcessor.from_pretrained(model_path)
#image_path = '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/train/image_0.jpg'

ocr_path = '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/train/ocr'
task_pickle = '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/validation/metadata.jsonl.pickle'
# image_out_path = '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/train/debug_image_0.jpg'
patch_size = (16, 16) # width, height
max_patches = 2048




label_str = ''
task_str = ''


def get_p2s_resize(img, patch_size):
    patch_width, patch_height = patch_size
    image_width, image_height = img.size
    scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
    num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
    num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
    resized_height = max(num_feasible_rows * patch_height, 1)
    resized_width = max(num_feasible_cols * patch_width, 1)
    return resized_height, resized_width


def scale_and_draw_bbox(img, ori_bbox, ori_height, ori_width, color='red'):
    """
        rescale bbox and draw it
        len4 coord
    """
    image_width, image_height = img.size
    x_scale = image_width / ori_width
    y_scale = image_height / ori_height
    new_coord = (ori_bbox[0] * x_scale, ori_bbox[1] * y_scale, ori_bbox[2] * x_scale, ori_bbox[3] * y_scale)
    img = pil_draw_box(img, new_coord, color=color)
    return img


def draw_layout_task(task_string):
    """
        画一个任务串
    """
    pass

def draw_patch_grids(resized_img, resized_width, resized_height, patch_width, patch_height):
    # 画网格线
    col_lines = [[(x,0), (x, resized_height)] for x in range(0, resized_width, patch_width) if x!=0]
    row_lines = [[(0, y), (resized_width, y)] for y in range(0, resized_height, patch_height) if y!=0]
    for a,b in col_lines + row_lines:
        pil_draw_line(resized_img, a, b, color='gray')
    return resized_img

def load_task_pickle(task_pickle_path):
    with open(task_pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def re_pick_locs(task_str):
    re_loc = re.compile(r'((?:<loc_\d+>){4})')
    re_digit = re.compile(r'\d+')
    loc_strs = re_loc.findall(task_str)
    locs = []
    for loc_str in loc_strs:
        locs.append([int(x) for x in re_digit.findall(loc_str)])
    return locs


def trans_locs_to_coords(locs, img, max_loc):
    image_width, image_height = img.size
    coords = []
    for loc in locs:
        coord = [round(loc[0] / max_loc * image_width), round(loc[1] / max_loc * image_height),
                 round(loc[2] / max_loc * image_width), round(loc[3] / max_loc * image_height)]
        coord = dataset_utils.pick_2_points(dataset_utils.fix_coord_error(coord))
        coords.append(coord)
    return coords


def show_one_task(img, task, color=('green', 'blue'), max_loc=500):
    """
        prompt:  根据上下文预测文字和位置: 链接 == * <layout_0>嘉</layout_0>诺撒 圣心<text_0><loc_114><loc_92><loc_153><loc_107></text_0> 中学<text_layout_0>a<layout_1>t</layout_1>e<text_1><loc_151><loc_109><loc_161><loc_122></text_1> 202<text_layout_1>1 <layout_2>51</layout_2>60<text_2><loc_104><loc_141><loc_136><loc_158></text_2>桥 贤儿 小 <text_layout_2>贤儿 <layout_3>（小桥</layout_3> 贤児， <text_3><loc_204><loc_107><loc_223><loc_120></text_3>，<text_layout_3> 本男<layout_4>性</layout_4>前 <text_4><loc_198><loc_140><loc_216><loc_156></text_4>员<text_layout_4> <layout_5>D</layout_5>J、 电<text_5><loc_328><loc_56><loc_385><loc_70></text_5>、音 <text_layout_5>活<layout_6>动</layout_6>制<text_6><loc_385><loc_70><loc_403><loc_86></text_6>人 。<text_layout_6>身 于<layout_7>东京都</layout_7> 。<text_7><loc_332><loc_120><loc_355><loc_135></text_7>高170cm 。A型血。 == 简介 == 日 <text_layout_7>前<layout_8>演员，</layout_8> 引退后 <text_8><loc_383><loc_219><loc_426><loc_237></text_8>时尚
        labels:  <layout_0><loc_78><loc_76><loc_98><loc_91><text_0>英文<text_layout_0>d<loc_111><loc_109><loc_122><loc_122><layout_1><loc_131><loc_109><loc_141><loc_122><text_1>=<text_layout_1>01019<loc_97><loc_124><loc_151><loc_140><layout_2><loc_61><loc_141><loc_82><loc_158><text_2>4小<text_layout_2>桥<loc_207><loc_59><loc_226><loc_74><layout_3><loc_212><loc_74><loc_266><loc_88><text_3>）<text_layout_3>日<loc_242><loc_107><loc_262><loc_120><layout_4><loc_241><loc_123><loc_260><loc_136><text_4>演<text_layout_4>、<loc_233><loc_140><loc_250><loc_156><layout_5><loc_196><loc_159><loc_205><loc_175><text_5>影导演<text_layout_5>乐<loc_311><loc_70><loc_329><loc_86><layout_6><loc_347><loc_70><loc_366><loc_86><text_6>作<text_layout_6>出<loc_330><loc_88><loc_350><loc_102><layout_7><loc_330><loc_103><loc_390><loc_118><text_7>身<text_layout_7>本<loc_39><loc_225><loc_57><loc_241><layout_8><loc_74><loc_225><loc_128><loc_241><text_8>经历

        仅画框框就可以了吧? 问内容的时候，画出框框； 自己看label 字符串;
        问框框的时候，直接看任务和框框，后面再看pred 就知道是否对齐了
        问框框和内容的额时候也类似
    """
    image_width, image_height = img.size

    prompt_str = task['prompt']
    label_str = task['labels']

    # collect draw box tasks:
    # prompt:  <text_\d+>  or text_layout_\d+ 任务
    # labels:  <layout_\d+>  or text_layout_\d+ 任务
    # 做个简单版本
    locs0 = re_pick_locs(prompt_str)
    locs1 = re_pick_locs(label_str)
    coords = trans_locs_to_coords(locs0, img, max_loc)
    for coord in coords:
        pil_draw_box(img, coord, color=color[0])
    coords = trans_locs_to_coords(locs1, img, max_loc)
    for coord in coords:
        pil_draw_box(img, coord, color=color[1])
    return img


def show_one_pred(img, pred_str, color='orange', max_loc=500):
    locs = re_pick_locs(pred_str)
    coords = trans_locs_to_coords(locs, img, max_loc)
    for coord in coords:
        pil_draw_box(img, coord, color=color)
    return img

def draw_a_task(task, show_grids=True, show_bbox=True, ocr_path=None, ocr_suffix='.ocr.json', show_task=True, pred_str=''):
    # 加载
    image_path = task['image']
    img = image_utils.pil_loader(image_path)
    logging.info(f"ori_size(w, h): {img.size}")

    # 大小
    patch_width, patch_height = patch_size
    image_width, image_height = img.size
    resized_height, resized_width = get_p2s_resize(img, patch_size)

    logging.info(f"resized(h, w): {resized_height}, {resized_width}")
    # 获得说缩放后的图片
    resized_img = img.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)

    # 显示 grid， 就是每个patch在哪里
    if show_grids is True:
        draw_patch_grids(resized_img, resized_width, resized_height, patch_width, patch_height)

    # 显示ocr file 的bbox
    if show_bbox is True:
        if ocr_path is None:
            logging.error("OCR path should not be None, if you want to draw bboxes!")
        else:
            ocr_file_path = os.path.join(ocr_path, task['id']+ocr_suffix)
            if not os.path.exists(ocr_file_path):
                logging.error(f"{ocr_file_path} does not exist!")
            else:
                ocr_data = dataset_utils.load_single_pp_ocr_json(ocr_file_path, with_len4_coord=True)
                for coord in ocr_data:
                    scale_and_draw_bbox(resized_img, coord, image_height, image_width)

    # 画label
    if show_task is True:
        resized_img = show_one_task(resized_img, task)

    if pred_str != '':
        resized_img = show_one_pred(resized_img, pred_str)

    return resized_img




if __name__ == '__main__':
    # 加载pickle, 然后画
    tasks = load_task_pickle(task_pickle)
    task_dict = {task['id']:task for task in tasks if task['task_type'] == 1}
    """
    image_1.jpg
    'id' = {str} 'image_1.jpg'
    'image' = {str} '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/train/image_1.jpg'
    'prompt' = {str} '根据上下文预测文字和位置: 链接 == * <layout_0>嘉</layout_0>诺撒 圣心<text_0><loc_114><loc_92><loc_153><loc_107></text_0> 中学<text_layout_0>a<layout_1>t</layout_1>e<text_1><loc_151><loc_109><loc_161><loc_122></text_1> 202<text_layout_1>1 <layout_2>51</layout_2>60<text_2><loc_
    'labels' = {str} '<layout_0><loc_78><loc_76><loc_98><loc_91><text_0>英文<text_layout_0>d<loc_111><loc_109><loc_122><loc_122><layout_1><loc_131><loc_109><loc_141><loc_122><text_1>=<text_layout_1>01019<loc_97><loc_124><loc_151><loc_140><layout_2><loc_61><loc_141><loc_82><loc_1
    'task_type' = {int} 1
    
    prompt:  根据上下文预测文字和位置: 链接 == * <layout_0>嘉</layout_0>诺撒 圣心<text_0><loc_114><loc_92><loc_153><loc_107></text_0> 中学<text_layout_0>a<layout_1>t</layout_1>e<text_1><loc_151><loc_109><loc_161><loc_122></text_1> 202<text_layout_1>1 <layout_2>51</layout_2>60<text_2><loc_104><loc_141><loc_136><loc_158></text_2>桥 贤儿 小 <text_layout_2>贤儿 <layout_3>（小桥</layout_3> 贤児， <text_3><loc_204><loc_107><loc_223><loc_120></text_3>，<text_layout_3> 本男<layout_4>性</layout_4>前 <text_4><loc_198><loc_140><loc_216><loc_156></text_4>员<text_layout_4> <layout_5>D</layout_5>J、 电<text_5><loc_328><loc_56><loc_385><loc_70></text_5>、音 <text_layout_5>活<layout_6>动</layout_6>制<text_6><loc_385><loc_70><loc_403><loc_86></text_6>人 。<text_layout_6>身 于<layout_7>东京都</layout_7> 。<text_7><loc_332><loc_120><loc_355><loc_135></text_7>高170cm 。A型血。 == 简介 == 日 <text_layout_7>前<layout_8>演员，</layout_8> 引退后 <text_8><loc_383><loc_219><loc_426><loc_237></text_8>时尚 
    labels:  <layout_0><loc_78><loc_76><loc_98><loc_91><text_0>英文<text_layout_0>d<loc_111><loc_109><loc_122><loc_122><layout_1><loc_131><loc_109><loc_141><loc_122><text_1>=<text_layout_1>01019<loc_97><loc_124><loc_151><loc_140><layout_2><loc_61><loc_141><loc_82><loc_158><text_2>4小<text_layout_2>桥<loc_207><loc_59><loc_226><loc_74><layout_3><loc_212><loc_74><loc_266><loc_88><text_3>）<text_layout_3>日<loc_242><loc_107><loc_262><loc_120><layout_4><loc_241><loc_123><loc_260><loc_136><text_4>演<text_layout_4>、<loc_233><loc_140><loc_250><loc_156><layout_5><loc_196><loc_159><loc_205><loc_175><text_5>影导演<text_layout_5>乐<loc_311><loc_70><loc_329><loc_86><layout_6><loc_347><loc_70><loc_366><loc_86><text_6>作<text_layout_6>出<loc_330><loc_88><loc_350><loc_102><layout_7><loc_330><loc_103><loc_390><loc_118><text_7>身<text_layout_7>本<loc_39><loc_225><loc_57><loc_241><layout_8><loc_74><loc_225><loc_128><loc_241><text_8>经历
    """
    # draw_a_task(task, show_grids=True, show_bbox=True, ocr_path=None, show_task=True, pred_str=''):
    pred_str='<layout_0><loc_151><loc_73><loc_178><loc_73><text_0> 何<text_layout_0> 被撤<loc_157><loc_129><loc_194><loc_129><layout_1><loc_160><loc_151><loc_194><loc_151><text_1> 委<text_layout_1> 员，此<loc_141><loc_194><loc_1    94><loc_194><layout_2><loc_258><loc_73><loc_275><loc_73><text_2> 有<text_layout_2> 官<loc_258><loc_151><loc_282><loc_151><layout_3><loc_275><loc_173><loc_292><loc_173><text_3> 确<text_layout_3> 消<loc_311><loc_103>    <loc_340><loc_101><layout_4><loc_311><loc_206><loc_344><loc_194><text_4> 中国<text_layout_4> 民政治<loc_360><loc_161><loc_407><loc_162><layout_5><loc_368><loc_194><loc_404><loc_194><text_5> 议    #####   <layout_0>    <loc_150><loc_65><loc_190><loc_65><text_0> 何<text_layout_0> 被撤<loc_149><loc_125><loc_190><loc_125><layout_1><loc_149><loc_155><loc_189><loc_155><text_1> 委<text_layout_1> 员，此<loc_131><loc_214><loc_189><loc_21    4><layout_2><loc_239><loc_65><loc_258><loc_65><text_2> 有<text_layout_2> 官<loc_260><loc_145><loc_277><loc_145><layout_3><loc_259><loc_187><loc_279><loc_187><text_3> 确<text_layout_3> 消<loc_322><loc_99><loc_341><l    oc_99><layout_4><loc_316><loc_220><loc_342><loc_220><text_4> 中国<text_layout_4>民政 治<loc_372><loc_160><loc_430><loc_160><layout_5><loc_370><loc_188><loc_411><loc_188><text_5> 议'
    resized_img = draw_a_task(task_dict['image_3.jpg'], show_grids=True, show_bbox=False, ocr_path=ocr_path, show_task=True, pred_str=pred_str)
    plt.imshow(resized_img)
    plt.show()
    # resized_img.save('/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/re_image_0.jpg')









