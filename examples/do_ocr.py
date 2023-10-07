import os
import shutil
from paddleocr import PaddleOCR, draw_ocr
import json
import datetime
from glob import iglob


def test():
    from libs.datasets.dataset_utils import load_single_pp_ocr_json
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False,
                    lang='ch')  # need to run only once to download and load model into memory
    img_path = '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki/image_1.jpg'
    target_path = img_path + '.ocr.json'
    ocr_result = ocr.ocr(img_path, cls=True)
    json.dump(ocr_result, open(target_path, 'w'), ensure_ascii=False, indent=2)
    print(load_single_pp_ocr_json(target_path))


if __name__ == '__main__':
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, show_log=False,
                    lang='ch')  # need to run only once to download and load model into memory
    task_list = [
        ('/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki_500k_long/train',
         '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki_500k_long/train/ocr'),
        ('/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki_500k_long/validation',
         '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki_500k_long/validation/ocr'),
        ('/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki_500k_long/test',
         '/home/ana/data4/datasets/wiki_zh/synthdog_out/cn_wiki_500k_long/test/ocr')
    ]

    print(datetime.datetime.now())
    for image_dir, target_dir in task_list:
        print(f"task: {image_dir} -> {target_dir}")
        counter = 0
        for fn in iglob(image_dir+"/*.jpg"):
            counter += 1
            if counter % 1000 == 0:
                print(datetime.datetime.now())
                print(f"task: {image_dir} -> {target_dir} ######### {counter}")

            if not fn.endswith('.jpg'):
                continue

            full_path = fn
            target_path = full_path.replace(image_dir, target_dir) + '.ocr.json'
            # 检查是否存在
            if os.path.exists(target_path):
                continue
            ocr_result = ocr.ocr(full_path, cls=True)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            json.dump(ocr_result, open(target_path, 'w'), ensure_ascii=False, indent=2)
        print(counter)
