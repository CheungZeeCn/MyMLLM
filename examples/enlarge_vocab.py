"""
生成完毕后 要改动一下 tokenizer中关于 add_prefix_space 的配置
tokenizer_config.json:  "clean_up_tokenization_spaces": true,
tokenizer.json:    "add_prefix_space": false
tokenizer.json:    "add_prefix_space": false
"""

from transformers import AutoProcessor, Pix2StructForConditionalGeneration, AutoTokenizer, AutoModel, T5Tokenizer, T5TokenizerFast
import torch
from torch import nn




# model = '/home/ana/data4/output_models/MyMLLM/100k_p2s_prerain_wo_ocr/checkpoint-3200'
model = '/home/ana/data4/models/pix2struct-base'
base_model = '/home/ana/data4/models/pix2struct-base'
target = '/home/ana/data4/models/pix2struct-base-raw-enlarge-vocab-special-mean'


processor = AutoProcessor.from_pretrained(base_model)
processor.tokenizer.decoder.add_prefix_space = False
model = Pix2StructForConditionalGeneration.from_pretrained(model)

model.config.max_length = 2048
model.config.num_beams = 1
model.config.text_config.max_length = 2048
model.config.vision_config.max_length = 2048

cn_toknizer = AutoTokenizer.from_pretrained('/home/ana/data4/models/ChatYuan-large-v2')
cn_model = AutoModel.from_pretrained('/home/ana/data4/models/ChatYuan-large-v2')
to_add = sorted(list(set(cn_toknizer.vocab.keys()) | set(["\n"]) - set(processor.tokenizer.vocab.keys())))

print(f"origin vocab len: {len(processor.tokenizer.vocab)}")
print(f"add tokens len: {len(to_add)}")
processor.tokenizer.add_tokens(to_add)
print(f"added vocab len: {len(processor.tokenizer.vocab)}")




# special tokens:
"""
layout:    
        <layout_0> </layout_0> ... <layout_99> </layout_99>
text:    
        <text_0> </text_0> ... <text_99> </text_99>
text_layout:    
        <text_layout_0> ... <text_layout_99>
loc:
    <loc_0>, <loc_1>, <loc_500>
    
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
"""
special_to_add = []
for i in range(0,100):
    special_to_add.append(f'<layout_{i}>')
    special_to_add.append(f'</layout_{i}>')
for i in range(0, 100):
    special_to_add.append(f'<text_{i}>')
    special_to_add.append(f'</text_{i}>')
for i in range(0, 100):
    special_to_add.append(f'<text_layout_{i}>')

for i in range(0, 501):
    special_to_add.append(f'<loc_{i}>')

print(f"special vocab to add len: {len(special_to_add)}")
processor.tokenizer.add_special_tokens({"additional_special_tokens": special_to_add})
print(f"special added vocab len: {len(processor.tokenizer.vocab)}")

print(f"sep vocab to add len: 1")
processor.tokenizer.add_special_tokens({"sep_token": "<sep>"})
print(f"special added vocab len: {len(processor.tokenizer.vocab)}")


with torch.no_grad():
    mean_emb = torch.mean(model.decoder.embed_tokens.weight, dim=0).data
    origin_emb_len = len(model.decoder.embed_tokens.weight)
    added_len = len(processor.tokenizer.vocab) - origin_emb_len
    model.resize_token_embeddings(len(processor.tokenizer.vocab))
    model.decoder.embed_tokens.weight = nn.Parameter(
        torch.concat([model.decoder.embed_tokens.weight[:origin_emb_len], mean_emb.repeat(added_len, 1)])
    )
    print(f"origin_emb_len is {origin_emb_len}, added_len is {added_len}, now model emb shape {model.decoder.embed_tokens.weight.shape}")


model.save_pretrained(target)
processor.save_pretrained(target)

model2 = Pix2StructForConditionalGeneration.from_pretrained(target)
print(model2)
print(model2.config)
print(f"model2 emb shape {model2.decoder.embed_tokens.weight.shape}")

