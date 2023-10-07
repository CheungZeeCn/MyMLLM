from transformers import AutoModelForCausalLM, LlamaForCausalLM, GPT2LMHeadModel
from transformers import AutoTokenizer

# model_path = '/home/ana/data4/models/chinese-llama-2-7b'
model_path = '/home/ana/data4/models/gpt2-chinese-cluecorpussmall'
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", load_in_4bit=True
)

tokenizer=AutoTokenizer.from_pretrained(model_path)
model_inputs = tokenizer(["今天天气不错，挺"], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs, max_new_tokens=128)

output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(output)