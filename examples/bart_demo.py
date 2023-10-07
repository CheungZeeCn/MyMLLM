from transformers import BartForConditionalGeneration, BartTokenizer




model = BartForConditionalGeneration.from_pretrained("/home/ana/data4/models/bart-base", forced_bos_token_id=0)

tok = BartTokenizer.from_pretrained("/home/ana/data4/models/bart-base")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"])


# assert tok.batch_decode(generated_ids, skip_special_tokens=True) == [
#     "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
# ]
print(generated_ids)
