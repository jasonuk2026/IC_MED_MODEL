from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')
# model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base')
inputs = tokenizer("Sunitinib is a tyrosine kinase inhibitor", return_tensors="pt")

print(inputs.input_ids[0].size().item())
print(tokenizer.all_special_ids)
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
print(tokenizer.convert_ids_to_tokens([    2, 26673,  1744,    42,  6671,  4000,  3922,     3]))