from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="spa_Latn")
lengths = []

with open("awajun-spanish-v1/train.es", encoding="utf-8") as f:
    for line in f:
        tokens = tokenizer.encode(line.strip(), add_special_tokens=True)
        lengths.append(len(tokens))

print(f"Máximo train.es: {max(lengths)}, Promedio: {sum(lengths)//len(lengths)}")
