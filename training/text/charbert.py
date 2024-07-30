import charactertokenizer

tokenizer = charactertokenizer.CharacterTokenizer.from_pretrained(
    "ruaccent/RUAccent-encoder"
)

if __name__ == "__main__":
    prompt = "<s>У Лукоморья дуб зеленый\n"
    encoded_prompt = tokenizer.encode(prompt, return_tensors="pt")
    print(" | ".join(tokenizer.decode([t]) for t in encoded_prompt[0]))
