from chartk import CharacterTokenizer

MODEL_NAME = "RUAccent-stressed-encoder"
tokenizer = CharacterTokenizer.from_pretrained(MODEL_NAME)

if __name__ == "__main__":
    prompt = "<s>У Луком+орья дуб зеленый\n"
    encoded_prompt = tokenizer.encode(prompt, return_tensors="pt")
    print(" | ".join(tokenizer.decode([t]) for t in encoded_prompt[0]))
