from chartk import CharacterTokenizer

MODEL_NAME = "RUAccent-stressed-encoder"
tokenizer = CharacterTokenizer.from_pretrained(MODEL_NAME)

SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

if __name__ == "__main__":
    prompt = SOS_TOKEN + "У Луком+орья дуб зелёный" + EOS_TOKEN
    encoded_prompt = tokenizer.encode(prompt, return_tensors="pt")
    print(" | ".join(tokenizer.decode([t]) for t in encoded_prompt[0]))
