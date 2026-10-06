import torch
from transformers import GPT2Tokenizer, GPT2Model

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").eval()
E = model.wte.weight            # the embedding matrix
cos = torch.nn.functional.cosine_similarity

with torch.no_grad():
    # 1. The raw lookup table
    print("Embedding matrix:", tuple(E.shape))   # (50257 tokens, 768 dims)
    ids = tok.encode("The cat sat on the mat.")
    for i in ids:
        print(f"{tok.decode([i])!r:8} id={i:<6} first 5 dims: {[round(x, 3) for x in E[i][:5].tolist()]}")

    # 2. Static similarity: are related words close together?
    def static(word):
        t = tok.encode(word)
        assert len(t) == 1, f"{word!r} is {len(t)} tokens"
        return E[t[0]]
    print("\nStatic embedding similarity:")
    for a, b in [(" cat", " dog"), (" cat", " car"), (" cat", " banana"), (" king", " queen")]:
        print(f"{a:7} vs {b:8} {cos(static(a), static(b), dim=0).item():.3f}")

    # 3. Context: the same token in different sentences
    def contextual(sentence, word):
        enc = tok(sentence, return_tensors="pt")
        hidden = model(**enc).last_hidden_state[0]
        pos = enc.input_ids[0].tolist().index(tok.encode(word)[0])
        return hidden[pos]
    river = contextual("We fished from the grassy river bank", " bank")
    money = contextual("I deposited my salary at the bank", " bank")
    money2 = contextual("She opened a savings account at the bank", " bank")
    print("\nContextual ' bank':")
    print(f"money  vs money2: {cos(money, money2, dim=0).item():.3f}")
    print(f"river  vs money : {cos(river, money, dim=0).item():.3f}")