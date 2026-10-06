import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("The cat sat on the mat.")
print(ids)
print([enc.decode([i]) for i in ids])


for text in ["Adelaide", "adelaide", " Adelaide", "unbelievably", "नमस्ते", "12345678"]:
    ids = enc.encode(text)
    print(f"{text!r:16} {len(ids)} tokens → {[enc.decode([i]) for i in ids]}")

    