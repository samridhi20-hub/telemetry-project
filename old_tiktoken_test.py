import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("The cat sat on the mat.")
print(ids)
print([enc.decode([i]) for i in ids])