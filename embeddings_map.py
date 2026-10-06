import torch, matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2Model

tok = GPT2Tokenizer.from_pretrained("gpt2")
E = GPT2Model.from_pretrained("gpt2").wte.weight.detach()

groups = {
    "animals":   [" cat", " dog", " horse", " cow", " lion", " bird"],
    "food":      [" bread", " rice", " cheese", " pizza", " banana", " orange"],
    "places":    [" London", " Paris", " Sydney", " India", " Tokyo", " Berlin"],
    "numbers":   [" one", " two", " three", " four", " five", " six"],
    "money":     [" money", " loan", " cash", " account", " credit", " deposit"],
    "nature":    [" river", " lake", " hill", " forest", " shore", " valley"],
    "tech":      [" Microsoft", " Google", " iPhone", " software", " computer", " Samsung"],
    "ambiguous": [" bank", " apple", " Apple"],
}

words, labels = [], []
for g, ws in groups.items():
    for w in ws:
        if len(tok.encode(w)) == 1:          # keep single-token words only
            words.append(w); labels.append(g)

X = torch.stack([E[tok.encode(w)[0]] for w in words])
X = X - X.mean(0)
U, S, V = torch.pca_lowrank(X, q=2)
P = (X @ V[:, :2]).numpy()

for g in groups:
    idx = [i for i, l in enumerate(labels) if l == g]
    plt.scatter(P[idx, 0], P[idx, 1], label=g, s=60)
    for i in idx:
        plt.annotate(words[i].strip(), P[i], fontsize=9, xytext=(4, 4), textcoords="offset points")
plt.legend(); plt.title("GPT-2 token embeddings → 2D (PCA)")
plt.tight_layout(); plt.show()