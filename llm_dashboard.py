# ~/llm_dashboard.py
import streamlit as st
import requests, time, pandas as pd
import matplotlib.pyplot as plt

BASE = "http://localhost:11434"

@st.cache_data
def get_models():
    return [m["name"] for m in requests.get(f"{BASE}/api/tags").json()["models"]]

def tokenize(model, text):
    # Ollama has no /api/tokenize endpoint; /api/generate reports
    # prompt_eval_count, which is the token count for the given prompt.
    r = requests.post(
        f"{BASE}/api/generate",
        json={"model": model, "prompt": text, "stream": False, "options": {"num_predict": 1}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["prompt_eval_count"]



def generate(model, prompt):
    r = requests.post(f"{BASE}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=120)
    r.raise_for_status()
    d = r.json()
    tps = d["eval_count"] / (d["eval_duration"] / 1e9)
    return {
        "response":       d["response"],
        "prompt_tokens":  d["prompt_eval_count"],
        "output_tokens":  d["eval_count"],
        "tokens_per_sec": round(tps, 1),
        "latency_s":      round(d["total_duration"] / 1e9, 2),
    }

# --- UI ---
st.title("Local LLM Telemetry Dashboard")
models = get_models()

text = st.text_input("Input text (for tokenization)", "The cat sat on the river bank")
if st.button("Compare tokenization"):
    rows = []
    for m in models:
        try:
            n_tokens = tokenize(m, text)
            rows.append({"model": m, "tokens": n_tokens, "chars/token": round(len(text)/n_tokens, 2)})
        except Exception as e:
            rows.append({"model": m, "tokens": None, "chars/token": None, "error": str(e)})
    st.dataframe(pd.DataFrame(rows))

prompt = st.text_area("Prompt (for generation telemetry)", "Explain what a neural network is in two sentences.")
if st.button("Run all models"):
    results = {}
    for m in models:
        with st.spinner(f"Running {m}…"):
            try:
                results[m] = generate(m, prompt)
            except Exception as e:
                st.error(f"{m} failed: {e}")

    # Metrics table
    df = pd.DataFrame(results).T
    st.dataframe(df[["prompt_tokens", "output_tokens", "tokens_per_sec", "latency_s"]])

    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(df.index, df["tokens_per_sec"])
    ax.set_ylabel("Tokens / second"); ax.set_title("Generation speed")
    st.pyplot(fig)

    # Responses
    for m, r in results.items():
        st.subheader(m)
        st.write(r["response"])