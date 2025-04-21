# analyze_ads_model.py

import io, re, os
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PARQUET_PATH = "DATA/train-00001-of-00002-823ac5dae71e0e87.parquet"
OUTPUT_DIR   = "outputs"
N_COLOR_BINS = 16
N_TEXT_PCA   = 50
TSNE_PERC    = 30
K_CLUSTERS   = 8
LDA_TOPICS   = 5
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── HELPERS ────────────────────────────────────────────────────────────────────
def parse_dim(s):
    m = re.match(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def load_and_clean(path):
    df = pd.read_parquet(path)
    df[["width","height"]] = df["dimensions"].apply(parse_dim).tolist()
    df.dropna(subset=["width","height"], inplace=True)
    df = df[(df.width>0)&(df.height>0)]
    df[["width","height"]] = df[["width","height"]].astype(int)
    df["aspect_ratio"] = df.width / df.height
    return df.reset_index(drop=True)


# ─── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def extract_image_features(df):
    base = MobileNet(include_top=False, pooling="avg",
                     input_shape=(224,224,3), weights="imagenet")

    img_feats, color_feats = [], []
    for rec in df.image:
        raw_bytes = rec["bytes"]
        img = Image.open(io.BytesIO(raw_bytes)) \
                   .convert("RGB") \
                   .resize((224,224))

        # deep features
        arr = preprocess_input(np.array(img))
        img_feats.append(base.predict(arr[None], verbose=0)[0])

        # color histogram
        hist = []
        for c in range(3):
            h,_ = np.histogram(np.array(img)[...,c].ravel(),
                               bins=N_COLOR_BINS, range=(0,255), density=True)
            hist.append(h)
        color_feats.append(np.concatenate(hist))

    return np.stack(img_feats), np.stack(color_feats)


def extract_text_and_tfidf(df):
    texts = []
    for rec in df.image:
        img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        texts.append(pytesseract.image_to_string(img))

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = tfidf.fit_transform(texts)

    pca = PCA(n_components=N_TEXT_PCA, random_state=RANDOM_STATE)
    X_text_pca = pca.fit_transform(X_tfidf.toarray())

    pd.to_pickle(tfidf, os.path.join(OUTPUT_DIR, "tfidf.pkl"))
    pd.to_pickle(pca, os.path.join(OUTPUT_DIR, "text_pca.pkl"))

    return X_text_pca, texts, tfidf


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_clean(PARQUET_PATH)
    print("Loaded", len(df), "ads")

    # 1) Extract features
    X_img,   X_color = extract_image_features(df)
    X_text,  texts,  tfidf  = extract_text_and_tfidf(df)

    # 2) Compute "emphasis" metadata:
    contrasts, red_props, text_lengths, word_counts = [], [], [], []
    for rec, txt in zip(df.image, texts):
        # load & resize
        img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB").resize((224,224))
        gray = np.array(img.convert("L"))
        contrasts.append(gray.std()/255.0)

        arr = np.array(img)
        red_mask = (arr[:,:,0] > 150) & (arr[:,:,0] > arr[:,:,1]) & (arr[:,:,0] > arr[:,:,2])
        red_props.append(red_mask.mean())

        text_lengths.append(len(txt))
        word_counts.append(len(txt.split()))

    df["contrast"]    = contrasts
    df["red_prop"]    = red_props
    df["text_length"]= text_lengths
    df["word_count"]  = word_counts

    # 3) Build meta‐feature matrix
    X_meta = df[["width","height","aspect_ratio",
                 "contrast","red_prop","text_length","word_count"]].values

    # 4) Stack everything
    from numpy import hstack
    X = hstack([X_img, X_color, X_text, X_meta])
    print("Feature matrix shape:", X.shape)

    # ─── UNSUPERVISED ANALYSIS ──────────────────────────────────────────────────

    # 5) PCA → t‑SNE
    X_pca = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(X)
    X_tsne = TSNE(n_components=2, perplexity=TSNE_PERC,
                  random_state=RANDOM_STATE).fit_transform(X_pca)

    plt.figure(figsize=(8,8))
    plt.scatter(X_tsne[:,0], X_tsne[:,1],
                c=pd.cut(df.aspect_ratio,5,labels=False),
                alpha=0.6, cmap="tab10")
    plt.title("t-SNE of Ads (by aspect-ratio)")
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne.png"))
    plt.close()

    # 6) K‑Means
    df["cluster"] = KMeans(n_clusters=K_CLUSTERS,
                           random_state=RANDOM_STATE).fit_predict(X_pca)
    print("Cluster sizes:\n", df.cluster.value_counts())

    # 7) LDA on text
    X_tfidf_all = tfidf.transform(texts)
    lda = LatentDirichletAllocation(n_components=LDA_TOPICS,
                                    random_state=RANDOM_STATE)
    df["topic"] = lda.fit_transform(X_tfidf_all).argmax(axis=1)

    fnames = tfidf.get_feature_names_out()
    for i, comp in enumerate(lda.components_):
        terms = [fnames[idx] for idx in comp.argsort()[-10:][::-1]]
        print(f"Topic {i}: {', '.join(terms)}")

    # 8) Cluster × Topic
    ctab = pd.crosstab(df.cluster, df.topic)
    ctab.to_csv(os.path.join(OUTPUT_DIR, "cluster_topic_crosstab.csv"))
    print("Cross−tab:\n", ctab)

    # 9) Emphasis summary per cluster
    summary = df.groupby("cluster")[[
        "contrast","red_prop","text_length","word_count"
    ]].mean()
    print("Emphasis by cluster:\n", summary)

    # 10) Save
    df.to_parquet(os.path.join(OUTPUT_DIR, "ads_with_clusters.parquet"),
                  index=False)

    print("All artifacts in", OUTPUT_DIR)
