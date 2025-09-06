# main.py
# Pusula Data Science Intern Case — EDA & Preprocessing
# Author: Onur Karasürmeli
# Email: <replace-with-your-email>

import os, re, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter

DATA_PATH = "Talent_Academy_Case_DT_2025.xlsx"
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TARGET = "TedaviSuresi"

def normalize_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

class FrequencyEncoderSk(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names) if self.feature_names else pd.DataFrame(X)
        self.maps_ = {}
        for c in X_df.columns:
            vc = X_df[c].astype(str).fillna("__nan__").value_counts(normalize=True)
            self.maps_[c] = vc.to_dict()
        self.output_feature_names_ = list(X_df.columns)
        return self
    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.output_feature_names_)
        for c in X_df.columns:
            mp = self.maps_.get(c, {})
            X_df[c] = X_df[c].astype(str).fillna("__nan__").map(mp).fillna(0.0)
        return X_df.values
    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_feature_names_)

class MultiLabelBinarizerTopKSk(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=20, sep=",", feature_name="multi"):
        self.top_k = top_k
        self.sep = sep
        self.feature_name = feature_name
    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            return X.squeeze()
        elif isinstance(X, pd.Series):
            return X
        else:
            return pd.Series(X.reshape(-1))
    def fit(self, X, y=None):
        series = self._to_series(X).dropna().astype(str)
        tokens = []
        for s in series:
            parts = [normalize_text(p) for p in s.split(self.sep)]
            tokens.extend([p for p in parts if p])
        from collections import Counter
        counts = Counter(tokens)
        self.vocab_ = [t for t, _ in counts.most_common(self.top_k)]
        return self
    def transform(self, X):
        series = self._to_series(X).fillna("").astype(str)
        binarized = np.zeros((len(series), len(self.vocab_)), dtype=int)
        for i, s in enumerate(series):
            parts = [normalize_text(p) for p in s.split(self.sep) if normalize_text(p)]
            present = set(parts)
            for j, tok in enumerate(self.vocab_):
                binarized[i, j] = 1 if tok in present else 0
        return binarized
    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.feature_name}__{t}" for t in self.vocab_])

def main():
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_excel(DATA_PATH)
    raw_n = len(df)
    df.columns = [c.strip() for c in df.columns]

    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].apply(normalize_text)

    df = df.drop_duplicates()
    clean_n = len(df)

    expected_cols = [
        "HastaNo", "Yas", "Cinsiyet", "KanGrubu", "Uyruk",
        "KronikHastalik", "Bolum", "Alerji", "Tanilar",
        "TedaviAdi", "TedaviSuresi", "UygulamaYerleri", "UygulamaSuresi"
    ]
    colmap = {c.lower(): c for c in df.columns}
    aligned = {}
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    for ec in expected_cols:
        if ec.lower() in colmap:
            aligned[ec] = colmap[ec.lower()]
        else:
            m = next((col for col in df.columns if norm(col) == norm(ec)), None)
            if m is not None:
                aligned[ec] = m

    target_col = aligned.get(TARGET, TARGET) if aligned.get(TARGET, TARGET) in df.columns else None

    present_cols = set(df.columns)
    multi_label_candidates = ["KronikHastalik", "Alerji", "Tanilar", "UygulamaYerleri"]
    multi_label_cols = [aligned[c] for c in multi_label_candidates if c in aligned and aligned[c] in present_cols]

    numeric_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols_model = [c for c in numeric_cols_all if c not in [target_col, aligned.get("HastaNo")]]

    single_cat_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) and c not in multi_label_cols]
    for ex in [aligned.get("HastaNo"), target_col]:
        if ex in single_cat_cols:
            single_cat_cols.remove(ex)

    low_card_cols, high_card_cols = [], []
    for c in single_cat_cols:
        nunique = df[c].nunique(dropna=True)
        (low_card_cols if nunique <= 20 else high_card_cols).append(c)

    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    low_card_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    high_card_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("freq", FrequencyEncoderSk(feature_names=high_card_cols))])

    transformers = []
    if numeric_cols_model: transformers.append(("num", numeric_pipeline, numeric_cols_model))
    if low_card_cols: transformers.append(("low_card", low_card_pipeline, low_card_cols))
    if high_card_cols: transformers.append(("high_card", high_card_pipeline, high_card_cols))
    for col in multi_label_cols:
        transformers.append((f"mlb_{col}", MultiLabelBinarizerTopKSk(top_k=20, sep=",", feature_name=col), [col]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    X = df.drop(columns=[target_col] if target_col else [])
    y = df[target_col] if target_col else None

    X_prepared = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    X_prepared_df = pd.DataFrame(X_prepared, columns=feature_names, index=df.index)
    n_rows, n_features = X_prepared_df.shape
    out_dir_abs = os.path.abspath(OUT_DIR)

    df.to_csv(os.path.join(OUT_DIR, "dataset_cleaned.csv"), index=False)
    X_prepared_df.to_csv(os.path.join(OUT_DIR, "X_prepared.csv"), index=False)
    if y is not None:
        pd.DataFrame({TARGET: y}).to_csv(os.path.join(OUT_DIR, "y_target.csv"), index=False)
    joblib.dump(preprocessor, os.path.join(OUT_DIR, "preprocess_pipeline.joblib"))

    # ---- Run summary (write to file and print) ----
    summary_lines = [
        "Pusula Data Science Intern Case — Run Summary",
        f"Data path          : {os.path.abspath(DATA_PATH)}",
        f"Outputs directory  : {out_dir_abs}",
        "",
        f"Raw rows           : {raw_n}",
        f"Rows after cleaning: {clean_n} (removed {raw_n - clean_n} exact duplicates)",
        f"Feature matrix     : {n_rows} rows × {n_features} features",
        f"Target column      : {target_col if target_col else 'N/A'}",
        "",
        "Artifacts:",
        f" - dataset_cleaned.csv",
        f" - X_prepared.csv",
        f" - y_target.csv" if target_col else " - (no target saved; column missing)",
        f" - preprocess_pipeline.joblib",
        f" - figures/ (charts saved here)"
    ]
    summary_txt = "\n".join(summary_lines)
    with open(os.path.join(OUT_DIR, "RUN_SUMMARY.txt"), "w", encoding="utf-8") as fsum:
        fsum.write(summary_txt + "\n")
    print("\n" + summary_txt + "\n")
    print("Tip: Open the outputs directory above to view CSVs, pipeline, and figures.")

    # Example figures
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        fig = plt.figure()
        df[c].dropna().hist(bins=30)
        plt.title(f"Histogram: {c}"); plt.xlabel(c); plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"hist_{c}.png")); plt.close(fig)

    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        fig = plt.figure(figsize=(6,5))
        plt.imshow(corr, aspect="auto")
        plt.xticks(range(len(num_cols)), num_cols, rotation=90)
        plt.yticks(range(len(num_cols)), num_cols)
        plt.title("Correlation (numeric)"); plt.colorbar(); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "corr_numeric.png")); plt.close(fig)

    print(f"Artifacts written to: {out_dir_abs}")

    # Show saved figures interactively
    import glob
    fig_files = sorted(glob.glob(os.path.join(FIG_DIR, "*.png")))
    for fp in fig_files:
        img = plt.imread(fp)
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(fp))
        plt.show()

if __name__ == "__main__":
    main()
