# =============================================================
#  ml_pipeline.py  —  AI Resume Skill Extraction & Job Matching
#  Stages 1–6: Loading → Preprocessing → Features →
#              Training → Evaluation → Best Model Selection
#
#  Datasets:
#    • Resume.csv       — 2484 resumes with Category labels
#    • job_dataset.csv  — 1068 job postings (Skills, Title, etc.)
#
#  Colab install:
#    !pip install -q scikit-learn pandas numpy matplotlib seaborn
# =============================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


# =============================================================
# STAGE 1 — Dataset Loading and Exploration
# =============================================================
print("=" * 60)
print("  STAGE 1: DATASET LOADING AND EXPLORATION")
print("=" * 60)

resume_df = pd.read_csv("Resume.csv")
resume_df.dropna(subset=["Resume_str", "Category"], inplace=True)
resume_df.reset_index(drop=True, inplace=True)

print("\n[Resume.csv]")
print(f"  Shape    : {resume_df.shape}")
print(f"  Columns  : {resume_df.columns.tolist()}")
print(f"  Nulls    :\n{resume_df.isnull().sum().to_string()}")
cat_counts = resume_df["Category"].value_counts()
print(f"\n  Category distribution ({resume_df['Category'].nunique()} classes):")
print(cat_counts.to_string())

job_df = pd.read_csv("job_dataset.csv")
job_df.dropna(subset=["Skills", "Title"], inplace=True)
job_df.reset_index(drop=True, inplace=True)

exp_map = {
    "Junior": "Entry-Level", "Fresher": "Entry-Level",
    "Mid-level": "Mid-Level", "Mid-Senior": "Mid-Level",
    "Mid-Senior Level": "Mid-Level", "Senior": "Senior-Level", "Lead": "Senior-Level",
}
job_df["ExperienceLevel"] = job_df["ExperienceLevel"].replace(exp_map)

print("\n[job_dataset.csv]")
print(f"  Shape    : {job_df.shape}")
print(f"  Columns  : {job_df.columns.tolist()}")
print(f"  Nulls    :\n{job_df.isnull().sum().to_string()}")
print(f"\n  Experience levels:\n{job_df['ExperienceLevel'].value_counts().to_string()}")
print(f"\n  Top 10 job titles:\n{job_df['Title'].value_counts().head(10).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
cat_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Resume — Category Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Category"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)
job_df["ExperienceLevel"].value_counts().plot(kind="bar", ax=axes[1], color="darkorange", edgecolor="white")
axes[1].set_title("Job Dataset — Experience Level", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Experience Level"); axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("stage1_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
print("\n[Stage 1 complete] ✓")


# =============================================================
# STAGE 2 — Text Preprocessing
# =============================================================
print("\n" + "=" * 60)
print("  STAGE 2: TEXT PREPROCESSING")
print("=" * 60)

STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","they","them","the",
    "a","an","and","or","but","in","on","at","to","for","of","with","is","are",
    "was","were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","this","that","these","those",
    "from","by","as","if","not","no","so","up","out","about","into","than","then",
    "when","there","their","also","can","which","who","what","all","both","any",
    "each","more","other","such","only","own","same","too","very","just",
    "because","after","before","between","under","over","again","further","per",
}


def preprocess(text: str) -> str:
    """Strip HTML → lowercase → remove punctuation → drop stopwords."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


resume_df["clean_text"] = resume_df["Resume_str"].apply(preprocess)
job_df["combined_text"]  = (job_df["Skills"].fillna("") + " " +
                             job_df["Keywords"].fillna("") + " " +
                             job_df["Responsibilities"].fillna(""))
job_df["clean_text"] = job_df["combined_text"].apply(preprocess)

resume_df["token_count"] = resume_df["clean_text"].apply(lambda x: len(x.split()))
job_df["token_count"]    = job_df["clean_text"].apply(lambda x: len(x.split()))

print("\n  Sample raw resume (first 250 chars):")
print(" ", resume_df["Resume_str"].iloc[0][:250])
print("\n  After preprocessing:")
print(" ", resume_df["clean_text"].iloc[0][:250])
print(f"\n  Resume — Mean tokens: {resume_df['token_count'].mean():.0f} | "
      f"Min: {resume_df['token_count'].min()} | Max: {resume_df['token_count'].max()}")
print(f"  Job    — Mean tokens: {job_df['token_count'].mean():.0f} | "
      f"Min: {job_df['token_count'].min()} | Max: {job_df['token_count'].max()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(resume_df["token_count"], bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Resume — Token Distribution"); axes[0].set_xlabel("Token Count")
axes[1].hist(job_df["token_count"], bins=30, color="darkorange", edgecolor="white")
axes[1].set_title("Job Postings — Token Distribution"); axes[1].set_xlabel("Token Count")
plt.tight_layout()
plt.savefig("stage2_token_dist.png", dpi=120, bbox_inches="tight")
plt.show()
print("\n[Stage 2 complete] ✓")


# =============================================================
# STAGE 3 — Feature Extraction  (TF-IDF + Skill NER)
# =============================================================
print("\n" + "=" * 60)
print("  STAGE 3: FEATURE EXTRACTION")
print("=" * 60)

# ── Skill keyword list ───────────────────────────────────────
SKILL_KEYWORDS = [
    "python","java","javascript","typescript","c++","c#","sql","r","scala","go",
    "rust","kotlin","swift","php","ruby","matlab","bash","shell",".net","vb.net",
    "machine learning","deep learning","nlp","natural language processing",
    "data science","data analysis","data engineering","data visualization",
    "tensorflow","pytorch","keras","scikit-learn","scikit","pandas","numpy",
    "matplotlib","opencv","hugging face","bert","gpt","llm","transformers",
    "computer vision","statistics","regression","classification","clustering",
    "neural network","feature engineering","mlops","prompt engineering",
    "html","css","react","angular","vue","nodejs","django","flask","fastapi",
    "asp.net","mvc","razor","linq","entity framework","spring boot",
    "microservices","rest api","graphql","grpc","wordpress","bootstrap",
    "aws","azure","gcp","docker","kubernetes","git","linux","ci/cd","devops",
    "terraform","ansible","jenkins","airflow","unity","unreal engine",
    "excel","power bi","tableau","hadoop","spark","kafka",
    "mongodb","postgresql","mysql","redis","elasticsearch","sql server",
    "communication","teamwork","leadership","problem solving","agile","scrum",
    "project management","research","testing","debugging","cybersecurity",
    "networking","cloud","blockchain","photoshop","illustrator","figma",
    "indesign","sketch","adobe","salesforce","sap","oracle","hubspot",
    "software development","software engineering","programming","developer",
    "backend","frontend","api","rest","game development","ar","vr",
    "ethical hacking","penetration testing","fintech","seo","content writing",
]

# ── Skill aliases: map abbreviations / tools to broader concepts ──
SKILL_ALIASES = {
    "scikit":           ["scikit-learn", "machine learning", "data science"],
    "sklearn":          ["scikit-learn", "machine learning"],
    "numpy":            ["data science", "data analysis", "python"],
    "pandas":           ["data science", "data analysis", "python"],
    "pytorch":          ["deep learning", "machine learning"],
    "tensorflow":       ["deep learning", "machine learning"],
    "keras":            ["deep learning", "machine learning"],
    "flask":            ["backend", "python"],
    "django":           ["backend", "python"],
    "fastapi":          ["backend", "python"],
    "react":            ["frontend", "javascript"],
    "angular":          ["frontend", "javascript"],
    "vue":              ["frontend", "javascript"],
    "nodejs":           ["backend", "javascript"],
    "spring boot":      ["backend", "java"],
    "programming":      ["software development", "developer", "software engineering"],
    "software development": ["programming", "developer", "software engineering"],
    "developer":        ["software development", "programming", "software engineering"],
    "software engineering": ["software development", "programming", "developer"],
    "data science":     ["machine learning", "data analysis", "python"],
    "machine learning": ["data science", "data analysis", "python"],
    "deep learning":    ["machine learning", "data science", "neural network"],
    "nlp":              ["natural language processing", "machine learning"],
    "computer vision":  ["deep learning", "machine learning", "opencv"],
    "devops":           ["docker", "kubernetes", "linux", "ci/cd"],
    "cloud":            ["aws", "azure", "gcp"],
    "aws":              ["cloud"], "azure": ["cloud"], "gcp": ["cloud"],
}

# ── Category rules: skill-based classification (much more accurate) ──
CATEGORY_SKILL_MAP = {
    "INFORMATION-TECHNOLOGY": [
        "python","java","javascript","c++","c#",".net","software development",
        "software engineering","programming","developer","backend","frontend",
        "api","microservices","devops","docker","kubernetes","git","linux",
        "cloud","aws","azure","testing","debugging","rest api","nodejs",
    ],
    "DATA-SCIENCE": [
        "machine learning","deep learning","data science","nlp","pytorch",
        "tensorflow","scikit-learn","scikit","pandas","numpy","statistics",
        "data analysis","computer vision","neural network","regression",
        "classification","clustering","bert","llm","transformers","r",
    ],
    "DESIGNER": [
        "figma","photoshop","illustrator","ux","ui","design","graphic",
        "sketch","adobe","indesign","visual design","typography","wireframe",
    ],
    "DIGITAL-MEDIA": [
        "seo","content writing","social media","digital marketing","copywriting",
        "blog","video editing","content strategy",
    ],
    "FINANCE": [
        "finance","accounting","financial","investment","banking","tax",
        "audit","excel","financial modeling","portfolio","risk management",
    ],
    "HR": [
        "hr","recruitment","talent acquisition","human resources","hiring",
        "payroll","onboarding","performance management","employee relations",
    ],
    "SALES": [
        "sales","crm","salesforce","business development","account management",
        "lead generation","revenue","customer acquisition",
    ],
    "ENGINEERING": [
        "mechanical engineering","electrical engineering","civil engineering",
        "autocad","solidworks","manufacturing","structural","embedded",
    ],
    "HEALTHCARE": [
        "healthcare","medical","clinical","nursing","patient care",
        "pharmacology","diagnosis","electronic health records",
    ],
    "BANKING": [
        "banking","financial analysis","credit","loans","treasury",
        "compliance","risk","portfolio management","forex",
    ],
    "TEACHER": [
        "teaching","curriculum","lesson planning","education","classroom",
        "students","pedagogy","instruction","training",
    ],
    "CONSULTANT": [
        "consulting","strategy","business analysis","stakeholder","presentation",
        "change management","process improvement","advisory",
    ],
}


def extract_skills(text: str) -> list:
    """Keyword boundary matching + UPPERCASE acronym detection."""
    tl = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        if re.search(r"\b" + re.escape(skill) + r"\b", tl):
            found.append(skill.title())
    skip = {"ID","HR","THE","AND","FOR","NOT","ARE","HAS","WAS",
            "YOU","BUT","ALL","ITS","CAN","MAY","OUR","MVC","API"}
    for a in re.findall(r"\b[A-Z]{2,6}\b", text):
        if a.title() not in found and a not in skip:
            found.append(a)
    return list(dict.fromkeys(found))


def expand_skills(skill_set: set) -> set:
    """Add semantically related skills via alias map."""
    expanded = set(skill_set)
    for s in list(skill_set):
        if s in SKILL_ALIASES:
            aliases = SKILL_ALIASES[s]
            if isinstance(aliases, list):
                expanded.update(aliases)
            else:
                expanded.add(aliases)
    return expanded


# Build TF-IDF on Resume corpus (used for ML classification)
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_resume = tfidf_vectorizer.fit_transform(resume_df["clean_text"].tolist())

print(f"\n  TF-IDF matrix shape  : {X_resume.shape}")
print(f"  Vocabulary size      : {len(tfidf_vectorizer.vocabulary_)}")

feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_means   = np.asarray(X_resume.mean(axis=0)).ravel()
top20_idx     = tfidf_means.argsort()[-20:][::-1]
print(f"\n  Top 20 TF-IDF terms:")
for i in top20_idx:
    print(f"    {feature_names[i]:<35} {tfidf_means[i]:.5f}")

sample_skills = extract_skills(resume_df["Resume_str"].iloc[0])
print(f"\n  Skills from Resume[0]: {sample_skills[:12]}")
print("\n[Stage 3 complete] ✓")


# =============================================================
# STAGE 4 — Training Multiple Classification Models
# =============================================================
print("\n" + "=" * 60)
print("  STAGE 4: TRAINING MULTIPLE CLASSIFICATION MODELS")
print("=" * 60)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(resume_df["Category"].tolist())

print(f"\n  Target   : Resume Category ({len(label_encoder.classes_)} classes)")
print(f"  Classes  : {list(label_encoder.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_resume, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}  |  Feats : {X_train.shape[1]}")

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "SVM (Linear)":        SVC(kernel="linear", C=1.0),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

trained_models = {}
predictions    = {}

print("\n  Training...")
for name, clf in CLASSIFIERS.items():
    print(f"    → {name:<25}", end=" ", flush=True)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    trained_models[name] = clf
    predictions[name]    = preds
    print(f"Accuracy = {accuracy_score(y_test, preds)*100:.2f}%")

print("\n[Stage 4 complete] ✓")


# =============================================================
# STAGE 5 — Performance Evaluation and Model Comparison
# =============================================================
print("\n" + "=" * 60)
print("  STAGE 5: PERFORMANCE EVALUATION AND MODEL COMPARISON")
print("=" * 60)

results = {}
for name, preds in predictions.items():
    results[name] = {
        "Accuracy (%)":  round(accuracy_score(y_test, preds) * 100, 2),
        "F1-Score (%)":  round(f1_score(y_test, preds, average="weighted") * 100, 2),
        "Precision (%)": round(precision_score(y_test, preds, average="weighted", zero_division=0) * 100, 2),
        "Recall (%)":    round(recall_score(y_test, preds, average="weighted") * 100, 2),
    }

results_df = pd.DataFrame(results).T.sort_values("Accuracy (%)", ascending=False)
results_df.index.name = "Model"

print("\n  ── Model Performance Summary ─────────────────────────")
print(results_df.to_string())

print("\n  ── Per-Class Classification Reports ──────────────────")
for name, preds in predictions.items():
    print(f"\n  [{name}]")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_, zero_division=0))

# Bar chart
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(results_df)); w = 0.2
metrics = ["Accuracy (%)", "F1-Score (%)", "Precision (%)", "Recall (%)"]
colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
for i, (m, c) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x + i*w, results_df[m], w, label=m, color=c, alpha=0.85, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x + w*1.5); ax.set_xticklabels(results_df.index, fontsize=10)
ax.set_ylim(0, 112); ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("stage5_model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()

# All 4 confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(24, 20))
axes = axes.flatten()
for idx, (name, preds) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                ax=axes[idx], linewidths=0.4, annot_kws={"size": 7})
    axes[idx].set_title(f"{name}  (Acc: {results[name]['Accuracy (%)']}%)", fontsize=12, fontweight="bold")
    axes[idx].set_xlabel("Predicted", fontsize=9); axes[idx].set_ylabel("Actual", fontsize=9)
    axes[idx].tick_params(axis="x", rotation=45, labelsize=7)
    axes[idx].tick_params(axis="y", rotation=0,  labelsize=7)
plt.suptitle("Confusion Matrices — All 4 Models", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("stage5_confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.show()
print("\n[Stage 5 complete] ✓")


# =============================================================
# STAGE 6 — Best Model Selection
# =============================================================
print("\n" + "=" * 60)
print("  STAGE 6: BEST MODEL SELECTION")
print("=" * 60)

best_name  = results_df["Accuracy (%)"].idxmax()
best_model = trained_models[best_name]
best_stats = results[best_name]

print(f"\n  ✓  Best Model  : {best_name}")
print(f"     Accuracy   : {best_stats['Accuracy (%)']}%")
print(f"     F1-Score   : {best_stats['F1-Score (%)']}%")
print(f"     Precision  : {best_stats['Precision (%)']}%")
print(f"     Recall     : {best_stats['Recall (%)']}%")

cm_best = confusion_matrix(y_test, predictions[best_name])
fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
            ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title(f"Best Model: {best_name}  |  Acc: {best_stats['Accuracy (%)']}%  |  F1: {best_stats['F1-Score (%)']}%",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=11); ax.set_ylabel("True Label", fontsize=11)
ax.tick_params(axis="x", rotation=45, labelsize=8); ax.tick_params(axis="y", rotation=0, labelsize=8)
plt.tight_layout()
plt.savefig("stage6_best_model_cm.png", dpi=120, bbox_inches="tight")
plt.show()
print("\n[Stage 6 complete] ✓")
print("\n" + "=" * 60)
print("  ALL STAGES COMPLETE ✓  —  run: streamlit run app.py")
print("=" * 60)


# =============================================================
# HELPER FUNCTIONS  (used by app.py)
# =============================================================

def _get_skills_raw(text: str) -> set:
    """Return set of matched skill keywords (lowercase) from text."""
    tl = text.lower()
    found = set()
    for skill in SKILL_KEYWORDS:
        if re.search(r"\b" + re.escape(skill) + r"\b", tl):
            found.add(skill)
    return found


def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Blended match score (0–100%) using three signals:
      55%  Skill coverage  — what % of JD skills the resume covers
      25%  Jaccard         — overlap / union of expanded skill sets
      20%  TF-IDF cosine   — surface-level text similarity
    Skill aliases expand abbreviations (scikit→scikit-learn, etc.)
    so that semantically equivalent skills are counted correctly.
    """
    rs_raw = _get_skills_raw(resume_text)
    js_raw = _get_skills_raw(jd_text)
    rs = expand_skills(rs_raw)
    js = expand_skills(js_raw)

    # Coverage: fraction of JD skills covered by resume
    if js:
        coverage = len(rs & js) / len(js)
    else:
        # JD has no detectable skills → use TF-IDF only
        coverage = 0.0

    # Jaccard on expanded sets
    union = rs | js
    jaccard = len(rs & js) / len(union) if union else 0.0

    # TF-IDF cosine on raw text
    rc = re.sub(r"[^a-z0-9\s]", " ", resume_text.lower())
    jc = re.sub(r"[^a-z0-9\s]", " ", jd_text.lower())
    try:
        vec   = TfidfVectorizer(ngram_range=(1, 2))
        tfidf = vec.fit_transform([rc, jc])
        tfidf_sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    except Exception:
        tfidf_sim = 0.0

    blended = (0.55 * coverage) + (0.25 * jaccard) + (0.20 * tfidf_sim)
    return round(min(max(blended, 0.0), 1.0) * 100, 1)


def predict_category(resume_text: str) -> str:
    """
    Predict resume job category.
    Uses skill-based rules first (more accurate for short/focused text),
    falls back to ML model if no clear rule-based winner.
    """
    tl = resume_text.lower()

    # Rule-based: score each category by skill keyword hits
    scores = {}
    for cat, keywords in CATEGORY_SKILL_MAP.items():
        score = sum(1 for kw in keywords
                    if re.search(r"\b" + re.escape(kw) + r"\b", tl))
        if score > 0:
            scores[cat] = score

    if scores:
        top_cat   = max(scores, key=scores.get)
        top_score = scores[top_cat]
        # Only trust rule-based if it has a clear winner (≥2 matches)
        if top_score >= 2:
            return top_cat

    # Fallback: ML model
    feat = tfidf_vectorizer.transform([preprocess(resume_text)])
    pred = best_model.predict(feat)[0]
    return label_encoder.inverse_transform([pred])[0]


def analyze_match(resume_text: str, jd_text: str) -> dict:
    """Full resume ↔ JD analysis. Returns dict for GUI."""
    rs_raw = _get_skills_raw(resume_text)
    js_raw = _get_skills_raw(jd_text)
    rs_exp = expand_skills(rs_raw)
    js_exp = expand_skills(js_raw)

    matching = sorted([s.title() for s in rs_exp & js_exp])
    missing  = sorted([s.title() for s in js_exp - rs_exp])

    # Display-friendly skill lists (original detected, not expanded)
    resume_skills = [s.title() for s in sorted(rs_raw)]
    jd_skills     = [s.title() for s in sorted(js_raw)]

    return {
        "match_score":        compute_match_score(resume_text, jd_text),
        "resume_skills":      resume_skills,
        "jd_skills":          jd_skills,
        "matching_skills":    matching,
        "missing_skills":     missing,
        "predicted_category": predict_category(resume_text),
    }