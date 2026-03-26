# =============================================================
#  app.py  —  Streamlit GUI  (Stage 7)
#  All ML/NLP logic is in ml_pipeline.py (Stages 1–6)
#
#  Run:   streamlit run app.py
#  Need:  ml_pipeline.py · Resume.csv · job_dataset.csv
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ml_pipeline import (
    resume_df, job_df,
    label_encoder, results, results_df,
    best_name, best_stats, predictions,
    X_test, y_test,
    analyze_match,
)

# =============================================================
# Page config
# =============================================================
st.set_page_config(page_title="Resume Job Matcher", page_icon="📄", layout="wide")
st.title("📄 Resume & Job Matching System")
st.caption("NLP · Skill Extraction · Semantic Skill Expansion · Multi-Model Classification")
st.markdown("---")

# =============================================================
# Sidebar
# =============================================================
with st.sidebar:
    st.header("📊 Dataset Info")
    st.metric("Total Resumes",     len(resume_df))
    st.metric("Resume Categories", resume_df["Category"].nunique())
    st.metric("Job Postings",      len(job_df))
    st.metric("Unique Job Titles", job_df["Title"].nunique())

    st.markdown("---")
    st.subheader("🏆 Best Model")
    st.success(f"**{best_name}**")
    st.metric("Accuracy",  f"{best_stats['Accuracy (%)']}%")
    st.metric("F1-Score",  f"{best_stats['F1-Score (%)']}%")
    st.metric("Precision", f"{best_stats['Precision (%)']}%")
    st.metric("Recall",    f"{best_stats['Recall (%)']}%")

    st.markdown("---")
    st.subheader("📂 Resume Categories")
    for cat, cnt in resume_df["Category"].value_counts().head(10).items():
        st.write(f"• {cat}: **{cnt}**")

# =============================================================
# Tabs
# =============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Match Analyzer",
    "📈 Model Comparison",
    "🔲 Confusion Matrix",
    "🗂️ Dataset Explorer",
])


# =============================================================
# TAB 1 — Match Analyzer
# =============================================================
with tab1:
    st.subheader("Resume ↔ Job Description Analyzer")
    st.caption(
        "Match score uses **skill coverage + Jaccard + TF-IDF** blended scoring. "
        "Category is predicted using **skill-based rules** (more accurate than pure ML on short text)."
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**📋 Your Resume**")
        resume_input = st.text_area(
            "resume", height=280, label_visibility="collapsed",
            placeholder="Paste resume text here…\n\nExample:\nPython developer with 2 years experience. Skills: SQL, Pandas, NumPy, Scikit-learn, PyTorch, Data Science, Machine Learning, AWS.",
        )
    with col2:
        st.markdown("**💼 Job Description**")
        jd_input = st.text_area(
            "jd", height=280, label_visibility="collapsed",
            placeholder="Paste job description here…\n\nExample:\nLooking for a Python Developer with experience in machine learning, data analysis, and cloud platforms.",
        )

    st.markdown("")
    analyze_btn = st.button("🚀 Analyze Match", use_container_width=True, type="primary")

    if analyze_btn:
        if not resume_input.strip() or not jd_input.strip():
            st.warning("⚠️ Please fill in both fields.")
        else:
            with st.spinner("Running analysis…"):
                out = analyze_match(resume_input, jd_input)

            score    = out["match_score"]
            matching = out["matching_skills"]
            missing  = out["missing_skills"]
            r_skills = out["resume_skills"]
            pred_cat = out["predicted_category"]

            # Metric cards
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Match Score",     f"{score}%")
            m2.metric("Resume Skills",   len(r_skills))
            m3.metric("Matching Skills", len(matching))
            m4.metric("Missing Skills",  len(missing))

            # Score bar
            bar_color = "#2ecc71" if score >= 70 else "#f39c12" if score >= 40 else "#e74c3c"
            st.markdown(
                f"""<div style="background:#e0e0e0;border-radius:8px;height:22px;margin:8px 0">
                  <div style="background:{bar_color};width:{score}%;height:22px;
                    border-radius:8px;display:flex;align-items:center;
                    padding-left:10px;color:white;font-size:13px;font-weight:600">
                    {score}%
                  </div></div>""",
                unsafe_allow_html=True,
            )

            # Verdict
            if score >= 70:
                st.success("✅ Strong Match — You are a great fit for this role!")
            elif score >= 40:
                st.warning("⚠️ Moderate Match — You qualify but some skills are missing.")
            else:
                st.error("❌ Weak Match — Significant skill gaps detected.")

            st.info(f"🏷️ Predicted Resume Category: **{pred_cat}**")

            # Skills breakdown
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**✅ Matching Skills**")
                if matching:
                    for s in matching: st.markdown(f"🟢 {s}")
                else:
                    st.write("No overlapping skills found.")
            with c2:
                st.markdown("**❌ Missing Skills**")
                if missing:
                    for s in missing: st.markdown(f"🔴 {s}")
                else:
                    st.write("No missing skills — great!")
            with c3:
                st.markdown("**📝 Your Skills**")
                if r_skills:
                    for s in r_skills[:20]: st.markdown(f"• {s}")
                    if len(r_skills) > 20: st.caption(f"…and {len(r_skills)-20} more")
                else:
                    st.write("No skills detected.")

            # Top job matches from dataset
            st.markdown("---")
            st.subheader("🔎 Top Matching Jobs from Dataset")
            if r_skills:
                resume_skill_set = {s.lower() for s in r_skills}
                match_rows = []
                for _, row in job_df.iterrows():
                    job_skills = {s.strip().lower() for s in str(row["Skills"]).split(";")}
                    overlap = resume_skill_set & job_skills
                    if overlap:
                        match_rows.append({
                            "Job Title":          row["Title"],
                            "Experience Level":   row["ExperienceLevel"],
                            "Years Required":     row["YearsOfExperience"],
                            "Skills Matched":     len(overlap),
                            "Matched Skills":     ", ".join(sorted(overlap)[:5]),
                        })
                if match_rows:
                    top_matches = (
                        pd.DataFrame(match_rows)
                        .sort_values("Skills Matched", ascending=False)
                        .drop_duplicates(subset=["Job Title", "Experience Level"])
                        .head(10)
                        .reset_index(drop=True)
                    )
                    top_matches.index += 1
                    st.dataframe(top_matches, use_container_width=True)
                else:
                    st.write("No matching jobs found in the dataset.")


# =============================================================
# TAB 2 — Model Comparison
# =============================================================
with tab2:
    st.subheader("📈 Model Performance Comparison")
    st.caption("Trained on Resume.csv · TF-IDF features · 80/20 stratified split")

    display_df = results_df.copy().reset_index()
    display_df.index = display_df.index + 1

    def highlight_best(row):
        if row["Accuracy (%)"] == results_df["Accuracy (%)"].max():
            return ["background-color:#d4edda;font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_best, axis=1)
            .format({"Accuracy (%)":"{:.2f}","F1-Score (%)":"{:.2f}",
                     "Precision (%)":"{:.2f}","Recall (%)":"{:.2f}"}),
        use_container_width=True,
    )

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(results_df)); w = 0.2
    metrics = ["Accuracy (%)", "F1-Score (%)", "Precision (%)", "Recall (%)"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x+i*w, results_df[m], w, label=m, color=c, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels(results_df.index, fontsize=10)
    ax.set_ylim(0, 112); ax.set_ylabel("Score (%)")
    ax.set_title("Accuracy · F1 · Precision · Recall", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🏆 Best Model Details")
    st.markdown(f"""
| Criterion | Value |
|---|---|
| **Model** | {best_name} |
| **Accuracy** | {best_stats['Accuracy (%)']}% |
| **F1-Score** | {best_stats['F1-Score (%)']}% |
| **Precision** | {best_stats['Precision (%)']}% |
| **Recall** | {best_stats['Recall (%)']}% |
| **Feature Method** | TF-IDF · 1–2 grams · 5000 features · sublinear_tf |
| **Train / Test Split** | 80% / 20% stratified |
| **Dataset** | Resume.csv — {len(resume_df)} samples, {len(label_encoder.classes_)} classes |
    """)


# =============================================================
# TAB 3 — Confusion Matrix
# =============================================================
with tab3:
    st.subheader("🔲 Confusion Matrices")
    view = st.radio("Display mode", ["All Models (2×2 grid)", "Single Model"], horizontal=True)

    if view == "All Models (2×2 grid)":
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        axes = axes.flatten()
        for idx, (name, preds) in enumerate(predictions.items()):
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_,
                        ax=axes[idx], linewidths=0.4, annot_kws={"size": 7})
            axes[idx].set_title(f"{name}  (Acc: {results[name]['Accuracy (%)']}%)",
                                  fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Predicted", fontsize=9)
            axes[idx].set_ylabel("Actual", fontsize=9)
            axes[idx].tick_params(axis="x", rotation=45, labelsize=7)
            axes[idx].tick_params(axis="y", rotation=0,  labelsize=7)
        plt.suptitle("Confusion Matrices — All 4 Models", fontsize=15, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        selected = st.selectbox("Select a model", list(predictions.keys()))
        cm = confusion_matrix(y_test, predictions[selected])
        fig, ax = plt.subplots(figsize=(15, 11))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title(f"{selected}  |  Acc: {results[selected]['Accuracy (%)']}%  |  "
                     f"F1: {results[selected]['F1-Score (%)']}%",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)


# =============================================================
# TAB 4 — Dataset Explorer
# =============================================================
with tab4:
    st.subheader("🗂️ Dataset Explorer")
    d1, d2 = st.tabs(["Resume.csv", "job_dataset.csv"])

    with d1:
        st.caption(f"Resume.csv — {len(resume_df):,} rows")
        cat_filter = st.multiselect("Filter by Category",
                                     sorted(resume_df["Category"].unique()), default=[])
        show_r = resume_df[resume_df["Category"].isin(cat_filter)] if cat_filter else resume_df
        st.dataframe(show_r[["ID","Category","Resume_str"]].head(50), use_container_width=True)
        st.caption(f"Showing {min(50,len(show_r))} of {len(show_r):,} records")

    with d2:
        st.caption(f"job_dataset.csv — {len(job_df):,} rows")
        c1, c2 = st.columns(2)
        with c1:
            title_filter = st.multiselect("Filter by Title",
                                           sorted(job_df["Title"].dropna().unique()), default=[])
        with c2:
            exp_filter = st.multiselect("Filter by Experience Level",
                                         sorted(job_df["ExperienceLevel"].dropna().unique()), default=[])
        show_j = job_df.copy()
        if title_filter: show_j = show_j[show_j["Title"].isin(title_filter)]
        if exp_filter:   show_j = show_j[show_j["ExperienceLevel"].isin(exp_filter)]
        st.dataframe(
            show_j[["JobID","Title","ExperienceLevel","YearsOfExperience","Skills"]].head(50),
            use_container_width=True,
        )
        st.caption(f"Showing {min(50,len(show_j))} of {len(show_j):,} records")