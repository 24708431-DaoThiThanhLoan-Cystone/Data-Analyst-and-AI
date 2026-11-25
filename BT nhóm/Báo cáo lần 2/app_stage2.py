"""Streamlit UI - IMDb Top 250: Regression + Classification + K-Means.
Run locally: streamlit run app_stage2.py
Docker: docker compose up --build
"""
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="IMDb Top 250 - Movie Intelligence", page_icon="🎬", layout="wide")

FEATURE_SPEC = {
    "year": (1888, 2026, 1, False, "Năm phát hành"),
    "rank_v1": (1, 250, 1, True, "Thứ hạng bảng cũ"),
    "rating_v1": (0, 10, 0.1, True, "IMDb rating bảng cũ"),
    "votes_v1": (0, 5000000, 1000, True, "Lượt bình chọn bảng cũ"),
    "runtime_v1": (1, 500, 1, True, "Thời lượng bảng cũ (phút)"),
    "votes_2026": (0, 10000000, 1000, False, "Lượt bình chọn 2026"),
    "runtime_2026": (1, 500, 1, False, "Thời lượng 2026 (phút)"),
    "metascore": (0, 100, 1, True, "Metascore"),
    "budget": (0, 1000000000, 100000, True, "Ngân sách (USD)"),
    "box_office": (0, 5000000000, 100000, True, "Doanh thu (USD)"),
}

@st.cache_resource
def load_artifact():
    return joblib.load(os.getenv("MODEL_PATH", "stage2_models.joblib"))

def get_mongo_collection():
    """MongoDB optional at runtime; app remains usable if DB is unavailable."""
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"),
                             serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        return client, client["imdb_project"]["prediction_history"]
    except Exception:
        return None, None

def build_inputs(artifact):
    defaults = artifact["feature_defaults"]
    values = {}
    cols = st.columns(2)
    for i, name in enumerate(artifact["feature_cols"]):
        low, high, step, optional, label = FEATURE_SPEC.get(
            name, (0.0, max(1.0, float(defaults.get(name, 1)) * 3), 0.1, True, name)
        )
        default = float(np.clip(defaults.get(name, low), low, high))
        with cols[i % 2]:
            unknown = st.checkbox(f"Không có dữ liệu: {label}", value=False,
                                  key=f"unknown_{name}") if optional else False
            if unknown:
                values[name] = np.nan
            else:
                values[name] = st.slider(
                    label, min_value=float(low), max_value=float(high),
                    value=float(default), step=float(step), key=f"feature_{name}"
                )
    return pd.DataFrame([values], columns=artifact["feature_cols"])

def predict(artifact, frame):
    rating = float(artifact["regression_model"].predict(frame)[0])
    class_label = str(artifact["classification_model"].predict(frame)[0])
    z = artifact["cluster_scaler"].transform(
        artifact["cluster_imputer"].transform(frame)
    )
    cluster = int(artifact["kmeans_pca"].predict(
        artifact["pca"].transform(z)
    )[0])
    return rating, class_label, cluster

def save_prediction(frame, rating, class_label, cluster):
    client, collection = get_mongo_collection()
    if collection is None:
        return False, "MongoDB chưa kết nối; kết quả vẫn hiển thị nhưng chưa lưu lịch sử."
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {k: (None if pd.isna(v) else float(v))
                  for k, v in frame.iloc[0].to_dict().items()},
        "predicted_rating": rating,
        "predicted_class": class_label,
        "cluster_label": cluster,
    }
    collection.insert_one(record)
    client.close()
    return True, "Đã lưu lịch sử dự đoán vào MongoDB."

def main():
    st.title("🎬 IMDb Top 250 — Movie Intelligence")
    st.caption("Dự đoán rating • Phân lớp rating • Phân cụm phim")
    try:
        artifact = load_artifact()
    except FileNotFoundError:
        st.error("Thiếu stage2_models.joblib. Hãy chạy toàn bộ notebook đến mục xuất mô hình.")
        st.stop()

    tab_predict, tab_viz, tab_history, tab_info = st.tabs(
        ["Dự đoán", "PCA & K-Means", "Lịch sử", "Thông tin mô hình"]
    )
    with tab_predict:
        st.subheader("Nhập thuộc tính phim")
        frame = build_inputs(artifact)
        if st.button("Thực hiện dự đoán", type="primary", use_container_width=True):
            rating, class_label, cluster = predict(artifact, frame)
            st.session_state["last_prediction"] = (frame, rating, class_label, cluster)
        if "last_prediction" in st.session_state:
            frame, rating, class_label, cluster = st.session_state["last_prediction"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Rating dự đoán", f"{rating:.2f}/10")
            c2.metric("Nhóm rating", class_label)
            c3.metric("Cluster", f"Cụm {cluster}")
            if "rating_v1" in frame and pd.isna(frame.iloc[0]["rating_v1"]):
                st.warning("Thiếu rating_v1 — kết quả có thể kém tin cậy hơn.")
            if st.button("Lưu kết quả vào MongoDB"):
                try:
                    ok, message = save_prediction(frame, rating, class_label, cluster)
                    (st.success if ok else st.warning)(message)
                except Exception as exc:
                    st.error(f"Không thể lưu MongoDB: {exc}")
            st.caption("Kết quả mang tính tham khảo, không phải đánh giá chính thức của IMDb.")

    with tab_viz:
        viz = artifact.get("viz", {})
        cum = np.asarray(viz.get("cum_var", np.cumsum(
            artifact["pca"].explained_variance_ratio_)))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(cum) + 1), cum, marker="o", label="Phương sai tích lũy")
        for threshold in (0.85, 0.90, 0.95):
            ax.axhline(threshold, linestyle="--", label=f"{threshold:.0%}")
        ax.set(title="PCA — phương sai tích lũy", xlabel="Số component", ylabel="Variance")
        ax.legend()
        st.pyplot(fig)
        if "X_pca_2d" in viz:
            xy = np.asarray(viz["X_pca_2d"])
            labels = np.asarray(viz.get("labels", np.zeros(len(xy))))
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            points = ax2.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="viridis", alpha=.8)
            ax2.set(title="K-Means trên PCA", xlabel="PC1", ylabel="PC2")
            fig2.colorbar(points, ax=ax2, label="Cluster")
            st.pyplot(fig2)

    with tab_history:
        client, collection = get_mongo_collection()
        if collection is None:
            st.info("Chưa kết nối MongoDB. Chạy `docker compose up --build` để bật database.")
        else:
            docs = list(collection.find({}, {"_id": 0}).sort("_id", -1).limit(100))
            client.close()
            if docs:
                st.dataframe(pd.json_normalize(docs), use_container_width=True)
            else:
                st.info("Chưa có lịch sử dự đoán.")

    with tab_info:
        st.write("**Hồi quy:**", artifact.get("regression_model_name"))
        st.write("**Phân lớp:**", artifact.get("classification_model_name"))
        st.write("**Số biến đầu vào:**", len(artifact["feature_cols"]))
        st.write("**Biến đầu vào:**", ", ".join(artifact["feature_cols"]))
        st.write("**Quy tắc nhóm rating:**", artifact.get("rating_tier_rule", ""))
        st.warning("Dự đoán là kết quả mô hình học từ dữ liệu Top 250, không đại diện toàn bộ phim.")

if __name__ == "__main__":
    main()
