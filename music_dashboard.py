"""
music_dashboard.py  —  Streamlit
─────────────────────────────────
Dépendances :
    pip install streamlit plotly pyspark librosa yt-dlp

Lancement :
    streamlit run music_dashboard.py
"""

import os
import tempfile
import subprocess
import warnings
from pathlib import Path

import dotenv

dotenv.load_dotenv()

if os.getenv("CESTQUIQUIADESPROBLEMESAVECSPARK") == "Leo":
    os.environ["PYSPARK_PYTHON"]        = r"C:\spark-env\Scripts\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\spark-env\Scripts\python.exe"
    os.environ["HADOOP_HOME"]           = r"C:\hadoop"
    os.environ["PATH"]                  = r"C:\hadoop\bin;" + os.environ["PATH"]

warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

from yt_dlp import YoutubeDL

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.extract_all_concise import extract_udf

COLORS = px.colors.qualitative.D3

@st.cache_resource
def get_spark():
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_resource
def load_raw():
    spark = get_spark()
    df = spark.read.parquet("./data/features/training_dataset")

    genre_col = "genre" if "genre" in df.columns else "genres"
    if genre_col == "genres":
        df = df.withColumn("genres", F.explode(F.col("genres")))
    else:
        df = df.withColumnRenamed("genre", "genres")

    user_ids = (
        df.select("user_id")
        .distinct()
        .filter(F.col("user_id").isNotNull())
        .orderBy("user_id")
        .toPandas()["user_id"]
        .tolist()
    )

    date_range = df.agg(
        F.min("uploaded_at").alias("date_min"),
        F.max("uploaded_at").alias("date_max"),
    ).toPandas().iloc[0]

    meta = {
        "user_ids": user_ids,
        "uploaded_at_min": date_range["date_min"],
        "uploaded_at_max": date_range["date_max"],
        "has_duration": "duration" in df.columns,
    }
    return df, meta


def compute_aggregates(df):
    total_tracks = df.count()

    agg_base = [
        F.round(F.mean("tempo"), 1).alias("tempo_mean"),
        F.countDistinct("genres").alias("n_genres"),
    ]
    if "duration" in df.columns:
        agg_base.append(F.round(F.mean("duration"), 1).alias("duration_avg"))

    global_stats = df.agg(*agg_base).toPandas().iloc[0]
    has_duration = "duration" in df.columns
    duration_avg = float(global_stats["duration_avg"] or 0) if has_duration else None

    return {
        "total": total_tracks,
        "tempo_mean": float(global_stats["tempo_mean"] or 0),
        "n_genres": int(global_stats["n_genres"] or 0),
        "duration_avg": duration_avg,
        "genres_df": _build_genres_df(df, total_tracks),
        "tempo_buckets": _build_tempo_buckets(df),
        "stats": _build_stats_by_genre(df),
        "zcr_flatness": _build_zcr_flatness_with_percentile(df),
        "feature_stats": _build_global_feature_stats(df),
    }

@st.cache_resource
def load_model():
    model_path = "data/model/rf_pipeline"
    if not os.path.exists(model_path):
        return None
    try:
        return PipelineModel.load(model_path)
    except Exception as e:
        st.warning(f"Impossible de charger le modèle : {e}")
        return None


def _build_genres_df(df, total_tracks):
    w_rank  = Window.orderBy(F.col("count").desc())
    w_cumul = Window.orderBy(F.col("count").desc()).rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )

    genres_df = (
        df.groupBy("genres").count()
        .withColumn("rank",      F.rank().over(w_rank))
        .withColumn("cumul_pct", F.round(
            F.sum("count").over(w_cumul) / total_tracks * 100, 1
        ))
        .orderBy("rank")
        .limit(12)
        .toPandas()
    )
    genres_df["pct"] = (
        genres_df["count"] / total_tracks * 100
    ).round(1) if total_tracks else 0

    return genres_df


def _build_tempo_buckets(df):
    bucket_order = ["< 80", "80-100", "100-110", "110-120", "120-130", "130-140", "140-160", "160+"]

    bucket_sort = (
        F.when(F.col("bucket") == "< 80",    0)
         .when(F.col("bucket") == "80-100",  1)
         .when(F.col("bucket") == "100-110", 2)
         .when(F.col("bucket") == "110-120", 3)
         .when(F.col("bucket") == "120-130", 4)
         .when(F.col("bucket") == "130-140", 5)
         .when(F.col("bucket") == "140-160", 6)
         .otherwise(7)
    )

    w_cumul = Window.orderBy(bucket_sort).rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    w_total = Window.partitionBy(F.lit(1))

    tempo_buckets = (
        df.withColumn("bucket",
            F.when(F.col("tempo") < 80,  "< 80")
             .when(F.col("tempo") < 100, "80-100")
             .when(F.col("tempo") < 110, "100-110")
             .when(F.col("tempo") < 120, "110-120")
             .when(F.col("tempo") < 130, "120-130")
             .when(F.col("tempo") < 140, "130-140")
             .when(F.col("tempo") < 160, "140-160")
             .otherwise("160+"))
        .groupBy("bucket").count()
        .withColumn("cumul",     F.sum("count").over(w_cumul))
        .withColumn("cumul_pct", F.round(
            F.col("cumul") / F.sum("count").over(w_total) * 100, 1
        ))
        .orderBy(bucket_sort)
        .toPandas()
    )

    tempo_buckets["bucket"] = pd.Categorical(
        tempo_buckets["bucket"], categories=bucket_order, ordered=True
    )
    return tempo_buckets


def _build_stats_by_genre(df):
    w_global = Window.partitionBy(F.lit(1))

    agg_exprs = [
        F.round(F.mean("tempo"), 1).alias("tempo"),
        F.round(F.mean("centroid_mean"), 0).alias("centroid"),
        F.round(F.mean("rms_mean"), 4).alias("rms"),
        F.round(F.mean("zcr_mean"), 4).alias("zcr"),
        F.round(F.mean("flatness_mean"), 5).alias("flatness"),
        F.round(F.mean("mfcc_1_mean"), 2).alias("mfcc1"),
        F.round(F.mean("mfcc_2_mean"), 2).alias("mfcc2"),
        F.round(F.mean("mfcc_3_mean"), 2).alias("mfcc3"),
        F.round(F.mean("mfcc_4_mean"), 2).alias("mfcc4"),
        F.round(F.mean("mfcc_5_mean"), 2).alias("mfcc5"),
        F.count("*").alias("count"),
    ]
    if "duration" in df.columns:
        agg_exprs.append(F.round(F.mean("duration"), 1).alias("duration_avg"))

    return (
        df.groupBy("genres").agg(*agg_exprs)
        .orderBy("count", ascending=False)
        .limit(12)
        .withColumn("global_centroid_avg", F.avg("centroid").over(w_global))
        .withColumn("global_centroid_std", F.stddev("centroid").over(w_global))
        .withColumn("z_centroid", F.round(
            (F.col("centroid") - F.col("global_centroid_avg"))
            / F.col("global_centroid_std"), 2
        ))
        .withColumn("global_rms_avg", F.avg("rms").over(w_global))
        .withColumn("global_rms_std", F.stddev("rms").over(w_global))
        .withColumn("z_rms", F.round(
            (F.col("rms") - F.col("global_rms_avg"))
            / F.col("global_rms_std"), 2
        ))
        .drop("global_centroid_avg", "global_centroid_std", "global_rms_avg", "global_rms_std")
        .toPandas()
        .fillna(0)
    )


def _build_zcr_flatness_with_percentile(df):
    w_zcr = Window.orderBy("zcr")
    w_flatness = Window.orderBy("flatness")

    return (
        df.groupBy("genres").agg(
            F.round(F.mean("zcr_mean"), 4).alias("zcr"),
            F.round(F.mean("flatness_mean"), 5).alias("flatness"),
            F.count("*").alias("count"),
        )
        .orderBy("count", ascending=False)
        .limit(12)
        .withColumn("pct_rank_zcr", F.round(F.percent_rank().over(w_zcr),      3))
        .withColumn("pct_rank_flatness", F.round(F.percent_rank().over(w_flatness), 3))
        .withColumn("noise_score", F.round((F.col("pct_rank_zcr") + F.col("pct_rank_flatness")) / 2, 3))
        .toPandas()
        .fillna(0)
    )


def _build_global_feature_stats(df):
    feature_cols = ["tempo", "centroid_mean", "rms_mean", "zcr_mean", "flatness_mean"]
    agg_exprs = []
    for c in feature_cols:
        agg_exprs += [
            F.round(F.mean(c), 6).alias(f"{c}__mean"),
            F.round(F.stddev(c), 6).alias(f"{c}__std"),
            F.round(F.min(c), 6).alias(f"{c}__min"),
            F.round(F.max(c), 6).alias(f"{c}__max"),
            F.round(F.expr(f"percentile_approx({c}, 0.10)"), 6).alias(f"{c}__p10"),
            F.round(F.expr(f"percentile_approx({c}, 0.90)"), 6).alias(f"{c}__p90"),
        ]
    return df.agg(*agg_exprs).toPandas().iloc[0].to_dict()


def predict_genre(track_feats: dict) -> list[tuple[str, float]] | None:
    model = load_model()
    if model is None:
        return None

    spark = get_spark()
    assembler = model.stages[1]
    rf_model = model.stages[2]
    indexer = model.stages[0]
    label_names = indexer.labels

    feature_cols = assembler.getInputCols()
    row_data = {col: float(track_feats.get(col, 0.0)) for col in feature_cols}
    row_df = spark.createDataFrame([row_data])
    row_df = assembler.transform(row_df)

    pred_pd = rf_model.transform(row_df).toPandas()
    if pred_pd.empty:
        return None

    proba_vec = pred_pd.iloc[0]["probability"].toArray()
    top3_idx  = proba_vec.argsort()[::-1][:3]
    return [(label_names[i], float(proba_vec[i])) for i in top3_idx]

def render_sidebar_filters(df_raw, meta):
    import datetime

    def _to_date(val):
        if val is None:
            return datetime.date.today()
        if isinstance(val, datetime.datetime):
            return val.date()
        if isinstance(val, datetime.date):
            return val
        return datetime.date.fromisoformat(str(val)[:10])

    global_date_min = _to_date(meta["uploaded_at_min"])
    global_date_max = _to_date(meta["uploaded_at_max"])

    with st.sidebar:
        st.header("🎛️ Filtres — stats globales")

        all_genres_raw = (
            df_raw.select("genres")
            .distinct()
            .filter(F.col("genres").isNotNull())
            .orderBy("genres")
            .toPandas()["genres"]
            .tolist()
        )

        selected_genres = st.multiselect(
            "Genres",
            options=all_genres_raw,
            default=[],
            placeholder="Tous les genres",
        )

        selected_users = st.multiselect(
            "Utilisateurs (user_id)",
            options=meta["user_ids"],
            default=[],
            placeholder="Tous les utilisateurs",
        )

        st.markdown("**Date d'upload**")
        date_from = st.date_input(
            "Du",
            value=global_date_min,
            min_value=global_date_min,
            max_value=global_date_max,
            key="date_from",
        )
        date_to = st.date_input(
            "Au",
            value=global_date_max,
            min_value=global_date_min,
            max_value=global_date_max,
            key="date_to",
        )
        if date_from > date_to:
            st.warning("⚠️ La date de début est postérieure à la date de fin.")

        duration_filter = None
        if meta["has_duration"]:
            dur_max_hint = int((meta.get("duration_avg_hint") or 300) * 3)
            duration_filter = st.slider(
                "Durée (secondes)",
                min_value=0,
                max_value=dur_max_hint,
                value=(0, dur_max_hint),
                step=10,
            )

        active = any([selected_genres, selected_users, date_from > global_date_min, date_to < global_date_max, duration_filter and duration_filter != (0, duration_filter[1])])
        st.caption("Les graphiques ci-contre reflètent les filtres actifs." if active else "Aucun filtre actif — affichage complet.")

    df = df_raw
    if selected_genres:
        df = df.filter(F.col("genres").isin(selected_genres))
    if selected_users:
        df = df.filter(F.col("user_id").isin(selected_users))
    if date_from != global_date_min:
        df = df.filter(F.col("uploaded_at") >= str(date_from))
    if date_to != global_date_max:
        df = df.filter(F.col("uploaded_at") <= str(date_to))
    if duration_filter and "duration" in df.columns:
        df = df.filter(
            (F.col("duration") >= duration_filter[0]) &
            (F.col("duration") <= duration_filter[1])
        )

    with st.spinner("Mise à jour des statistiques…"):
        d = compute_aggregates(df)

    d["_selected_users"] = selected_users
    d["_date_from"] = date_from
    d["_date_to"] = date_to
    d["_global_date_min"] = global_date_min
    d["_global_date_max"] = global_date_max

    return d


def show_track_feats(track_feats, feat_stats, stats):
    st.success("Features extraites ✔")
    _render_genre_prediction(track_feats)
    _render_track_kpis(track_feats)
    _render_bullet_charts(track_feats, feat_stats)
    _render_mfcc_radar(track_feats, stats)
    _render_chromagram(track_feats)
    _render_tonnetz_contrast(track_feats)
    st.divider()


def _render_genre_prediction(track_feats: dict) -> None:
    results = predict_genre(track_feats)

    if results is None:
        st.info(
            "ℹ️ Modèle non disponible — entraîne-le d'abord avec `train_model.py`. "
            "Le chemin attendu : `data/model/rf_pipeline`."
        )
        return

    st.markdown("#### 🤖 Prédiction du genre par le modèle")
    top_genre, top_proba = results[0]
    st.success(f"**Genre prédit : {top_genre}** — confiance {top_proba:.1%}")

    cols = st.columns(3)
    for i, (genre, proba) in enumerate(results):
        with cols[i]:
            st.metric(f"#{i+1}", genre, f"{proba:.1%}")
            st.progress(proba)

    st.divider()


def _render_track_kpis(track_feats):
    st.markdown("#### 📊 Features de ta piste")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tempo", f"{track_feats['tempo']:.1f} bpm")
    k2.metric("Durée", f"{track_feats['duration']:.0f} s")
    k3.metric("Centroid", f"{track_feats['centroid_mean']:.0f} Hz")
    k4.metric("RMS", f"{track_feats['rms_mean']:.4f}")
    k5.metric("ZCR", f"{track_feats['zcr_mean']:.4f}")


def _render_bullet_charts(track_feats, feat_stats):
    st.markdown("#### 🔍 Ta piste dans le contexte du dataset")
    st.write(
        "Chaque bullet chart positionne ta valeur (barre rouge) "
        "dans la distribution du dataset : gris foncé = P10–P90, gris clair = min–max."
    )

    compare_cfg = [
        ("Tempo (bpm)", "tempo", "{:.1f} bpm"),
        ("Centroid (Hz)","centroid_mean", "{:.0f} Hz"),
        ("RMS", "rms_mean", "{:.4f}"),
        ("ZCR", "zcr_mean", "{:.4f}"),
        ("Flatness", "flatness_mean", "{:.5f}"),
    ]

    for i in range(0, len(compare_cfg), 2):
        cols = st.columns(2)
        for j, (label, feat_key, _fmt) in enumerate(compare_cfg[i:i+2]):
            track_val = track_feats.get(feat_key, 0.0)
            ds_mean = feat_stats.get(f"{feat_key}__mean", 0.0) or 0.0
            ds_std = feat_stats.get(f"{feat_key}__std",  0.0) or 0.0
            ds_min = feat_stats.get(f"{feat_key}__min",  0.0) or 0.0
            ds_max = feat_stats.get(f"{feat_key}__max",  0.0) or 1.0
            ds_p10 = feat_stats.get(f"{feat_key}__p10", ds_min) or ds_min
            ds_p90 = feat_stats.get(f"{feat_key}__p90", ds_max) or ds_max

            fig_b = go.Figure(go.Indicator(
                mode="number+gauge",
                value=track_val,
                number={"valueformat": ".4g", "font": {"size": 22}},
                gauge={
                    "shape": "bullet",
                    "axis": {"range": [ds_min, ds_max]},
                    "threshold": {
                        "line": {"color": "#FF3333", "width": 3},
                        "thickness": 0.85,
                        "value": track_val,
                    },
                    "steps": [
                        {"range": [ds_min, ds_max], "color": "#e8e8e8"},
                        {"range": [ds_p10, ds_p90], "color": "#aaaaaa"},
                        {"range": [ds_mean - ds_std, ds_mean + ds_std], "color": "#666666"},
                    ],
                    "bar": {"color": "#FF3333", "thickness": 0.3},
                },
                title={"text": label, "font": {"size": 13}},
            ))
            fig_b.update_layout(height=110, margin=dict(l=10, r=10, t=30, b=10))

            with cols[j]:
                st.markdown(f"**{label}**")
                st.plotly_chart(fig_b, use_container_width=True)
                delta = track_val - ds_mean
                delta_pct = delta / ds_mean * 100 if ds_mean else 0
                sign = "+" if delta_pct >= 0 else ""
                color = "🔴" if abs(delta_pct) > 30 else "🟡" if abs(delta_pct) > 10 else "🟢"
                st.caption(f"{color} {sign}{delta_pct:.1f}% vs moyenne dataset")


def _render_mfcc_radar(track_feats: dict, stats: pd.DataFrame) -> None:
    st.markdown("#### 🕸️ Profil timbral — ta piste vs genres du dataset")
    st.write(
        "Ton profil MFCC 1–5 (rouge, premier plan) superposé aux "
        "6 genres les plus représentés du dataset."
    )

    top6 = stats.head(6)
    mfcc_labels = ["MFCC 1","MFCC 2","MFCC 3","MFCC 4","MFCC 5","MFCC 1"]
    fig_cmp = go.Figure()

    for i, row in top6.iterrows():
        vals = [row["mfcc1"], row["mfcc2"], row["mfcc3"],
                row["mfcc4"], row["mfcc5"]] + [row["mfcc1"]]
        fig_cmp.add_trace(go.Scatterpolar(
            r=vals, theta=mfcc_labels,
            fill="toself", opacity=0.22, name=row["genres"],
            line=dict(color=COLORS[i % len(COLORS)], width=1),
        ))

    track_mfcc = [track_feats.get(f"mfcc_{j}_mean", 0.0) for j in range(1, 6)]
    track_mfcc += track_mfcc[:1]
    fig_cmp.add_trace(go.Scatterpolar(
        r=track_mfcc, theta=mfcc_labels,
        fill="toself", opacity=0.6,
        name="🎵 Ta piste",
        line=dict(color="#FF3333", width=3),
        fillcolor="rgba(255,51,51,0.18)",
    ))

    fig_cmp.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=40, r=40, t=20, b=80),
        height=460,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


def _render_chromagram(track_feats: dict) -> None:
    st.markdown("#### 🎹 Empreinte chromatique de ta piste")
    st.write(
        "Le chromagramme moyen indique quelles notes (C → B) dominent dans ta piste. "
        "Utile pour identifier la tonalité et les couleurs harmoniques."
    )
    notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    chroma_vals = [track_feats.get(f"chroma_{i+1}_mean", 0.0) for i in range(12)]
    fig_chroma = px.bar_polar(
        r=chroma_vals, theta=notes,
        color=chroma_vals, color_continuous_scale="Plasma",
    )
    fig_chroma.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_chroma, use_container_width=True)


def _render_tonnetz_contrast(track_feats: dict) -> None:
    with st.expander("Détails — Tonnetz & Contrast spectral"):
        t1, t2 = st.columns(2)

        with t1:
            st.markdown("**Tonnetz (6 dimensions)**")
            tn_labels = ["5th","Minor 3rd","Major 3rd","5th im","m3 im","M3 im"]
            tn_vals = [track_feats.get(f"tonnetz_{i+1}_mean", 0.0) for i in range(6)]
            fig_tn = px.bar(
                x=tn_labels, y=tn_vals,
                labels={"x": "", "y": "Valeur moyenne"},
                color=tn_vals, color_continuous_scale="RdBu_r",
            )
            fig_tn.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_tn, use_container_width=True)

        with t2:
            st.markdown("**Contrast spectral (7 bandes)**")
            ct_labels = [f"Band {i+1}" for i in range(7)]
            ct_vals = [track_feats.get(f"contrast_{i+1}_mean", 0.0) for i in range(7)]
            fig_ct = px.bar(
                x=ct_labels, y=ct_vals,
                labels={"x": "", "y": "Contraste moyen"},
                color=ct_vals, color_continuous_scale="Viridis",
            )
            fig_ct.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_ct, use_container_width=True)


def render_track_analysis_section(d: dict) -> None:
    st.divider()
    st.header("Analyse ta propre piste")
    st.divider()
    st.write(
        "Upload un fichier MP3 ou colle une URL YouTube pour extraire les features audio "
        "et les comparer au dataset."
    )

    input_mode = st.radio(
        "Source",
        ["Fichier MP3", "URL YouTube", "Historique des musiques téléchargées"],
        horizontal=True,
        label_visibility="collapsed",
    )

    audio_path  = None
    music_name  = None
    track_feats = None

    if input_mode == "Fichier MP3":
        audio_path, music_name = _input_mp3()
    elif input_mode == "URL YouTube":
        audio_path, music_name = _input_youtube()
    elif input_mode == "Historique des musiques téléchargées":
        _render_history_section(d)
        return

    if audio_path:
        with st.spinner("Extraction des features audio…"):
            track_feats = _analyse_single_track(audio_path, music_name)
            if track_feats is not None:
                import librosa as _librosa
                _librosa.load(audio_path, sr=None, mono=True, duration=60)
                track_feats["duration"] = float(_librosa.get_duration(path=audio_path))

        if track_feats is not None:
            show_track_feats(track_feats, d["feature_stats"], d["stats"])
        else:
            st.error(
                "Impossible d'extraire les features. "
                "Vérifie que `librosa` est installé et que le fichier est valide."
            )

def _input_mp3():
    uploaded = st.file_uploader(
        "Dépose ton fichier audio",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
    )
    if not uploaded:
        return None, None
    suffix = Path(uploaded.name).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name
    st.success(f"Fichier chargé : `{uploaded.name}`")
    return path, uploaded.name


def _input_youtube():
    yt_url = st.text_input("URL YouTube", placeholder="https://www.youtube.com/watch?v=...")
    if not yt_url:
        return None, None

    with st.spinner("Téléchargement via yt-dlp…"):
        try:
            tmpdir = tempfile.mkdtemp()
            out_template = os.path.join(tmpdir, "track.%(ext)s")

            with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(yt_url, download=False)
                music_name = info.get("title", "Unknown Title")

            result = subprocess.run(
                [
                    "yt-dlp",
                    "--extract-audio", "--audio-format", "mp3",
                    "--audio-quality", "0",
                    "-o", out_template,
                    "--no-playlist", "--js-runtimes", "node",
                    yt_url,
                ],
                capture_output=True, text=True, timeout=120,
            )
            mp3_files = list(Path(tmpdir).glob("*.mp3"))
            if mp3_files:
                st.success("Audio téléchargé ✔")
                return str(mp3_files[0]), music_name
            st.error(
                f"yt-dlp n'a pas pu extraire l'audio.\n"
                f"stderr: {result.stderr[:400]}\n"
                "Vérifiez l'URL ou installez yt-dlp (`pip install yt-dlp`)."
            )
        except FileNotFoundError:
            st.error("yt-dlp introuvable — installez-le : `pip install yt-dlp`")
        except subprocess.TimeoutExpired:
            st.error("Timeout — vidéo trop longue ou connexion lente.")
    return None, None


def _render_history_section(d):
    st.header("Historique des musiques téléchargées")
    history_path = "data/features/history_features"
    spark = get_spark()

    if not os.path.exists(history_path):
        st.warning("Pas de fichier `history_features.parquet` trouvé.")
        return

    history_feats_df = spark.read.parquet(history_path)
    count = history_feats_df.count()

    if count == 0:
        st.warning("Aucune musique dans l'historique.")
        return

    st.success(f"{count} pistes chargées depuis l'historique.")
    track_names = history_feats_df.toPandas()["filename"].tolist()
    selected_track = st.selectbox("Choisis une piste à afficher", track_names, index=0)

    if st.button("Afficher les features"):
        track_feats = (
            history_feats_df
            .filter(F.col("filename") == selected_track)
            .withColumn("duration", F.lit(3005))
            .toPandas()
            .iloc[0]
            .to_dict()
        )
        show_track_feats(track_feats, d["feature_stats"], d["stats"])


def _analyse_single_track(path, music_name):
    spark = get_spark()
    abs_path = os.path.abspath(path)
    file_uri = f"file://{abs_path}"

    feat_df = (
        spark.createDataFrame([(file_uri,)], ["path"])
        .withColumn("f", extract_udf(F.col("path")))
        .select("f.*")
        .withColumn("filename", F.lit(music_name or "Unknown Track"))
    )

    feat_df.write.mode("append").parquet("data/features/history_features")

    result_df = feat_df.toPandas()
    if result_df.empty or int(result_df.iloc[0]["ok"]) != 1:
        st.error("L'UDF Spark a retourné ok=0 — vérifiez librosa.")
        return None

    result = result_df.iloc[0].to_dict()
    result.pop("ok", None)
    return result


def render_global_stats_section(d):
    st.header("Statistiques globales du dataset")
    st.divider()

    _render_active_filters_banner(
        d.get("_selected_users", []),
        d.get("_date_from"),
        d.get("_date_to"),
        d.get("_global_date_min"),
        d.get("_global_date_max"),
    )

    _render_kpi_row(d)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        _chart_genres(d["genres_df"])
    with col2:
        _chart_tempo(d["tempo_buckets"])

    col3, col4 = st.columns(2)
    with col3:
        _chart_centroid(d["stats"])
    with col4:
        _chart_rms(d["stats"])

    _chart_mfcc_radar(d["stats"])
    _chart_zcr_flatness(d["zcr_flatness"])
    _render_summary_table(d["stats"])


def _render_active_filters_banner(selected_users, date_from, date_to, global_date_min, global_date_max):
    if not date_from or not date_to:
        return
    parts = []
    if selected_users:
        parts.append(f"👤 **Utilisateurs :** {', '.join(str(u) for u in selected_users)}")
    if date_from != global_date_min or date_to != global_date_max:
        parts.append(f"📅 **Période :** {date_from} → {date_to}")
    if parts:
        st.info("  \n".join(parts))


def _render_kpi_row(d):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pistes totales",f"{d['total']:,}")
    c2.metric("Durée moyenne", f"{d['duration_avg']:.0f} s" if d["duration_avg"] else "—")
    c3.metric("Genres distincts", str(d["n_genres"]))
    c4.metric("Tempo moyen", f"{d['tempo_mean']} bpm")


def _chart_genres(genres_df):
    st.subheader("Genres les plus présents")
    st.write(
        "Répartition des pistes par genre. "
        "La courbe orange indique la **part cumulative** : "
        "les N premiers genres représentent X % du dataset "
        "(calculé avec `SUM OVER ORDER BY count DESC`)."
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=genres_df["genres"],
            y=genres_df["count"],
            text=genres_df["pct"].apply(lambda x: f"{x}%"),
            textposition="outside",
            marker_color=COLORS[: len(genres_df)],
            name="Pistes",
            showlegend=False,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=genres_df["genres"],
            y=genres_df["cumul_pct"],
            mode="lines+markers+text",
            text=genres_df["cumul_pct"].apply(lambda x: f"{x}%"),
            textposition="top center",
            line=dict(color="#FF7F0E", width=2, dash="dot"),
            marker=dict(size=7),
            name="Cumul %",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Nombre de pistes", secondary_y=False)
    fig.update_yaxes(title_text="Part cumulative (%)", range=[0, 115],
                     secondary_y=True, showgrid=False)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        xaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_tempo(tempo_df):
    st.subheader("Distribution des tempos")
    st.write(
        "Nombre de pistes par tranche de BPM. "
        "La courbe bleue montre le **cumul progressif** : "
        "à partir de quelle tranche on a couvert 50 %, 80 %… du dataset "
        "(window `SUM ROWS UNBOUNDED PRECEDING`)."
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=tempo_df["bucket"].astype(str),
            y=tempo_df["count"],
            marker_color="#3266ad",
            name="Pistes",
            showlegend=False,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=tempo_df["bucket"].astype(str),
            y=tempo_df["cumul_pct"],
            mode="lines+markers+text",
            text=tempo_df["cumul_pct"].apply(lambda x: f"{x}%"),
            textposition="top right",
            line=dict(color="#17BECF", width=2),
            marker=dict(size=7),
            name="Cumul %",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Nombre de pistes", secondary_y=False)
    fig.update_yaxes(title_text="Cumul (%)", range=[0, 115],
                     secondary_y=True, showgrid=False)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_centroid(stats):
    st.subheader("Brillance spectrale par genre")
    st.write(
        "Centroid spectral moyen par genre. "
        "La couleur indique le **z-score** par rapport à la moyenne de tous les genres "
        "(window `AVG / STDDEV OVER ()`) : "
        "🔴 rouge = plus brillant que la moyenne, 🔵 bleu = plus sombre."
    )

    fig = px.bar(
        stats,
        x="genres", y="centroid",
        color="z_centroid",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        labels={"centroid": "Hz", "genres": "", "z_centroid": "Z-score"},
        text=stats["z_centroid"].apply(lambda z: f"z={z:+.2f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_colorbar=dict(title="Z-score", thickness=12),
        margin=dict(l=0, r=0, t=10, b=0),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_rms(stats):
    st.subheader("Énergie RMS par genre")
    st.write(
        "RMS moyen par genre. "
        "La couleur indique le **z-score** par rapport à la moyenne de tous les genres "
        "(window `AVG / STDDEV OVER ()`) : "
        "🔴 rouge = plus fort que la moyenne, 🔵 bleu = plus doux."
    )

    fig = px.bar(
        stats,
        x="genres", y="rms",
        color="z_rms",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        labels={"rms": "RMS", "genres": "", "z_rms": "Z-score"},
        text=stats["z_rms"].apply(lambda z: f"z={z:+.2f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_colorbar=dict(title="Z-score", thickness=12),
        margin=dict(l=0, r=0, t=10, b=0),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_mfcc_radar(stats):
    st.subheader("Profil timbral moyen par genre (MFCC 1–5)")
    st.write(
        "Les **MFCC** capturent la forme du spectre sonore telle que l'oreille humaine la perçoit. "
        "Des profils radar très différents entre genres signifient que ces features sont bien discriminantes."
    )
    top6 = stats.head(6)
    mfcc_labels = ["MFCC 1","MFCC 2","MFCC 3","MFCC 4","MFCC 5","MFCC 1"]
    fig_radar = go.Figure()

    for i, row in top6.iterrows():
        vals = [row["mfcc1"], row["mfcc2"], row["mfcc3"],
                row["mfcc4"], row["mfcc5"]] + [row["mfcc1"]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=mfcc_labels,
            fill="toself", opacity=0.55, name=row["genres"],
            line=dict(color=COLORS[i % len(COLORS)]),
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=40, r=40, t=20, b=60),
        height=440,
    )
    st.plotly_chart(fig_radar, use_container_width=True)


def _chart_zcr_flatness(zcr_flatness_df):
    st.subheader("Complexité harmonique — ZCR vs Spectral Flatness")
    st.write(
        "Chaque bulle est un genre. Sa couleur reflète un **score de bruit combiné** "
        "calculé par `PERCENT_RANK() OVER` sur ZCR et Flatness séparément, puis moyenné. "
        "🔴 rouge = genre bruité/percussif, 🔵 bleu = genre tonal/mélodique. "
        "Le tooltip affiche le percentile exact sur chaque axe."
    )

    fig = px.scatter(
        zcr_flatness_df,
        x="zcr", y="flatness",
        text="genres",
        size="count", size_max=45,
        color="noise_score",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0.5,
        labels={
            "zcr":         "ZCR moyen",
            "flatness":    "Flatness moyen",
            "noise_score": "Score bruit",
        },
        hover_data={
            "pct_rank_zcr":      ":.0%",
            "pct_rank_flatness": ":.0%",
            "noise_score":       ":.0%",
            "count":             True,
        },
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        coloraxis_colorbar=dict(title="Score bruit", thickness=12),
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_summary_table(stats):
    with st.expander("Tableau récapitulatif par genre"):
        cols = ["genres","count","tempo","centroid","rms","zcr","flatness",
                "z_centroid","z_rms"]
        renamed = {"genres": "Genre", "count": "Pistes", "tempo": "Tempo (bpm)", "centroid": "Centroid (Hz)", "rms": "RMS", "zcr": "ZCR", "flatness": "Flatness", "z_centroid": "Z Centroid", "z_rms": "Z RMS",}
        st.dataframe(
            stats[cols].rename(columns=renamed).set_index("Genre"),
            use_container_width=True,
        )


def main():
    st.set_page_config(page_title="Music Features Dashboard", layout="wide")
    st.title("Music Features Dashboard")
    st.caption("Analyse des features ML extraites par Spark")

    with st.spinner("Chargement des données Spark..."):
        df_raw, meta = load_raw()

    d = render_sidebar_filters(df_raw, meta)
    render_track_analysis_section(d)
    render_global_stats_section(d)

if __name__ == "__main__":
    main()