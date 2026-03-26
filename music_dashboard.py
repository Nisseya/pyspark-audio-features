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
    os.environ["PYSPARK_PYTHON"] = r"C:\spark-env\Scripts\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\spark-env\Scripts\python.exe"
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

import warnings

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from yt_dlp import YoutubeDL

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.extract_all_concise import extract_udf

warnings.filterwarnings("ignore")

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Music Features Dashboard", layout="wide")
st.title("Music Features Dashboard")
st.caption("Analyse des features ML extraites par Spark")

# ── Session Spark (cached) ────────────────────────────────────────────────────
@st.cache_resource
def get_spark():
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

@st.cache_data
def load_data():
    spark = get_spark()
    out_path = "data/features/training_dataset"
    df = spark.read.parquet(out_path)

    total_tracks = df.count()

    # Durée moyenne (colonne optionnelle) — on l'intègre dans le même agg
    has_duration = "duration" in df.columns
    agg_base = [
        F.round(F.mean("tempo"), 1).alias("tempo_mean"),
        F.countDistinct("genres").alias("n_genres"),
    ]
    if has_duration:
        agg_base.append(F.round(F.mean("duration"), 1).alias("duration_avg"))

    global_stats_pdf = df.agg(*agg_base).toPandas()
    global_stats = global_stats_pdf.iloc[0]

    duration_avg = float(global_stats["duration_avg"] or 0) if has_duration else None

    df_genres = df.withColumn("genres", F.explode(F.col("genres")))

    genres_df = (
        df_genres.groupBy("genres").count()
        .orderBy("count", ascending=False)
        .limit(12)
        .toPandas()
    )
    genres_df["pct"] = (genres_df["count"] / total_tracks * 100).round(1)

    df = df_genres

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
        .toPandas()
    )
    bucket_order = ["< 80","80-100","100-110","110-120","120-130","130-140","140-160","160+"]
    tempo_buckets["bucket"] = pd.Categorical(
        tempo_buckets["bucket"], categories=bucket_order, ordered=True
    )
    tempo_buckets = tempo_buckets.sort_values("bucket")

    stats_by_genre = (
        df.groupBy("genres").agg(
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
        )
        .orderBy("count", ascending=False)
        .limit(12)
        .toPandas()
        .fillna(0)
    )

    # Statistiques globales par feature pour la comparaison (min, max, mean, std)
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
    global_feature_stats = df.agg(*agg_exprs).toPandas().iloc[0].to_dict()

    return {
        "total": total_tracks,
        "tempo_mean": float(global_stats["tempo_mean"] or 0),
        "n_genres": int(global_stats["n_genres"] or 0),
        "duration_avg": duration_avg,
        "genres_df": genres_df,
        "tempo_buckets": tempo_buckets,
        "stats": stats_by_genre,
        "feature_stats": global_feature_stats,
    }

# ── Chargement ────────────────────────────────────────────────────────────────
with st.spinner("Chargement des données Spark..."):
    d = load_data()

genres_df = d["genres_df"]
tempo_df  = d["tempo_buckets"]
stats     = d["stats"]
feat_stats = d["feature_stats"]

COLORS = px.colors.qualitative.D3

# ════════════════════════════════════════════════════════════════════════════════
# ── SECTION : Analyse ta propre piste ────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

def show_track_feats(track_feats):
    st.success("Features extraites ✔")

    # ── KPIs de la piste ──────────────────────────────────────────────────
    st.markdown("#### 📊 Features de ta piste")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tempo",     f"{track_feats['tempo']:.1f} bpm")
    k2.metric("Durée",     f"{track_feats['duration']:.0f} s")
    k3.metric("Centroid",  f"{track_feats['centroid_mean']:.0f} Hz")
    k4.metric("RMS",       f"{track_feats['rms_mean']:.4f}")
    k5.metric("ZCR",       f"{track_feats['zcr_mean']:.4f}")

    # ── Barres contextuelles : ta piste vs dataset (bullet charts) ────────
    st.markdown("#### 🔍 Ta piste dans le contexte du dataset")
    st.write(
        "Chaque bullet chart positionne ta valeur (barre rouge) "
        "dans la distribution du dataset : gris foncé = P10–P90, gris clair = min–max."
    )

    compare_cfg = [
        ("Tempo (bpm)",   "tempo",          "{:.1f} bpm"),
        ("Centroid (Hz)", "centroid_mean",  "{:.0f} Hz"),
        ("RMS",           "rms_mean",       "{:.4f}"),
        ("ZCR",           "zcr_mean",       "{:.4f}"),
        ("Flatness",      "flatness_mean",  "{:.5f}"),
    ]

    for i in range(0, len(compare_cfg), 2):
        cols = st.columns(2)

        for j, (label, feat_key, fmt) in enumerate(compare_cfg[i:i+2]):
            track_val = track_feats.get(feat_key, 0.0)
            ds_mean   = feat_stats.get(f"{feat_key}__mean", 0.0) or 0.0
            ds_std    = feat_stats.get(f"{feat_key}__std", 0.0)  or 0.0
            ds_min    = feat_stats.get(f"{feat_key}__min", 0.0)  or 0.0
            ds_max    = feat_stats.get(f"{feat_key}__max", 0.0)  or 1.0
            ds_p10    = feat_stats.get(f"{feat_key}__p10", ds_min) or ds_min
            ds_p90    = feat_stats.get(f"{feat_key}__p90", ds_max) or ds_max

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

            fig_b.update_layout(
                height=110,
                margin=dict(l=10, r=10, t=30, b=10),
            )

            with cols[j]:
                st.markdown(f"**{label}**")
                st.plotly_chart(fig_b, use_container_width=True)
                delta = track_val - ds_mean
                delta_pct = delta / ds_mean * 100 if ds_mean else 0
                sign = "+" if delta_pct >= 0 else ""
                color = "🔴" if abs(delta_pct) > 30 else "🟡" if abs(delta_pct) > 10 else "🟢"
                st.caption(f"{color} {sign}{delta_pct:.1f}% vs moyenne dataset")

    # ── Radar MFCC : ta piste vs genres du dataset ────────────────────────
    st.markdown("#### 🕸️ Profil timbral — ta piste vs genres du dataset")
    st.write(
        "Ton profil MFCC 1–5 (rouge, premier plan) superposé aux "
        "6 genres les plus représentés du dataset."
    )

    top6 = stats.head(6)
    fig_cmp = go.Figure()
    mfcc_labels = ["MFCC 1","MFCC 2","MFCC 3","MFCC 4","MFCC 5","MFCC 1"]

    for i, row in top6.iterrows():
        vals = [row["mfcc1"], row["mfcc2"], row["mfcc3"], row["mfcc4"], row["mfcc5"]] + [row["mfcc1"]]
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

    # ── Chromagramme polaire ───────────────────────────────────────────────
    st.markdown("#### 🎹 Empreinte chromatique de ta piste")
    st.write(
        "Le chromagramme moyen indique quelles notes (C → B) dominent dans ta piste. "
        "Utile pour identifier la tonalité et les couleurs harmoniques."
    )
    notes       = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    chroma_vals = [track_feats.get(f"chroma_{i+1}_mean", 0.0) for i in range(12)]
    fig_chroma  = px.bar_polar(
        r=chroma_vals, theta=notes,
        color=chroma_vals,
        color_continuous_scale="Plasma",
    )
    fig_chroma.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_chroma, use_container_width=True)

    # ── Tonnetz : relations harmoniques ───────────────────────────────────
    with st.expander("Détails — Tonnetz & Contrast spectral"):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Tonnetz (6 dimensions)**")
            tn_labels = ["5th","Minor 3rd","Major 3rd","5th im","m3 im","M3 im"]
            tn_vals   = [track_feats.get(f"tonnetz_{i+1}_mean", 0.0) for i in range(6)]
            fig_tn = px.bar(x=tn_labels, y=tn_vals,
                            labels={"x": "", "y": "Valeur moyenne"},
                            color=tn_vals, color_continuous_scale="RdBu_r")
            fig_tn.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_tn, use_container_width=True)
        with t2:
            st.markdown("**Contrast spectral (7 bandes)**")
            ct_labels = [f"Band {i+1}" for i in range(7)]
            ct_vals   = [track_feats.get(f"contrast_{i+1}_mean", 0.0) for i in range(7)]
            fig_ct = px.bar(x=ct_labels, y=ct_vals,
                            labels={"x": "", "y": "Contraste moyen"},
                            color=ct_vals, color_continuous_scale="Viridis")
            fig_ct.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_ct, use_container_width=True)

    st.divider()

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

audio_path_for_analysis = None
music_name = None
track_feats = None

if input_mode == "Fichier MP3":
    uploaded = st.file_uploader(
        "Dépose ton fichier audio",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
    )
    if uploaded:
        suffix = Path(uploaded.name).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            audio_path_for_analysis = tmp.name
        music_name = uploaded.name
        st.success(f"Fichier chargé : `{uploaded.name}`")

elif input_mode == "URL YouTube":
    yt_url = st.text_input("URL YouTube", placeholder="https://www.youtube.com/watch?v=...")
    if yt_url:
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
                        "--extract-audio",
                        "--audio-format", "mp3",
                        "--audio-quality", "0",
                        "-o", out_template,
                        "--no-playlist",
                        "--js-runtimes", "node",
                        yt_url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                mp3_files = list(Path(tmpdir).glob("*.mp3"))
                if mp3_files:
                    audio_path_for_analysis = str(mp3_files[0])
                    st.success("Audio téléchargé ✔")
                else:
                    st.error(
                        f"yt-dlp n'a pas pu extraire l'audio.\n"
                        f"stderr: {result.stderr[:400]}\n"
                        " Vérifiez l'URL ou installez yt-dlp (`pip install yt-dlp`)."
                    )
            except FileNotFoundError:
                st.error("yt-dlp introuvable — installez-le : `pip install yt-dlp`")
            except subprocess.TimeoutExpired:
                st.error("Timeout — vidéo trop longue ou connexion lente.")

elif input_mode == "Historique des musiques téléchargées":
    st.header("🎵 Historique des musiques téléchargées")
    history_path = "data/features/history_features"
    spark = get_spark()

    if os.path.exists(history_path):
        history_feats_df = spark.read.parquet(history_path)
        if not history_feats_df.head(1):
            st.warning("Aucune musique dans l'historique.")
        else:
            st.success(f"{history_feats_df.count()} pistes chargées depuis l'historique.")
    else:
        st.warning("Pas de fichier `history_features.parquet` trouvé.")
    
    if history_feats_df is not None and history_feats_df.head(1):
        track_names = history_feats_df.toPandas()["filename"].tolist()
        selected_track_name = st.selectbox(
            "Choisis une piste à afficher",
            track_names,
            index=0,
        )
        
        if st.button("Afficher les features"):
            track_feats_row = history_feats_df.filter(
                F.col("filename") == selected_track_name
            ).withColumn("duration", F.lit(3005)).first().asDict()

            show_track_feats(track_feats_row)

def analyse_single_track(path: str):
    spark = get_spark()

    abs_path = os.path.abspath(path)
    file_uri = f"file://{abs_path}"

    path_df = spark.createDataFrame([(file_uri,)], ["path"])

    feat_df = (
        path_df
        .withColumn("f", extract_udf(F.col("path")))
        .select("f.*")
    )
    
    feat_df = feat_df.withColumn("filename", F.lit(music_name or "Unknown Track"))
    output_path = "data/features/history_features"
    feat_df.write.mode("append").parquet(output_path)

    row = feat_df.first()

    if row is None or int(row["ok"]) != 1:
        st.error("L'UDF Spark a retourné ok=0 — vérifiez librosa.")
        return None

    result = row.asDict()
    result.pop("ok", None)
    return result

if audio_path_for_analysis:
    with st.spinner("Extraction des features audio…"):
        track_feats = analyse_single_track(audio_path_for_analysis)
        
        # TODO : Ajouter la duration dans l'UDF pour éviter de recharger le fichier avec librosa ici
        if track_feats is not None:
            import librosa as _librosa
            _, _ = _librosa.load(audio_path_for_analysis, sr=None, mono=True, duration=60)
            track_feats["duration"] = float(
                _librosa.get_duration(path=audio_path_for_analysis)
            )
            
    if track_feats is not None:
        show_track_feats(track_feats)
    else:
        st.error("Impossible d'extraire les features. Vérifie que `librosa` est installé et que le fichier est valide.")


# ════════════════════════════════════════════════════════════════════════════════
# ── Métriques globales du dataset ─────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

st.header("Statistiques globales du dataset")
st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pistes totales",   f"{d['total']:,}")
c2.metric("Durée moyenne",    f"{d['duration_avg']:.0f} s" if d["duration_avg"] else "—")
c3.metric("Genres distincts", str(d["n_genres"]))
c4.metric("Tempo moyen",      f"{d['tempo_mean']} bpm")

st.divider()

# ── Ligne 1 : genres + tempo ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Genres les plus présents")
    st.write(
        "Répartition des pistes par genre dans le dataset. "
        "Un déséquilibre marqué entre genres peut biaiser un modèle de classification — "
        "à surveiller avant l'entraînement."
    )
    fig = px.bar(
        genres_df,
        x="count", y="genres", orientation="h",
        text=genres_df["pct"].apply(lambda x: f"{x}%"),
        color="genres", color_discrete_sequence=COLORS,
    )
    fig.update_layout(
        showlegend=False, yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=10, b=0), height=380,
        xaxis_title="Nombre de pistes", yaxis_title="",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Distribution des tempos")
    st.write(
        "Nombre de pistes par tranche de BPM. "
        "Un pic autour de 120–130 BPM est typique des datasets pop/electronic. "
        "Le tempo est l'une des features les plus discriminantes pour séparer "
        "genres lents (jazz, classique) et rapides (metal, drum & bass)."
    )
    fig = px.bar(
        tempo_df, x="bucket", y="count",
        color_discrete_sequence=["#3266ad"],
        labels={"bucket": "BPM", "count": "Pistes"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
    st.plotly_chart(fig, use_container_width=True)

# ── Ligne 2 : centroid + rms ──────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Brillance spectrale par genre")
    st.write(
        "Le **centroid spectral** est la moyenne pondérée des fréquences présentes dans le signal (en Hz). "
        "Un centroid élevé = son aigu et brillant (cymbales, voix criarde). "
        "Un centroid faible = son grave et chaud (basse, contrebasse). "
        "Feature clé pour distinguer metal et classique."
    )
    fig = px.bar(
        stats, x="genres", y="centroid",
        color="genres", color_discrete_sequence=COLORS,
        labels={"centroid": "Hz", "genres": ""},
    )
    fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0), height=340)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Énergie RMS par genre")
    st.write(
        "Le **RMS (Root Mean Square)** mesure l'énergie moyenne du signal — il correspond au volume perçu. "
        "Metal et electronic sont typiquement les plus forts (compression forte, peu de dynamique). "
        "Classique et jazz ont un RMS plus faible car ils exploitent toute la plage dynamique."
    )
    fig = px.bar(
        stats, x="genres", y="rms",
        color="genres", color_discrete_sequence=COLORS,
        labels={"rms": "RMS", "genres": ""},
    )
    fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0), height=340)
    st.plotly_chart(fig, use_container_width=True)

# ── Radar MFCC ────────────────────────────────────────────────────────────────
st.subheader("Profil timbral moyen par genre (MFCC 1–5)")
st.write(
    "Les **MFCC (Mel-Frequency Cepstral Coefficients)** capturent la forme du spectre sonore "
    "telle que l'oreille humaine la perçoit. Les premiers coefficients décrivent l'enveloppe globale du timbre. "
    "Des profils radar très différents entre genres signifient que ces features sont bien discriminantes — "
    "bon signe pour la classification."
)

top6 = stats.head(6)
fig_radar = go.Figure()
for i, row in top6.iterrows():
    vals = [row["mfcc1"], row["mfcc2"], row["mfcc3"], row["mfcc4"], row["mfcc5"]]
    vals += vals[:1]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=["MFCC 1","MFCC 2","MFCC 3","MFCC 4","MFCC 5","MFCC 1"],
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

# ── Scatter ZCR vs Flatness ───────────────────────────────────────────────────
st.subheader("Complexité harmonique — ZCR vs Spectral Flatness")
st.write(
    "Le **ZCR (Zero-Crossing Rate)** compte combien de fois par seconde le signal passe par zéro — "
    "élevé pour les sons percussifs et bruités, faible pour les sons tonals. "
    "La **Spectral Flatness** mesure si le spectre ressemble à du bruit blanc (valeur proche de 1) "
    "ou à un son avec des harmoniques bien définies (valeur proche de 0). "
    "Les genres en haut à droite sont bruités et peu structurés, ceux en bas à gauche sont tonals et mélodiques. "
    "La taille des bulles représente le nombre de pistes."
)

fig_s = px.scatter(
    stats,
    x="zcr", y="flatness", text="genres",
    size="count", size_max=45,
    color="genres", color_discrete_sequence=COLORS,
    labels={"zcr": "ZCR moyen", "flatness": "Flatness moyen"},
)
fig_s.update_traces(textposition="top center")
fig_s.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0), height=420)
st.plotly_chart(fig_s, use_container_width=True)

# ── Tableau récap ─────────────────────────────────────────────────────────────
with st.expander("Tableau récapitulatif par genre"):
    cols = ["genres","count","tempo","centroid","rms","zcr","flatness"]
    renamed = {
        "genres": "Genre", "count": "Pistes", "tempo": "Tempo (bpm)",
        "centroid": "Centroid (Hz)", "rms": "RMS", "zcr": "ZCR", "flatness": "Flatness",
    }
    st.dataframe(
        stats[cols].rename(columns=renamed).set_index("Genre"),
        use_container_width=True,
    )