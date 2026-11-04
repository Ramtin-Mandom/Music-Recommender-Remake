"""Interactive test bench for the recommendation engine.

Run with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from music_recommender import LEGACY_CONFIG, EngineConfig, Filters, Recommender  # noqa: E402
from music_recommender.features import GROUP_LABELS  # noqa: E402

MODEL_PATH = ROOT / "artifacts" / "model.joblib"
RADAR_FEATURES = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness", "speechiness"]
TABLE_COLS = ["rank", "name", "artists", "year", "popularity", "similarity", "why"]

st.set_page_config(page_title="Music Recommender Remake", layout="wide")


# ----------------------------------------------------------------------
# Engine loading (cached across reruns and sessions)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading 170k tracks and fitting the engine...")
def get_engine() -> Recommender:
    if MODEL_PATH.exists():
        try:
            return Recommender.load(MODEL_PATH)
        except Exception:  # stale or incompatible file -> refit
            pass
    return Recommender.from_csv(config=EngineConfig())


@st.cache_resource(show_spinner="Fitting the legacy (CMPT 310) model...")
def get_legacy_engine() -> Recommender:
    return Recommender(LEGACY_CONFIG).fit(get_engine().tracks)


engine = get_engine()
ss = st.session_state
ss.setdefault("liked", [])  # list of (row, weight)
ss.setdefault("disliked", [])
ss.setdefault("results", {})  # source tab -> (recs, seed rows), for the Explain tab


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def track_label(row: int) -> str:
    t = engine.tracks.iloc[row]
    return f"{t['name']} — {t['artists_display']} ({t['year']})"


def track_picker(key: str, label: str = "Search for a song") -> int | None:
    """Search box + selectbox of ranked matches; returns the chosen row."""
    c1, c2 = st.columns([3, 2])
    query = c1.text_input(label, key=f"{key}_q", placeholder="e.g. Bohemian Rhapsody")
    artist = c2.text_input("Artist (optional)", key=f"{key}_a", placeholder="e.g. Queen")
    if not query.strip():
        return None
    hits = engine.search(query, artist or None, limit=15)
    if hits.empty:
        suggestions = engine.catalog.suggest(query)
        st.warning("No match." + (f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""))
        return None
    rows = hits["row"].astype(int).tolist()
    return st.selectbox("Matches (best first)", rows, format_func=track_label, key=f"{key}_sel")


def overrides() -> dict:
    return {
        "diversity": ss.diversity,
        "popularity_weight": ss.popularity_weight,
        "strategy": ss.strategy,
        "max_per_artist": ss.max_per_artist,
    }


def current_filters() -> Filters:
    y0, y1 = ss.years
    lo, hi = int(engine.tracks["year"].min()), int(engine.tracks["year"].max())
    return Filters(
        year_min=y0 if y0 > lo else None,
        year_max=y1 if y1 < hi else None,
        min_popularity=ss.min_pop or None,
        exclude_explicit=ss.no_explicit,
    )


def show_recs(recs: pd.DataFrame, key: str) -> None:
    if recs.empty:
        st.info("No tracks satisfy the current filters.")
        return
    st.dataframe(
        recs[TABLE_COLS],
        hide_index=True,
        width="stretch",
        column_config={
            "similarity": st.column_config.ProgressColumn("similarity", min_value=0.0, max_value=1.0, format="%.2f"),
            "popularity": st.column_config.NumberColumn("popularity", format="%d"),
            "year": st.column_config.NumberColumn("year", format="%d"),
            "why": st.column_config.TextColumn("closest on"),
        },
    )
    st.download_button(
        "Download CSV",
        recs.drop(columns=["row"]).to_csv(index=False),
        file_name=f"{key}_recommendations.csv",
        mime="text/csv",
        key=f"{key}_dl",
    )


def radar(seed_rows: list[int], rec_rows: list[int]) -> go.Figure:
    fig = go.Figure()
    for label, rows, colour in (("Seeds", seed_rows, "#1f77b4"), ("Recommendations", rec_rows, "#ff7f0e")):
        if not rows:
            continue
        vals = engine.tracks.iloc[rows][RADAR_FEATURES].mean().tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=vals + vals[:1],
                theta=RADAR_FEATURES + RADAR_FEATURES[:1],
                fill="toself",
                name=label,
                line_color=colour,
                opacity=0.6,
            )
        )
    fig.update_layout(
        polar={"radialaxis": {"range": [0, 1]}},
        height=380,
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        legend={"orientation": "h"},
    )
    return fig


def remember(source: str, recs: pd.DataFrame, seeds: list[int]) -> None:
    """Keep each tab's latest results separately (Streamlit runs every tab on each rerun)."""
    if recs.empty or not seeds:
        ss.results.pop(source, None)
    else:
        ss.results[source] = (recs, seeds)


# ----------------------------------------------------------------------
# Sidebar: engine knobs and filters
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Engine")
    st.caption(f"{len(engine.tracks):,} tracks · {len(engine.pipeline.columns)}-d feature space")
    st.number_input("Results (k)", 1, 50, 10, key="k")
    st.slider(
        "Diversity (MMR)",
        0.0,
        1.0,
        engine.config.diversity,
        0.05,
        key="diversity",
        help="0 = most similar tracks only; higher values spread the playlist out.",
    )
    st.slider(
        "Popularity prior",
        0.0,
        0.5,
        engine.config.popularity_weight,
        0.01,
        key="popularity_weight",
        help="Blend factor nudging better-known tracks up.",
    )
    st.selectbox(
        "Multi-seed strategy",
        ["hybrid", "centroid", "nearest_seed"],
        key="strategy",
        help="How several liked songs are combined (see docs/ALGORITHM.md).",
    )
    st.number_input("Max tracks per artist", 1, 10, engine.config.max_per_artist, key="max_per_artist")
    st.header("Filters")
    lo, hi = int(engine.tracks["year"].min()), int(engine.tracks["year"].max())
    st.slider("Release years", lo, hi, (lo, hi), key="years")
    st.slider("Minimum popularity", 0, 100, 0, key="min_pop")
    st.checkbox("Exclude explicit tracks", key="no_explicit")
    st.header("Compare")
    st.checkbox("Show the legacy CMPT 310 model side by side", key="compare")

st.title("Music Recommender Remake")
st.caption("Content-based k-NN recommendations over Spotify audio features — test bench")

tab_radio, tab_playlist, tab_mood, tab_explain, tab_eval, tab_data = st.tabs(
    ["Song radio", "Playlist builder", "Mood explorer", "Explain", "Evaluate", "Dataset"]
)

# ----------------------------------------------------------------------
with tab_radio:
    st.subheader("Songs that sound like one track")
    seed = track_picker("radio")
    if seed is not None:
        recs = engine.similar_to(seed, k=ss.k, filters=current_filters(), **overrides())
        remember("Song radio", recs, [seed])
        if ss.compare:
            left, right = st.columns(2)
            with left:
                st.markdown("**Remake engine**")
                show_recs(recs, "radio")
            with right:
                st.markdown("**Legacy CMPT 310 model**")
                legacy = get_legacy_engine().similar_to(seed, k=ss.k, filters=current_filters())
                show_recs(legacy, "radio_legacy")
        else:
            show_recs(recs, "radio")
            with st.expander("Sound profile: seeds vs. recommendations", expanded=True):
                st.plotly_chart(radar([seed], recs["row"].astype(int).tolist()), width="stretch")

# ----------------------------------------------------------------------
with tab_playlist:
    st.subheader("Build a playlist from songs you like (and don't)")
    row = track_picker("pl", "Add a song")
    c1, c2, c3 = st.columns([1, 1, 2])
    weight = c3.slider("Weight of this liked song", 0.25, 3.0, 1.0, 0.25, key="pl_w")
    if row is not None and c1.button("Add to liked", type="primary"):
        if row not in [r for r, _ in ss.liked]:
            ss.liked.append((row, weight))
    if row is not None and c2.button("Add to disliked"):
        if row not in ss.disliked:
            ss.disliked.append(row)

    lc, dc = st.columns(2)
    with lc:
        st.markdown(f"**Liked ({len(ss.liked)})**")
        for i, (r, w) in enumerate(list(ss.liked)):
            a, b = st.columns([4, 1])
            a.write(f"{track_label(r)} · weight {w:g}")
            if b.button("Remove", key=f"rm_like_{i}"):
                ss.liked.pop(i)
                st.rerun()
    with dc:
        st.markdown(f"**Disliked ({len(ss.disliked)})**")
        for i, r in enumerate(list(ss.disliked)):
            a, b = st.columns([4, 1])
            a.write(track_label(r))
            if b.button("Remove", key=f"rm_dis_{i}"):
                ss.disliked.pop(i)
                st.rerun()

    if ss.liked:
        rows = [r for r, _ in ss.liked]
        recs = engine.recommend_rows(
            rows,
            k=ss.k,
            seed_weights=[w for _, w in ss.liked],
            disliked_rows=ss.disliked,
            filters=current_filters(),
            **overrides(),
        )
        remember("Playlist builder", recs, rows)
        st.markdown("---")
        if ss.compare:
            left, right = st.columns(2)
            with left:
                st.markdown("**Remake engine**")
                show_recs(recs, "playlist")
            with right:
                st.markdown("**Legacy CMPT 310 model** (plain average, no dislikes)")
                show_recs(
                    get_legacy_engine().recommend_rows(rows, k=ss.k, filters=current_filters()), "playlist_legacy"
                )
        else:
            show_recs(recs, "playlist")
            with st.expander("Sound profile: seeds vs. recommendations", expanded=True):
                st.plotly_chart(radar(rows, recs["row"].astype(int).tolist()), width="stretch")
    else:
        ss.results.pop("Playlist builder", None)
        st.info("Add at least one liked song to get recommendations.")

# ----------------------------------------------------------------------
with tab_mood:
    st.subheader("Describe the sound you want")
    st.caption("Tick the attributes you care about; unticked ones are ignored in the distance.")
    targets: dict[str, float] = {}
    spec = [
        ("energy", 0.0, 1.0, 0.8, 0.01),
        ("danceability", 0.0, 1.0, 0.7, 0.01),
        ("valence", 0.0, 1.0, 0.7, 0.01),
        ("acousticness", 0.0, 1.0, 0.2, 0.01),
        ("instrumentalness", 0.0, 1.0, 0.0, 0.01),
        ("speechiness", 0.0, 1.0, 0.05, 0.01),
        ("tempo", 40, 220, 120, 1),
        ("loudness", -40.0, 0.0, -6.0, 0.5),
        ("year", 1921, 2020, 2010, 1),
    ]
    cols = st.columns(3)
    for i, (name, lo_, hi_, default, step) in enumerate(spec):
        with cols[i % 3]:
            use = st.checkbox(name, value=name in ("energy", "danceability", "valence"), key=f"use_{name}")
            val = st.slider(
                name, lo_, hi_, default, step, key=f"mood_{name}", disabled=not use, label_visibility="collapsed"
            )
            if use:
                targets[name] = val
    if targets:
        recs = engine.from_targets(targets, k=ss.k, filters=current_filters(), **overrides())
        show_recs(recs, "mood")
    else:
        st.info("Select at least one attribute.")

# ----------------------------------------------------------------------
with tab_explain:
    st.subheader("Why was a track recommended?")
    if not ss.results:
        st.info("Generate recommendations in *Song radio* or *Playlist builder* first.")
    else:
        source = st.radio("Results from", list(ss.results), horizontal=True)
        recs, seeds = ss.results[source]
        pick = st.selectbox("Recommendation", recs["row"].astype(int).tolist(), format_func=track_label)
        exp = engine.explain(seeds, pick)
        st.caption(
            "**share** = fraction of the remaining distance each feature accounts for (what differs most); "
            "**closeness** = 0-1 similarity on that feature alone."
        )
        fig = go.Figure(go.Bar(x=exp["share"], y=exp["feature"], orientation="h", marker_color="#ff7f0e"))
        fig.update_layout(
            height=420,
            yaxis={"autorange": "reversed"},
            xaxis={"title": "share of distance"},
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(fig, width="stretch")
        with st.expander("Per-feature table"):
            st.dataframe(exp[["feature", "weight", "closeness", "share"]], hide_index=True, width="stretch")
        st.markdown("**Seeds vs. this recommendation**")
        st.dataframe(engine.describe([*seeds, pick]).drop(columns=["id"]), hide_index=True, width="stretch")

# ----------------------------------------------------------------------
with tab_eval:
    st.subheader("Offline evaluation")
    st.markdown(
        "Benchmarks the legacy model, the remake and single-change ablations on held-out artists. "
        "**precision@k** = share of the top-k that are *other songs by the seed's artist* "
        "(artist is not a model input). See `docs/EVALUATION.md`."
    )
    report = ROOT / "reports" / "benchmark.csv"
    n_q = st.slider("Queries per task", 50, 500, 150, 50)
    if st.button("Run benchmark", type="primary"):
        from music_recommender.evaluation import benchmark_configs, run_benchmark

        log = st.empty()
        lines: list[str] = []

        def _log(msg: str) -> None:
            lines.append(msg)
            log.code("\n".join(lines))

        with st.spinner("Evaluating..."):
            table = run_benchmark(engine.tracks, benchmark_configs(engine.config), n_queries=n_q, log=_log)
        ss.bench = table
    table = ss.get("bench")
    if table is None and report.exists():
        table = pd.read_csv(report)
        st.caption(f"Showing saved results from `{report.relative_to(ROOT).as_posix()}`.")
    if table is not None:
        fig = go.Figure()
        for metric in ["radio_precision@k", "playlist_precision@k"]:
            fig.add_trace(go.Bar(name=metric, x=table["engine"], y=table[metric]))
        fig.update_layout(barmode="group", height=380, margin={"t": 30})
        st.plotly_chart(fig, width="stretch")
        st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("Original CMPT 310 results (for reference)"):
        legacy_dir = ROOT / "data" / "legacy"
        st.markdown("5-fold CV bucket-agreement results of the original project:")
        st.dataframe(pd.read_csv(legacy_dir / "cv_optimization_results.csv"), hide_index=True)
        st.markdown("User study (scores out of 10, participants anonymised):")
        st.dataframe(pd.read_csv(legacy_dir / "user_study_scores.csv"), hide_index=True)

# ----------------------------------------------------------------------
with tab_data:
    st.subheader("Catalogue")
    t = engine.tracks
    c1, c2 = st.columns(2)
    c1.metric("Tracks", f"{len(t):,}")
    c2.metric("Distinct songs", f"{t['version_key'].nunique():,}")
    c3, c4 = st.columns(2)
    c3.metric("Artists", f"{t['primary_artist'].nunique():,}")
    c4.metric("Years", f"{t['year'].min()}-{t['year'].max()}")
    feat = st.selectbox("Feature distribution", [*RADAR_FEATURES, "tempo", "loudness", "year", "popularity"])
    fig = go.Figure(go.Histogram(x=t[feat], nbinsx=60, marker_color="#1f77b4"))
    fig.update_layout(height=320, margin={"t": 20})
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Feature weights in the active engine**")
    w = pd.DataFrame([{"feature": GROUP_LABELS[g], "weight": v} for g, v in engine.config.weights.items()]).sort_values(
        "weight", ascending=False
    )
    st.dataframe(w, hide_index=True)
