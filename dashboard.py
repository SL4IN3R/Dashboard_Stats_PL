# ============================================================
# DASHBOARD INTERACTIF — PROJET ANALYSE DE DONNÉES EPT 2025-2026
# Performances en Premier League (2015-2023) — Understat.com
# Lancer avec :  streamlit run dashboard.py
# ============================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # évite la fuite mémoire KMeans/MKL sous Windows

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from statsmodels.tsa.seasonal import seasonal_decompose

# ------------------------------------------------------------
# Configuration générale
# ------------------------------------------------------------
st.set_page_config(
    page_title="Premier League 2015-2023 — Dashboard AD",
    page_icon="⚽",
    layout="wide",
)

C_DOM = "#185FA5"   # bleu   → domicile
C_EXT = "#3B6D11"   # vert   → extérieur
C_NEU = "#f29407"   # orange → neutre / total
PALETTE = ["#185FA5", "#f29407", "#3B6D11", "#B02E2E", "#6A3D9A", "#0F8B8D", "#C74B85", "#7A6C5D"]

VARS_NUM = ["h_goals", "a_goals", "h_xg", "a_xg",
            "h_shot", "a_shot", "h_shotOnTarget", "a_shotOnTarget",
            "h_deep", "a_deep", "h_ppda", "a_ppda"]

NOMS_VARS = {
    "h_goals": "Buts dom.", "a_goals": "Buts ext.",
    "h_xg": "xG dom.", "a_xg": "xG ext.",
    "h_shot": "Tirs dom.", "a_shot": "Tirs ext.",
    "h_shotOnTarget": "Cadrés dom.", "a_shotOnTarget": "Cadrés ext.",
    "h_deep": "Deep dom.", "a_deep": "Deep ext.",
    "h_ppda": "PPDA dom.", "a_ppda": "PPDA ext.",
    "total_goals": "Buts totaux", "total_xg": "xG total",
    "goal_diff": "Diff. de buts", "xg_diff": "Diff. de xG",
}


@st.cache_data
def charger_donnees():
    chemin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "df_pretraite.csv")
    df = pd.read_csv(chemin, parse_dates=["date"])
    return df


def stats_par_equipe(df):
    """Profil moyen de chaque équipe (moyenne domicile/extérieur)."""
    dom = df.groupby("team_h")[["h_goals", "h_xg", "h_shot", "h_shotOnTarget", "h_deep", "h_ppda"]].mean()
    ext = df.groupby("team_a")[["a_goals", "a_xg", "a_shot", "a_shotOnTarget", "a_deep", "a_ppda"]].mean()
    dom.columns = ["Buts", "xG", "Tirs", "Tirs cadrés", "Deep passes", "PPDA"]
    ext.columns = dom.columns
    profils = ((dom + ext) / 2).dropna()
    profils.index.name = "Équipe"
    return profils


def points_par_equipe(df):
    pts_dom = df.groupby("team_h")["h_points"].sum()
    pts_ext = df.groupby("team_a")["a_points"].sum()
    pts = pts_dom.add(pts_ext, fill_value=0).sort_values(ascending=False)
    pts.name = "Points"
    return pts


def gini(valeurs):
    v = np.sort(np.asarray(valeurs, dtype=float))
    n = len(v)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v) / (n * np.sum(v))) - (n + 1) / n


# ------------------------------------------------------------
# Chargement + filtres globaux
# ------------------------------------------------------------
df_full = charger_donnees()

st.sidebar.title("⚽ Premier League")
st.sidebar.caption("Analyse de Données — EPT 2025-2026\nSource : Understat.com")

saisons = sorted(df_full["season"].unique())
saison_min, saison_max = st.sidebar.select_slider(
    "Saisons analysées",
    options=saisons,
    value=(saisons[0], saisons[-1]),
)
df = df_full[(df_full["season"] >= saison_min) & (df_full["season"] <= saison_max)].copy()

equipes = sorted(set(df["team_h"]) | set(df["team_a"]))
st.sidebar.markdown(f"**{len(df)}** matchs · **{len(equipes)}** équipes")
st.sidebar.divider()
st.sidebar.markdown(
    "**Sections du projet**\n"
    "- Vue d'ensemble\n"
    "- Équipes\n"
    "- Statistiques descriptives\n"
    "- Corrélations\n"
    "- ACP\n"
    "- Clustering\n"
    "- Séries temporelles\n"
    "- Prédiction (rég. logistique)"
)

st.title("Performances en Premier League (2015-2023)")
st.caption(f"Période affichée : saisons {saison_min}-{saison_min+1} à {saison_max}-{saison_max+1}")

onglets = st.tabs([
    "📊 Vue d'ensemble", "🏟️ Équipes", "📈 Descriptives",
    "🔗 Corrélations", "🧭 ACP", "🎯 Clustering", "⏱️ Séries temporelles",
    "🔮 Prédiction",
])


# ============================================================
# ONGLET 1 — VUE D'ENSEMBLE
# ============================================================
with onglets[0]:
    total = len(df)
    vic_dom = (df["result"] == "V").sum()
    nuls = (df["result"] == "N").sum()
    vic_ext = (df["result"] == "D").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matchs joués", f"{total:,}".replace(",", " "))
    c2.metric("Buts / match", f"{df['total_goals'].mean():.2f}")
    c3.metric("Victoires domicile", f"{vic_dom / total * 100:.1f}%")
    c4.metric("Matchs nuls", f"{nuls / total * 100:.1f}%")
    c5.metric("Victoires extérieur", f"{vic_ext / total * 100:.1f}%")

    col_g, col_d = st.columns([1, 1.4])

    with col_g:
        fig = go.Figure(go.Pie(
            labels=["Victoire domicile", "Match nul", "Victoire extérieur"],
            values=[vic_dom, nuls, vic_ext],
            marker_colors=[C_DOM, C_NEU, C_EXT],
            hole=0.45,
        ))
        fig.update_layout(title="Distribution des résultats", height=380,
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        par_saison = df.groupby("season").agg(
            buts_dom=("h_goals", "mean"),
            buts_ext=("a_goals", "mean"),
            buts_tot=("total_goals", "mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=par_saison["season"], y=par_saison["buts_dom"],
                        name="Buts domicile", line=dict(color=C_DOM, width=2.5))
        fig.add_scatter(x=par_saison["season"], y=par_saison["buts_ext"],
                        name="Buts extérieur", line=dict(color=C_EXT, width=2.5))
        fig.add_scatter(x=par_saison["season"], y=par_saison["buts_tot"],
                        name="Buts totaux", line=dict(color=C_NEU, width=2.5, dash="dot"))
        fig.update_layout(title="Buts moyens par match selon la saison",
                          xaxis_title="Saison", yaxis_title="Buts moyens",
                          height=380, hovermode="x unified", margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("xG vs buts réels — le modèle Understat est-il fiable ?")
    col_g, col_d = st.columns(2)
    with col_g:
        fig = px.scatter(df, x="h_xg", y="h_goals", opacity=0.35,
                         labels={"h_xg": "xG domicile", "h_goals": "Buts domicile"},
                         color_discrete_sequence=[C_DOM], trendline="ols",
                         title="Domicile : xG vs buts réels")
        fig.update_layout(height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        fig = px.scatter(df, x="a_xg", y="a_goals", opacity=0.35,
                         labels={"a_xg": "xG extérieur", "a_goals": "Buts extérieur"},
                         color_discrete_sequence=[C_EXT], trendline="ols",
                         title="Extérieur : xG vs buts réels")
        fig.update_layout(height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"**Lecture :** sur la période, l'équipe à domicile gagne {vic_dom/total*100:.1f}% "
        f"des matchs contre {vic_ext/total*100:.1f}% pour l'équipe en déplacement — "
        f"l'avantage du terrain est bien réel. Les xG moyens "
        f"({df['h_xg'].mean():.2f} dom. / {df['a_xg'].mean():.2f} ext.) sont très proches "
        f"des buts réels ({df['h_goals'].mean():.2f} / {df['a_goals'].mean():.2f})."
    )


# ============================================================
# ONGLET 2 — ÉQUIPES
# ============================================================
with onglets[1]:
    pts = points_par_equipe(df)
    profils = stats_par_equipe(df)

    col_g, col_d = st.columns([1.3, 1])

    with col_g:
        top_n = st.slider("Nombre d'équipes affichées", 5, min(20, len(pts)), 10)
        top = pts.head(top_n).sort_values()
        fig = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker_color=C_DOM, text=top.values.astype(int), textposition="outside",
        ))
        fig.update_layout(title=f"Top {top_n} — Points cumulés sur la période",
                          xaxis_title="Points", height=max(380, 32 * top_n),
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        equipe_sel = st.selectbox("Profil d'une équipe", profils.index.tolist(),
                                  index=profils.index.tolist().index("Manchester City")
                                  if "Manchester City" in profils.index else 0)

        # Radar en percentiles (PPDA inversé : pressing intense = PPDA faible)
        perc = profils.rank(pct=True) * 100
        perc["PPDA"] = 100 - perc["PPDA"]
        axes_radar = ["Buts", "xG", "Tirs", "Tirs cadrés", "Deep passes", "PPDA"]
        labels_radar = ["Buts", "xG", "Tirs", "Tirs cadrés", "Deep passes", "Pressing"]
        valeurs = perc.loc[equipe_sel, axes_radar].tolist()

        fig = go.Figure(go.Scatterpolar(
            r=valeurs + valeurs[:1],
            theta=labels_radar + labels_radar[:1],
            fill="toself", line_color=C_DOM, fillcolor="rgba(24,95,165,0.25)",
            name=equipe_sel,
        ))
        fig.update_layout(
            title=f"{equipe_sel} — percentile parmi les {len(profils)} équipes",
            polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")),
            height=420, margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Évolution des points par saison pour l'équipe choisie
    pts_dom_s = df[df["team_h"] == equipe_sel].groupby("season")["h_points"].sum()
    pts_ext_s = df[df["team_a"] == equipe_sel].groupby("season")["a_points"].sum()
    pts_saison = pts_dom_s.add(pts_ext_s, fill_value=0)

    fig = go.Figure(go.Bar(x=pts_saison.index, y=pts_saison.values, marker_color=C_NEU,
                           text=pts_saison.values.astype(int), textposition="outside"))
    fig.update_layout(title=f"{equipe_sel} — Points par saison",
                      xaxis_title="Saison", yaxis_title="Points",
                      height=340, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Statistiques moyennes de toutes les équipes"):
        st.dataframe(profils.round(3).sort_values("xG", ascending=False),
                     use_container_width=True)


# ============================================================
# ONGLET 3 — STATISTIQUES DESCRIPTIVES
# ============================================================
with onglets[2]:
    st.subheader("Tendance centrale, dispersion et forme des distributions")

    vars_desc = VARS_NUM + ["total_goals"]
    tableau = pd.DataFrame({
        "Variable": [NOMS_VARS[v] for v in vars_desc],
        "Moyenne": [df[v].mean() for v in vars_desc],
        "Médiane": [df[v].median() for v in vars_desc],
        "Écart-type": [df[v].std() for v in vars_desc],
        "CV (%)": [df[v].std() / df[v].mean() * 100 for v in vars_desc],
        "Min": [df[v].min() for v in vars_desc],
        "Max": [df[v].max() for v in vars_desc],
        "Skewness": [stats.skew(df[v]) for v in vars_desc],
        "Kurtosis": [stats.kurtosis(df[v]) for v in vars_desc],
    }).set_index("Variable").round(3)
    st.dataframe(tableau, use_container_width=True, height=320)
    st.caption("CV < 15% : variable homogène · CV > 30% : forte dispersion. "
               "Skewness > 0 : asymétrie à droite.")

    col_g, col_d = st.columns(2)

    with col_g:
        var_hist = st.selectbox("Variable à explorer",
                                vars_desc, format_func=lambda v: NOMS_VARS[v])
        fig = px.histogram(df, x=var_hist, nbins=40, color_discrete_sequence=[C_DOM],
                           labels={var_hist: NOMS_VARS[var_hist]},
                           title=f"Distribution — {NOMS_VARS[var_hist]}")
        fig.add_vline(x=df[var_hist].mean(), line_color=C_NEU, line_dash="dash",
                      annotation_text=f"moyenne = {df[var_hist].mean():.2f}")
        fig.update_layout(height=400, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        paires = {"Buts": ("h_goals", "a_goals"), "xG": ("h_xg", "a_xg"),
                  "Tirs": ("h_shot", "a_shot"), "PPDA": ("h_ppda", "a_ppda")}
        paire_sel = st.selectbox("Comparaison domicile vs extérieur", list(paires))
        vh, va = paires[paire_sel]
        fig = go.Figure()
        fig.add_box(y=df[vh], name="Domicile", marker_color=C_DOM)
        fig.add_box(y=df[va], name="Extérieur", marker_color=C_EXT)
        fig.update_layout(title=f"Boxplot — {paire_sel} (domicile vs extérieur)",
                          height=400, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Inégalité de la domination — Courbe de Lorenz & indice de Gini")
    pts = points_par_equipe(df).sort_values()
    cum_pts = np.concatenate([[0], np.cumsum(pts.values) / pts.sum()])
    cum_eq = np.linspace(0, 1, len(pts) + 1)
    G = gini(pts.values)

    col_g, col_d = st.columns([1.2, 1])
    with col_g:
        fig = go.Figure()
        fig.add_scatter(x=cum_eq, y=cum_eq, name="Égalité parfaite",
                        line=dict(color="gray", dash="dash"))
        fig.add_scatter(x=cum_eq, y=cum_pts, name="Courbe de Lorenz",
                        line=dict(color=C_DOM, width=2.5), fill="tonexty",
                        fillcolor="rgba(24,95,165,0.15)")
        fig.update_layout(title=f"Courbe de Lorenz des points cumulés — Gini = {G:.3f}",
                          xaxis_title="Part cumulée des équipes",
                          yaxis_title="Part cumulée des points",
                          height=420, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        st.metric("Indice de Gini", f"{G:.4f}")
        st.markdown(
            "- **0** → tous les clubs ont le même nombre de points\n"
            "- **1** → un seul club concentre tous les points\n\n"
            f"Un Gini de **{G:.3f}** traduit une inégalité marquée : "
            "quelques équipes (le « Big 6 ») concentrent une grande partie des points, "
            "d'autant que les clubs promus/relégués ne jouent pas toutes les saisons."
        )


# ============================================================
# ONGLET 4 — CORRÉLATIONS
# ============================================================
with onglets[3]:
    df_corr = df[VARS_NUM].rename(columns=NOMS_VARS)
    mat = df_corr.corr()

    col_g, col_d = st.columns([1.4, 1])
    with col_g:
        fig = px.imshow(mat.round(2), text_auto=True, zmin=-1, zmax=1,
                        color_continuous_scale="RdYlGn", aspect="auto",
                        title="Matrice de corrélation (Pearson)")
        fig.update_layout(height=560, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown("**Explorer une relation**")
        vx = st.selectbox("Variable X", VARS_NUM, index=VARS_NUM.index("h_xg"),
                          format_func=lambda v: NOMS_VARS[v])
        vy = st.selectbox("Variable Y", VARS_NUM, index=VARS_NUM.index("h_goals"),
                          format_func=lambda v: NOMS_VARS[v])
        r = df[vx].corr(df[vy])
        st.metric("Coefficient r de Pearson", f"{r:.3f}")
        fig = px.scatter(df, x=vx, y=vy, opacity=0.3, trendline="ols",
                         color_discrete_sequence=[C_DOM],
                         labels={vx: NOMS_VARS[vx], vy: NOMS_VARS[vy]})
        fig.update_layout(height=380, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Lecture :** les corrélations fortes se trouvent au sein d'un même camp "
        "(ex. tirs ↔ tirs cadrés ↔ xG domicile). Les statistiques domicile et extérieur "
        "sont faiblement corrélées négativement : quand une équipe domine, l'autre subit."
    )


# ============================================================
# ONGLET 5 — ACP
# ============================================================
with onglets[4]:
    X = StandardScaler().fit_transform(df[VARS_NUM])
    pca = PCA().fit(X)
    ve = pca.explained_variance_ratio_ * 100
    vc = np.cumsum(ve)
    n_kaiser = int((pca.explained_variance_ > 1).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Composantes (règle de Kaiser)", n_kaiser)
    c2.metric("Variance CP1 + CP2", f"{vc[1]:.1f}%")
    c3.metric(f"Variance des {n_kaiser} CP retenues", f"{vc[n_kaiser-1]:.1f}%")

    col_g, col_d = st.columns(2)

    with col_g:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=[f"CP{i+1}" for i in range(len(ve))], y=ve,
                    marker_color=C_DOM, name="Variance expliquée")
        fig.add_scatter(x=[f"CP{i+1}" for i in range(len(ve))], y=vc,
                        line=dict(color=C_NEU, width=2.5), name="Variance cumulée",
                        secondary_y=True)
        fig.update_layout(title="Éboulis des valeurs propres",
                          height=440, margin=dict(t=50, b=10))
        fig.update_yaxes(title_text="% variance", secondary_y=False)
        fig.update_yaxes(title_text="% cumulé", range=[0, 105], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        # Cercle des corrélations (CP1 × CP2)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        fig = go.Figure()
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                        line=dict(color="lightgray"), showlegend=False)
        for i, v in enumerate(VARS_NUM):
            couleur = C_DOM if v.startswith("h_") else C_EXT
            fig.add_annotation(x=loadings[i, 0], y=loadings[i, 1], ax=0, ay=0,
                               axref="x", ayref="y", xref="x", yref="y",
                               showarrow=True, arrowhead=2, arrowcolor=couleur)
            fig.add_scatter(x=[loadings[i, 0]], y=[loadings[i, 1]], mode="text",
                            text=[NOMS_VARS[v]], textposition="top center",
                            textfont=dict(color=couleur, size=11), showlegend=False)
        fig.update_layout(
            title="Cercle des corrélations (bleu = domicile, vert = extérieur)",
            xaxis=dict(title=f"CP1 ({ve[0]:.1f}%)", range=[-1.15, 1.15],
                       zeroline=True, scaleanchor="y"),
            yaxis=dict(title=f"CP2 ({ve[1]:.1f}%)", range=[-1.15, 1.15], zeroline=True),
            height=440, margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Plan factoriel des matchs
    coords = pca.transform(X)
    df_plan = pd.DataFrame({
        "CP1": coords[:, 0], "CP2": coords[:, 1],
        "Résultat": df["result"].map({"V": "Victoire domicile", "N": "Nul",
                                      "D": "Victoire extérieur"}).values,
        "Match": (df["team_h"] + " " + df["h_goals"].astype(str) + "-"
                  + df["a_goals"].astype(str) + " " + df["team_a"]).values,
    })
    fig = px.scatter(df_plan, x="CP1", y="CP2", color="Résultat", opacity=0.45,
                     hover_name="Match",
                     color_discrete_map={"Victoire domicile": C_DOM, "Nul": C_NEU,
                                         "Victoire extérieur": C_EXT},
                     title="Plan factoriel des matchs (CP1 × CP2)",
                     labels={"CP1": f"CP1 ({ve[0]:.1f}%)", "CP2": f"CP2 ({ve[1]:.1f}%)"})
    fig.update_layout(height=520, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Lecture :** CP1 oppose la domination offensive du camp domicile à celle du camp "
        "extérieur : les victoires à domicile (bleu) et à l'extérieur (vert) se séparent "
        "nettement le long des deux premiers axes."
    )


# ============================================================
# ONGLET 6 — CLUSTERING
# ============================================================
with onglets[5]:
    profils = stats_par_equipe(df)
    X_eq = StandardScaler().fit_transform(profils)

    # Méthode du coude + silhouette
    ks = range(2, 9)
    inerties, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_eq)
        inerties.append(km.inertia_)
        silhouettes.append(silhouette_score(X_eq, km.labels_))

    col_g, col_d = st.columns(2)
    with col_g:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=list(ks), y=inerties, name="Inertie (coude)",
                        line=dict(color=C_DOM, width=2.5))
        fig.add_scatter(x=list(ks), y=silhouettes, name="Silhouette",
                        line=dict(color=C_NEU, width=2.5), secondary_y=True)
        fig.update_layout(title="Choix de k — méthode du coude et silhouette",
                          xaxis_title="Nombre de clusters k",
                          height=420, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    best_k = list(ks)[int(np.argmax(silhouettes))]
    with col_d:
        st.metric("k optimal (silhouette max)", best_k)
        k_sel = st.slider("Nombre de clusters à afficher", 2, 8, best_k)
        km = KMeans(n_clusters=k_sel, n_init=10, random_state=42).fit(X_eq)
        profils_cl = profils.copy()
        profils_cl["Cluster"] = km.labels_
        st.markdown("**Composition des clusters** (xG moyen par cluster)")
        resume = (profils_cl.groupby("Cluster")
                  .agg(Équipes=("xG", "size"), xG=("xG", "mean"),
                       Buts=("Buts", "mean"), PPDA=("PPDA", "mean"))
                  .round(2))
        st.dataframe(resume, use_container_width=True)

    # Projection des équipes sur le plan ACP
    pca_eq = PCA(n_components=2).fit(X_eq)
    coords_eq = pca_eq.transform(X_eq)
    df_cl = pd.DataFrame({
        "CP1": coords_eq[:, 0], "CP2": coords_eq[:, 1],
        "Équipe": profils.index,
        "Cluster": [f"Cluster {c}" for c in km.labels_],
    })
    fig = px.scatter(df_cl, x="CP1", y="CP2", color="Cluster", text="Équipe",
                     color_discrete_sequence=PALETTE,
                     title=f"K-Means (k={k_sel}) — profils d'équipes projetés sur le plan ACP")
    fig.update_traces(textposition="top center", textfont_size=10, marker_size=10)
    fig.update_layout(height=560, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Dendrogramme — Classification Ascendante Hiérarchique (Ward)"):
        fig = ff.create_dendrogram(X_eq, labels=profils.index.tolist(),
                                   linkagefun=lambda x: linkage(x, method="ward"))
        fig.update_layout(height=520, margin=dict(t=40, b=120),
                          xaxis_tickangle=-60)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Lecture :** le clustering sépare naturellement les cadors offensifs à pressing "
        "intense (PPDA faible), le ventre mou, et les équipes reléguables à faible volume "
        "de jeu. Survolez les points pour identifier chaque équipe."
    )


# ============================================================
# ONGLET 7 — SÉRIES TEMPORELLES
# ============================================================
with onglets[6]:
    serie = (df.set_index("date")
               .resample("MS")["total_goals"]
               .mean())
    serie_pleine = serie.interpolate()  # mois sans matchs (juin/juillet) interpolés

    col_g, col_d = st.columns([2, 1])
    with col_g:
        fenetre = st.slider("Fenêtre de la moyenne mobile (mois)", 3, 12, 6)
    with col_d:
        st.metric("Buts moyens / match (période)", f"{df['total_goals'].mean():.2f}")

    mm = serie_pleine.rolling(fenetre, center=True).mean()
    fig = go.Figure()
    fig.add_scatter(x=serie.index, y=serie.values, name="Buts moyens / mois",
                    line=dict(color=C_DOM, width=1.5), opacity=0.75)
    fig.add_scatter(x=mm.index, y=mm.values, name=f"Moyenne mobile ({fenetre} mois)",
                    line=dict(color=C_NEU, width=3))
    fig.update_layout(title="Évolution mensuelle des buts moyens par match",
                      xaxis_title="Date", yaxis_title="Buts moyens / match",
                      height=420, hovermode="x unified", margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    nb_annees = df["date"].dt.year.nunique()
    if len(serie_pleine.dropna()) >= 24:
        decomp = seasonal_decompose(serie_pleine.dropna(), model="additive", period=12)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=["Tendance", "Saisonnalité", "Résidus"])
        fig.add_scatter(x=decomp.trend.index, y=decomp.trend.values,
                        line=dict(color=C_DOM), showlegend=False, row=1, col=1)
        fig.add_scatter(x=decomp.seasonal.index, y=decomp.seasonal.values,
                        line=dict(color=C_EXT), showlegend=False, row=2, col=1)
        fig.add_scatter(x=decomp.resid.index, y=decomp.resid.values, mode="markers",
                        marker=dict(color=C_NEU, size=4), showlegend=False, row=3, col=1)
        fig.update_layout(title="Décomposition additive (période = 12 mois)",
                          height=560, margin=dict(t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Lecture :** la tendance montre une hausse progressive des buts par match "
            "sur la période (jeu de plus en plus offensif), avec une composante saisonnière "
            "intra-saison : les fêtes de fin d'année (Boxing Day) et les fins de saison "
            "sont traditionnellement plus prolifiques."
        )
    else:
        st.warning("Sélectionnez au moins 2 saisons pour afficher la décomposition "
                   "saisonnière (24 mois minimum).")


# ============================================================
# ONGLET 8 — PRÉDICTION (RÉGRESSION LOGISTIQUE)
# ============================================================
with onglets[7]:
    st.subheader("Prédire le résultat d'un match à partir des statistiques de jeu")
    st.caption(
        "Régression logistique multinomiale — cible : V (victoire domicile), "
        "N (nul), D (victoire extérieur). Features : xG, tirs, tirs cadrés, "
        "deep passes et PPDA des deux équipes."
    )

    FEATURES = ["h_xg", "a_xg", "h_shot", "a_shot",
                "h_shotOnTarget", "a_shotOnTarget",
                "h_deep", "a_deep", "h_ppda", "a_ppda"]

    X_pred = df[FEATURES]
    y_pred_cible = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_pred, y_pred_cible, test_size=0.2, random_state=42, stratify=y_pred_cible
    )
    scaler_pred = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(scaler_pred.transform(X_train), y_train)
    y_hat = model.predict(scaler_pred.transform(X_test))

    accuracy = accuracy_score(y_test, y_hat)
    classe_majo = y_train.value_counts().idxmax()
    accuracy_naive = (y_test == classe_majo).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy du modèle", f"{accuracy*100:.1f}%")
    c2.metric(f"Modèle naïf (toujours « {classe_majo} »)", f"{accuracy_naive*100:.1f}%")
    c3.metric("Gain vs naïf", f"+{(accuracy - accuracy_naive)*100:.1f} pts")
    c4.metric("Matchs de test", len(y_test))

    col_g, col_d = st.columns(2)

    with col_g:
        LABELS_RES = ["V", "N", "D"]
        NOMS_RES = ["Victoire dom.", "Nul", "Victoire ext."]
        cm = confusion_matrix(y_test, y_hat, labels=LABELS_RES)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=[f"Prédit {n}" for n in NOMS_RES],
                        y=[f"Réel {n}" for n in NOMS_RES],
                        title="Matrice de confusion (ensemble de test)")
        fig.update_layout(height=440, margin=dict(t=50, b=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        coefs = pd.DataFrame(model.coef_, columns=FEATURES, index=model.classes_)
        importance = coefs.abs().mean(axis=0).sort_values()
        fig = go.Figure(go.Bar(
            x=importance.values,
            y=[NOMS_VARS[v] for v in importance.index],
            orientation="h", marker_color=C_DOM,
        ))
        fig.update_layout(
            title="Importance des variables<br><sup>moyenne des |coefficients| sur les 3 classes</sup>",
            xaxis_title="Importance", height=440, margin=dict(t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Rapport de classification détaillé"):
        rapport = classification_report(y_test, y_hat, output_dict=True)
        df_rapport = pd.DataFrame(rapport).T.round(3)
        df_rapport.index = df_rapport.index.map(
            {"V": "Victoire dom. (V)", "N": "Nul (N)", "D": "Victoire ext. (D)",
             "accuracy": "Accuracy", "macro avg": "Moyenne macro",
             "weighted avg": "Moyenne pondérée"})
        st.dataframe(df_rapport, use_container_width=True)
        st.caption(
            "Le modèle prédit bien les victoires nettes mais peine sur les matchs "
            "nuls — un résultat classique : le nul est la classe la moins fréquente "
            "et la plus « aléatoire » (un match dominé peut finir 0-0)."
        )

    # --- Simulateur interactif ---
    st.divider()
    st.subheader("🎮 Simulateur — entrez les stats d'un match")
    st.caption("Réglez les statistiques des deux équipes, le modèle prédit le résultat. "
               "Rappel : PPDA faible = pressing intense.")

    col_dom, col_ext = st.columns(2)
    saisie = {}
    BORNES = {  # (min, max, défaut domicile, défaut extérieur, pas)
        "xg":           (0.0, 6.0, float(df["h_xg"].mean().round(1)), float(df["a_xg"].mean().round(1)), 0.1),
        "shot":         (0, 40, int(df["h_shot"].mean()), int(df["a_shot"].mean()), 1),
        "shotOnTarget": (0, 20, int(df["h_shotOnTarget"].mean()), int(df["a_shotOnTarget"].mean()), 1),
        "deep":         (0, 40, int(df["h_deep"].mean()), int(df["a_deep"].mean()), 1),
        "ppda":         (2.0, 30.0, float(df["h_ppda"].mean().round(1)), float(df["a_ppda"].mean().round(1)), 0.5),
    }
    NOMS_SLIDERS = {"xg": "xG", "shot": "Tirs", "shotOnTarget": "Tirs cadrés",
                    "deep": "Deep passes", "ppda": "PPDA (pressing)"}

    with col_dom:
        st.markdown(f"**🏠 Équipe à domicile**")
        for var, (vmin, vmax, def_h, _, pas) in BORNES.items():
            saisie[f"h_{var}"] = st.slider(f"{NOMS_SLIDERS[var]} (dom.)",
                                           vmin, vmax, def_h, pas)
    with col_ext:
        st.markdown(f"**✈️ Équipe à l'extérieur**")
        for var, (vmin, vmax, _, def_a, pas) in BORNES.items():
            saisie[f"a_{var}"] = st.slider(f"{NOMS_SLIDERS[var]} (ext.)",
                                           vmin, vmax, def_a, pas)

    x_match = pd.DataFrame([[saisie[f] for f in FEATURES]], columns=FEATURES)
    probas = model.predict_proba(scaler_pred.transform(x_match))[0]
    probas_dict = dict(zip(model.classes_, probas))
    verdict = model.classes_[np.argmax(probas)]
    NOM_VERDICT = {"V": "🏠 Victoire de l'équipe à domicile",
                   "N": "🤝 Match nul",
                   "D": "✈️ Victoire de l'équipe à l'extérieur"}

    col_g, col_d = st.columns([1, 1.3])
    with col_g:
        st.markdown(f"### Prédiction : {NOM_VERDICT[verdict]}")
        st.metric("Confiance du modèle", f"{probas_dict[verdict]*100:.1f}%")
    with col_d:
        fig = go.Figure(go.Bar(
            x=[probas_dict["V"] * 100, probas_dict["N"] * 100, probas_dict["D"] * 100],
            y=["Victoire domicile", "Match nul", "Victoire extérieur"],
            orientation="h", marker_color=[C_DOM, C_NEU, C_EXT],
            text=[f"{probas_dict[k]*100:.1f}%" for k in ["V", "N", "D"]],
            textposition="outside",
        ))
        fig.update_layout(title="Probabilités prédites", xaxis_range=[0, 105],
                          xaxis_title="Probabilité (%)", height=300,
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
