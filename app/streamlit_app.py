"""
Skipsprediksjon i Trondheimsfjorden - Streamlit-dashboard.

Kjør lokalt:  streamlit run app/streamlit_app.py
Prognosene er forhåndsberegnet (ais-demo-data) og ligger i app/data/, resultatene (ais-evaluate)
i results/results.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA, RESULTS = ROOT / "data", ROOT.parent / "results" / "results.json"

# Samme farger som bussappen: svart = faktisk, mørkeblå = modell, grå = baseline
COLOR_ACTUAL, COLOR_MODEL, COLOR_BASELINE = "#1A1A1A", "#1F4E79", "#8A8F98"
COLOR_RULE, COLOR_INPUT, COLOR_TRACK = "#C9A227", "#8FB3D9", "#D5D8DD"
MODELS = {"GRU": COLOR_MODEL, "Rute": COLOR_BASELINE, "Fysikk": COLOR_RULE}
NAMES = {"GRU": "GRU", "rute": "Rute", "fysikk": "Fysikk"}

st.set_page_config(page_title="Skipsprediksjon i Trondheimsfjorden",
                   page_icon=":material/directions_boat:", layout="wide")


def fmt_num(x: float, dec: int = 1, sign: bool = False) -> str:
    """Tall med desimalkomma: 3.64 -> '3,6', med sign=True '+3,6'."""
    return (f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}").replace(".", ",")


def fmt_m(x: float) -> str:
    return f"{x:,.0f} m".replace(",", " ")


def card_title(text: str):
    st.markdown(f"**{text}**")


@st.cache_data
def load():
    res = pd.DataFrame(json.loads(RESULTS.read_text(encoding="utf-8")))
    res["modell"] = res["modell"].map(NAMES)
    return (pd.read_parquet(DATA / "tracks.parquet"), pd.read_parquet(DATA / "predictions.parquet"),
            res)


tracks, preds, res = load()


def result(group: str, model: str, h: int = 30) -> pd.Series:
    return res[(res.gruppe == group) & (res.modell == model) & (res.horisont == h)].iloc[0]


def change(model: str, other: str, group: str = "alle", h: int = 30) -> str:
    return fmt_num((result(group, model, h).snitt / result(group, other, h).snitt - 1) * 100,
                   sign=True)


# ================= Topp =================
st.title("Skipsprediksjon i Trondheimsfjorden")
st.write(
    "Hvor er skipet om 10, 20 og 30 minutter? En **GRU** (et nevralt nett i PyTorch) er trent på "
    "AIS-posisjoner fra Kystverket for april-juli 2026 og testet på september, som den aldri har sett. "
    "Den får bare de siste 30 minuttene av turen og sammenlignes med to baselines: en **fysikkbaseline** "
    "som antar samme fart og kurs, og en **rutebaseline** som gjør det andre skip gjorde på samme sted."
)

k1, k2, k3, k4 = st.columns(4)
gru30 = result("alle", "GRU")
k1.metric("Feil om 30 min (GRU)", fmt_m(gru30.snitt), delta=f"{change('GRU', 'Rute')} % mot rute",
          delta_color="inverse", border=True,
          help=f"Gjennomsnittlig avstand mellom forutsagt og faktisk posisjon. 95 % KI: "
               f"{fmt_m(gru30.lav)}-{fmt_m(gru30.hoy)} (bootstrap over turer).")
k2.metric("Feil om 10 min (GRU)", fmt_m(result("alle", "GRU", 10).snitt),
          delta=f"{change('GRU', 'Rute', h=10)} % mot rute", delta_color="inverse", border=True)
k3.metric("Skip den aldri har sett (30 min)", fmt_m(result("nye skip", "GRU").snitt),
          delta=f"{change('GRU', 'Rute', 'nye skip')} % mot rute", delta_color="inverse", border=True,
          help="Omtrent hvert femte skip er holdt helt utenfor treningen.")
k4.metric("Rett kurs (30 min)", fmt_m(result("situasjon: rett kurs", "GRU").snitt),
          delta=f"{change('GRU', 'Fysikk', 'situasjon: rett kurs')} % mot fysikk",
          delta_color="inverse", border=True,
          help="Når skipet holder kurs og fart er fysikkbaselinen sterkest.")

with st.container(border=True):
    for col, title, text in zip(st.columns(3), [
        "Fysikk: samme fart og kurs",
        "Rute: gjør som andre skip",
        "GRU: lærer av de siste 30 minuttene",
    ], [
        "Forlenger farten og kursen fra de to siste minuttene. Treffer godt på rett strekning, bommer "
        "i svinger.",
        "Finner skip i treningsdataene på samme sted med lik fart og kurs, og tar medianen av hva de "
        "gjorde.",
        "Leser posisjon, fart og plassering i fjorden og forutsier avviket fra fysikkbaselinen.",
    ], strict=True):
        with col:
            card_title(title)
            st.caption(text)

# ================= Utforsk en tur: kontroller til venstre, kart til høyre =================
st.subheader("Utforsk en tur fra testperioden")
left, right = st.columns([1.2, 3])

first = preds[preds["model"] == preds["model"].iloc[0]]
info = first.groupby("trip_id").agg(mmsi=("mmsi", "first"), start=("time", "min"),
                                    end=("time", "max"), new_ship=("new_ship", "first"))
info["turns"] = first.groupby("trip_id")["situation"].apply(lambda s: (s == "sving").mean())
info["sog"] = tracks.groupby("trip_id")["sog"].median()
info = info.sort_values("turns", ascending=False)
# Første tur som vises: den der GRU-en slår rutebaselinen mest om 30 min (står i bildeteksten)
err30 = preds[preds["horizon"] == 30].pivot_table(index="trip_id", columns="model",
                                                  values="error_m")
default_trip = ((err30["Rute"] - err30["GRU"]).idxmax() if "GRU" in err30 else info.index[0])


def label(trip_id: int) -> str:
    r = info.loc[trip_id]
    return f"{r['start']:%d.%m kl. %H:%M}, MMSI {r['mmsi']}{', nytt skip' if r['new_ship'] else ''}"


with left:
    with st.container(border=True):
        trip_id = st.selectbox("Tur", info.index, format_func=label,
                               index=info.index.get_loc(default_trip),
                               help="Sortert etter hvor mye av turen skipet svinger.")
        st.caption("Utvalget er de 20 turene i fart med flest svinger og 10 tilfeldige, alle fra "
                   "september. Turen som vises først er den der GRU-en slår rutebaselinen mest. "
                   "Velg andre turer for å se hvor alle modellene bommer.")

p_trip = preds[preds["trip_id"] == trip_id]
track = tracks[tracks["trip_id"] == trip_id]
times = sorted(p_trip["time"].unique())
models = [m for m in MODELS if m in set(p_trip["model"])]
by_time = dict(list(p_trip.groupby("time")))
trip_err = p_trip[p_trip["horizon"] == 30].groupby("model")["error_m"].mean()


def fit_zoom(lat: pd.Series, lon: pd.Series, width: int = 900, height: int = 500) -> float:
    """Zoom som får alle punktene med (Web Mercator, 512 px fliser)."""
    merc = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    lon_span, y_span = max(lon.max() - lon.min(), 1e-3), max(merc.max() - merc.min(), 1e-5)
    zoom_x = np.log2(360 * width / (512 * lon_span))
    zoom_y = np.log2(2 * np.pi * height / (512 * y_span))
    return float(np.clip(min(zoom_x, zoom_y), 7, 14))


def moment(t) -> tuple[list, dict, pd.DataFrame]:
    """Sporene, feilboksen og punktene kartet skal vise for ett tidspunkt. Rekkefølgen på
    sporene må være lik i alle rammer."""
    past = track[(track["time"] > t - pd.Timedelta("30min")) & (track["time"] <= t)]
    future = track[(track["time"] >= t) & (track["time"] <= t + pd.Timedelta("30min"))]
    now, p_now = past.iloc[-1], by_time[t].sort_values("horizon")
    traces = [
        go.Scattermap(lat=past["lat"], lon=past["lon"], mode="lines", name="Siste 30 min (input)",
                      line={"color": COLOR_INPUT, "width": 5}),
        go.Scattermap(lat=future["lat"], lon=future["lon"], mode="lines", name="Faktisk",
                      line={"color": COLOR_ACTUAL, "width": 4}),
    ]
    lines = [f"<b>Kl. {t:%H:%M}</b>, feil i meter (10 / 20 / 30 min)"]
    for model in models:
        g = p_now[p_now["model"] == model]
        traces.append(go.Scattermap(
            lat=[now["lat"], *g["lat"]], lon=[now["lon"], *g["lon"]], mode="lines+markers+text",
            name=model, line={"color": MODELS[model], "width": 3},
            marker={"size": [0, 8, 10, 13], "color": MODELS[model]},
            text=["", "", "", model], textposition="top right",  # navn ved punktet, ikke bare farge
            textfont={"color": MODELS[model], "size": 13},
            hovertext=["nå", *[f"{model}, om {h} min" for h in g["horizon"]]], hoverinfo="text"))
        lines.append(f"{model}: " + " / ".join(f"{e:.0f}" for e in g["error_m"]))
    traces.append(go.Scattermap(lat=[now["lat"]], lon=[now["lon"]], mode="markers", name="Nå",
                                marker={"size": 14, "color": COLOR_ACTUAL}))
    lines.append(f"Neste 30 min: {p_now['situation'].iloc[0]}")
    box = {"text": "<br>".join(lines), "x": 0.99, "y": 0.99, "xref": "paper", "yref": "paper",
           "xanchor": "right", "yanchor": "top", "align": "left", "showarrow": False,
           "font": {"color": COLOR_ACTUAL, "size": 13}, "bgcolor": "rgba(255,255,255,0.92)",
           "bordercolor": "#C9CDD3", "borderwidth": 1, "borderpad": 6}
    # Kartet følger skipet. Sentrum er der skipet faktisk var 15 min senere: det sporet er jevnt,
    # så kartet glir i stedet for å hoppe etter prediksjonene. Utsnittet speiles rundt sentrum,
    # så nå-posisjonen, det faktiske sporet og alle prediksjonene kommer med.
    mid = future.iloc[min(15, len(future) - 1)]
    pts = pd.concat([future[["lat", "lon"]], p_now[["lat", "lon"]]])
    view = pd.concat([pts, 2 * mid[["lat", "lon"]].astype(float) - pts])
    return traces, box, view


def center(view: pd.DataFrame) -> dict:
    return {"lat": float(view["lat"].mean()), "lon": float(view["lon"].mean())}


# Tidsglideren er en del av Plotly-figuren, så den går i nettleseren uten at Streamlit laster
# siden og kartet på nytt.
start = len(times) // 2
frames, zooms = [], []
for i, t in enumerate(times):
    traces, box, view = moment(t)
    zooms.append(fit_zoom(view["lat"], view["lon"]))
    frames.append(go.Frame(name=str(i), data=traces, traces=list(range(1, len(traces) + 1)),
                           layout={"annotations": [box], "map": {"center": center(view)}}))
# Fast zoom for hele turen (den som får med det største utsnittet), så kartet ikke hopper
zoom = min(zooms) - 0.1
traces, box, view = moment(times[start])
fig = go.Figure(
    data=[go.Scattermap(lat=track["lat"], lon=track["lon"], mode="lines", name="Hele turen",
                        line={"color": COLOR_TRACK, "width": 3}, hoverinfo="skip"), *traces],
    frames=frames)
step = {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
        "transition": {"duration": 0}}
play = {"mode": "immediate", "fromcurrent": True, "frame": {"duration": 250, "redraw": True},
        "transition": {"duration": 0}}
fig.update_layout(
    map={"style": "open-street-map", "zoom": zoom,
         "center": center(view)},
    uirevision=f"{trip_id}-{zoom:.2f}",  # beholder zoom og panorering mens du drar i glidebryteren
    annotations=[box], height=600, margin={"l": 0, "r": 0, "t": 30, "b": 0},
    font={"family": "sans-serif", "color": COLOR_ACTUAL},
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0,
            "font": {"size": 13}},
    sliders=[{"active": start, "currentvalue": {"visible": False}, "pad": {"t": 10},
              "x": 0.18, "len": 0.82, "bgcolor": "#F3F5F8", "activebgcolor": COLOR_MODEL,
              "steps": [{"method": "animate", "args": [[str(i)], step],
                         "label": f"{t:%H:%M}" if t.minute % 10 == 0 else ""}  # færre merker
                        for i, t in enumerate(times)]}],
    updatemenus=[{"type": "buttons", "direction": "left", "x": 0.0, "y": 0, "xanchor": "left",
                  "yanchor": "top", "pad": {"t": 18}, "bgcolor": "#F3F5F8",
                  "buttons": [{"label": "▶ Spill av", "method": "animate", "args": [None, play]},
                              {"label": "⏸ Pause", "method": "animate", "args": [[None], step]}]}],
)

with right:
    with st.container(border=True):
        r = info.loc[trip_id]
        card_title(f"MMSI {r['mmsi']}, {r['start']:%d.%m.%Y} kl. {r['start']:%H:%M}-{r['end']:%H:%M}"
                   + (" (skip modellen aldri har sett)" if r["new_ship"] else ""))
        m1, m2, m3 = st.columns(3)
        m1.metric("Fart (median)", f"{fmt_num(r['sog'])} knop")
        m2.metric("Feil om 30 min, GRU", fmt_m(trip_err["GRU"]) if "GRU" in trip_err else "-",
                  help="Gjennomsnitt over alle tidspunktene på turen.")
        m3.metric("Feil om 30 min, rute", fmt_m(trip_err["Rute"]),
                  help="Gjennomsnitt over alle tidspunktene på turen.")
        if "GRU" in trip_err:
            diff = trip_err["Rute"] - trip_err["GRU"]
            better = "GRU-en" if diff > 0 else "rutebaselinen"
            st.caption(f"På denne turen bommet {better} i snitt **{fmt_m(abs(diff))} mindre** "
                       "om 30 minutter.")
        st.plotly_chart(fig, width="stretch")

with left:
    with st.container(border=True):
        card_title("Slik leser du kartet")
        st.markdown(
            "- **Lyseblå:** de 30 minuttene modellene ser.\n"
            "- **Svart:** hvor skipet faktisk gikk de neste 30 minuttene.\n"
            "- **Mørkeblå (GRU), grå (rute), gul (fysikk):** forutsagt posisjon om 10, 20 og 30 "
            "min, med navnet ved 30-minuttersprikken."
        )
        st.caption("Dra i glidebryteren under kartet eller trykk Spill av. Kartet følger skipet, "
                   "og feilboksen oppe til "
                   "høyre følger med.")

# ================= Resultater og metode =================
c, e = st.columns(2)
with c:
    with st.container(border=True):
        card_title("Resultater på testperioden (september 2026)")
        groups = {"alle": "Alle", "nye skip": "Skip den aldri har sett",
                  "situasjon: rett kurs": "Rett kurs", "situasjon: sving": "Sving",
                  "situasjon: fartsendring": "Fartsendring", "situasjon: lav fart": "Lav fart"}
        n_all = result("alle", "Rute").n
        rows = []
        for g, name in groups.items():
            if g in set(res.gruppe):
                row = {"Gruppe": name, "Andel": f"{result(g, 'Rute').n / n_all:.0%}"}
                row |= {m: fmt_m(result(g, m).snitt) for m in ("Fysikk", "Rute", "GRU")}
                rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"Gjennomsnittlig feil om 30 minutter over {n_all:,} tidspunkter (hvert 5. "
                   "minutt). Situasjonen er hva skipet gjør de neste 30 minuttene.".replace(",", " "))

        card_title("Metode")
        st.markdown(
            """
- **Data:** AIS-posisjoner fra Kystverket for april-september 2026 i Trondheimsfjorden, renset og
  gjort om til turer med faste 1-minutters steg (2 millioner minutter fra 1080 skip).
- **Tidsbasert splitt:** trening april-juli, valg av epoke på august, test september. I tillegg er
  omtrent hvert femte skip holdt helt utenfor trening og validering.
- **GRU:** to lag med 128 enheter. Forutsier avviket fra fysikkbaselinen, så rett kurs kommer gratis
  og kapasiteten går til svinger. Trent med gjennomsnittlig avstand som tap og tidlig stopp på august.
- **Usikkerhet:** 95 % konfidensintervall med bootstrap over hele turer.
"""
        )

with e:
    with st.container(border=True):
        card_title("Feil per horisont")
        h = res[res.gruppe == "alle"].pivot(index="horisont", columns="modell", values="snitt")
        st.dataframe(h.rename(index=lambda x: f"Om {x} min").rename_axis("Horisont").map(fmt_m)[["Fysikk", "Rute", "GRU"]])
        st.caption("Feilen vokser raskt for fysikkbaselinen, fordi én sving er nok til å havne langt "
                   "unna. GRU-en og rutebaselinen har lært hvor leia går.")

        card_title("Begrensninger")
        st.markdown(
            """
- Bare én fjord og et halvt år. Vinter, uvær og andre farvann er ikke testet.
- Fiskefartøy under 15 m og fritidsbåter under 45 m er ikke med i de åpne dataene.
- Modellen vet ikke hvor skipet skal. Når et skip snur eller går inn til kai uten forvarsel,
  bommer alle modellene.
- Én treningskjøring: konfidensintervallene dekker variasjonen i testdataene, ikke i treningen.
"""
        )

st.caption("Data: Kystverket (NLOD) | Kode: [github.com/jeghetermats/ais-skipsprediksjon](https://github.com/jeghetermats/ais-skipsprediksjon)")
