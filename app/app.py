import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =========================
# Configuración general
# =========================
st.set_page_config(
    page_title="SmartQC AI",
    page_icon="🧪",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "smartqc_logo.png"
UNIR_LOGO_PATH = ASSETS_DIR / "unir_logo_white.png"


# =========================
# Login simulado
# =========================
def render_login_screen() -> None:
    try:
        logo = Image.open(LOGO_PATH)
    except Exception:
        logo = None

    top_left, top_center, top_right = st.columns([1, 1.5, 1])

    with top_center:
        if logo is not None:
            st.image(logo, width=280)

    st.markdown("## SmartQC AI")
    st.caption(
        "Plataforma inteligente para monitoreo de control de calidad, detección proactiva de anomalías y generación de hallazgos para apoyo a la toma de decisiones."
    )

    left, center, right = st.columns([1.2, 1, 1.2])

    with center:
        st.markdown("### Acceso a la plataforma")
        st.caption("Pantalla demostrativa de autenticación para presentación del prototipo.")

        with st.form("login_form", clear_on_submit=False):
            usuario = st.text_input("Usuario", placeholder="analista.calidad")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••")
            perfil = st.selectbox(
                "Perfil",
                ["Analista de calidad", "Supervisor", "Administrador"],
            )
            recordar = st.checkbox("Mantener sesión activa", value=True)
            ingresar = st.form_submit_button("Ingresar", use_container_width=True)

        st.info("Acceso simulado. Puedes usar cualquier usuario y contraseña para la demostración.")

        if ingresar:
            if usuario.strip() and password.strip():
                with st.spinner("Validando credenciales..."):
                    time.sleep(1)
                st.session_state["logged_in"] = True
                st.session_state["usuario_demo"] = usuario.strip()
                st.session_state["perfil_demo"] = perfil
                st.session_state["recordar_demo"] = recordar
                st.rerun()
            else:
                st.error("Ingresa usuario y contraseña para continuar.")

    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.info("🔐 Acceso por perfil")
    k2.info("🧪 Monitoreo de control")
    k3.info("🧠 Alertamiento asistido por IA")


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    render_login_screen()
    st.stop()


# =========================
# Configuración app
# =========================
@dataclass
class AppConfig:
    default_contamination: float = 0.08
    random_state: int = 42


CONFIG = AppConfig()


# =========================
# Datos de ejemplo y simulación
# =========================
def load_example_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    fechas = pd.date_range("2025-01-01", periods=n, freq="D")
    base = rng.normal(loc=100, scale=2.2, size=n)

    anomaly_idx = [18, 39, 67, 91, 110]
    base[anomaly_idx] = base[anomaly_idx] + np.array([8, -7, 10, -9, 12])

    return pd.DataFrame(
        {
            "fecha": fechas,
            "resultado": np.round(base, 2),
            "analito": "Analito piloto",
            "equipo": "Equipo piloto",
            "lote": "L001",
            "nivel_control": "Nivel 1",
            "media_objetivo": 100.0,
            "ds_objetivo": 2.0,
        }
    )


def generate_synthetic_qc_data(
    n: int = 240,
    analito: str = "Glucosa control",
    equipo: str = "Analizador A",
    lote: str = "L001",
    nivel_control: str = "Nivel 1",
    media_objetivo: float = 100.0,
    ds_objetivo: float = 2.0,
    incluir_shift: bool = True,
    incluir_tendencia: bool = True,
    incluir_imprecision: bool = True,
    num_outliers: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2025-01-01", periods=n, freq="D")

    resultados = rng.normal(
        loc=media_objetivo,
        scale=ds_objetivo * 0.6,
        size=n,
    )
    tipo_evento = np.array(["normal"] * n, dtype=object)

    if incluir_shift and n >= 80:
        start = int(n * 0.25)
        end = min(start + int(n * 0.12), n)
        resultados[start:end] += ds_objetivo * 1.4
        tipo_evento[start:end] = "shift"

    if incluir_tendencia and n >= 120:
        start = int(n * 0.48)
        end = min(start + int(n * 0.12), n)
        resultados[start:end] += np.linspace(0, -2.5 * ds_objetivo, end - start)
        tipo_evento[start:end] = "tendencia"

    if incluir_imprecision and n >= 160:
        start = int(n * 0.70)
        end = min(start + int(n * 0.10), n)
        resultados[start:end] = rng.normal(
            loc=media_objetivo - 0.3,
            scale=ds_objetivo * 1.5,
            size=end - start,
        )
        tipo_evento[start:end] = "imprecision"

    if num_outliers > 0 and n > 10:
        posiciones = rng.choice(
            np.arange(5, n - 5),
            size=min(num_outliers, max(n - 10, 1)),
            replace=False,
        )
        for pos in posiciones:
            magnitud = rng.choice([-1, 1]) * rng.uniform(3.2, 4.5) * ds_objetivo
            resultados[pos] = media_objetivo + magnitud
            tipo_evento[pos] = "outlier"

    return pd.DataFrame(
        {
            "fecha": fechas,
            "resultado": np.round(resultados, 2),
            "analito": analito,
            "equipo": equipo,
            "lote": lote,
            "nivel_control": nivel_control,
            "media_objetivo": media_objetivo,
            "ds_objetivo": ds_objetivo,
            "tipo_evento_sintetico": tipo_evento,
        }
    )


def load_lis_simulated_data(
    analito: str,
    equipo: str,
    lote: str,
    nivel_control: str,
    fecha_inicio,
    fecha_fin,
) -> pd.DataFrame:
    dias = max((pd.to_datetime(fecha_fin) - pd.to_datetime(fecha_inicio)).days + 1, 30)

    base_media = {
        "Glucosa control": 100.0,
        "Serología control": 1.2,
        "Química clínica control": 85.0,
    }.get(analito, 100.0)

    base_ds = {
        "Glucosa control": 2.0,
        "Serología control": 0.12,
        "Química clínica control": 1.8,
    }.get(analito, 2.0)

    seed = abs(hash(f"{analito}-{equipo}-{lote}-{nivel_control}")) % (2**32)

    df = generate_synthetic_qc_data(
        n=dias,
        analito=analito,
        equipo=equipo,
        lote=lote,
        nivel_control=nivel_control,
        media_objetivo=base_media,
        ds_objetivo=base_ds,
        incluir_shift=True,
        incluir_tendencia=True,
        incluir_imprecision=True,
        num_outliers=4,
        seed=seed,
    )

    df["fecha"] = pd.date_range(pd.to_datetime(fecha_inicio), periods=len(df), freq="D")
    return df


# =========================
# Utilidades de procesamiento
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def detect_date_column(df: pd.DataFrame) -> str | None:
    candidates = ["fecha", "date", "datetime", "fecha_hora", "timestamp"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def detect_value_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "resultado",
        "valor",
        "value",
        "resultado_control",
        "medicion",
        "measurement",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols[0] if numeric_cols else None


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    df = normalize_columns(df)

    date_col = detect_date_column(df)
    value_col = detect_value_column(df)

    if date_col is None:
        raise ValueError("No se encontró una columna de fecha. Usa una columna como 'fecha'.")
    if value_col is None:
        raise ValueError("No se encontró una columna numérica de resultado. Usa una columna como 'resultado'.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col).reset_index(drop=True)

    df["media_movil_5"] = df[value_col].rolling(5, min_periods=1).mean()
    df["desv_movil_5"] = df[value_col].rolling(5, min_periods=2).std().fillna(0)
    df["delta"] = df[value_col].diff().fillna(0)
    df["zscore"] = (df[value_col] - df[value_col].mean()) / (df[value_col].std(ddof=0) + 1e-9)

    return df, date_col, value_col


def run_isolation_forest(df: pd.DataFrame, contamination: float, value_col: str) -> pd.DataFrame:
    features = [value_col, "media_movil_5", "desv_movil_5", "delta", "zscore"]
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=CONFIG.random_state,
        n_estimators=200,
    )
    preds = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    result = df.copy()
    result["anomalia_if"] = np.where(preds == -1, 1, 0)
    result["score_if"] = scores
    return result


def apply_simple_westgard_proxy(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    result = df.copy()

    if "media_objetivo" in result.columns and result["media_objetivo"].notna().any():
        mean_ = float(result["media_objetivo"].dropna().iloc[0])
    else:
        mean_ = float(result[value_col].mean())

    if "ds_objetivo" in result.columns and result["ds_objetivo"].notna().any():
        std_ = float(result["ds_objetivo"].dropna().iloc[0]) + 1e-9
    else:
        std_ = float(result[value_col].std(ddof=0)) + 1e-9

    result["media_control"] = mean_
    result["ls_1s"] = mean_ + std_
    result["li_1s"] = mean_ - std_
    result["ls_2s"] = mean_ + 2 * std_
    result["li_2s"] = mean_ - 2 * std_
    result["ls_3s"] = mean_ + 3 * std_
    result["li_3s"] = mean_ - 3 * std_

    result["wg_1_3s"] = np.where(
        (result[value_col] > result["ls_3s"]) | (result[value_col] < result["li_3s"]),
        1,
        0,
    )
    result["wg_1_2s"] = np.where(
        (result[value_col] > result["ls_2s"]) | (result[value_col] < result["li_2s"]),
        1,
        0,
    )

    upper_2s = result[value_col] > result["ls_2s"]
    lower_2s = result[value_col] < result["li_2s"]
    result["wg_2_2s"] = np.where(
        (upper_2s & upper_2s.shift(1, fill_value=False))
        | (lower_2s & lower_2s.shift(1, fill_value=False)),
        1,
        0,
    )

    return result


def build_summary(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    total = len(df)
    anomalies = int(df["anomalia_if"].sum()) if "anomalia_if" in df.columns else 0
    wg = int(df["wg_1_3s"].sum()) if "wg_1_3s" in df.columns else 0
    overlap = int(((df["anomalia_if"] == 1) & (df["wg_1_3s"] == 1)).sum())

    return {
        "total_registros": total,
        "anomalias_if": anomalies,
        "alertas_westgard": wg,
        "coincidencias": overlap,
        "fecha_inicio": df[date_col].min(),
        "fecha_fin": df[date_col].max(),
        "media": df[value_col].mean(),
        "desv": df[value_col].std(ddof=0),
    }


def generate_report_text(summary: dict, analito: str = "analito piloto") -> str:
    return f"""
Reporte ejecutivo de SmartQC AI

Se analizaron {summary['total_registros']} registros correspondientes al periodo comprendido entre {summary['fecha_inicio'].date()} y {summary['fecha_fin'].date()} para el analito {analito}.

El modelo Isolation Forest identificó {summary['anomalias_if']} posibles anomalías, mientras que la regla de referencia tipo Westgard 1_3s detectó {summary['alertas_westgard']} alertas. Se observaron {summary['coincidencias']} coincidencias entre ambos enfoques, lo que sugiere que el modelo puede complementar la detección tradicional al señalar comportamientos atípicos no necesariamente capturados por una sola regla estadística.

La media observada fue de {summary['media']:.2f} con una desviación estándar de {summary['desv']:.2f}. Con base en estos resultados, el prototipo demuestra viabilidad como herramienta de apoyo para la supervisión proactiva de controles de calidad.
""".strip()


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Sesión")
    st.success(f"Usuario: {st.session_state.get('usuario_demo', 'demo')}")
    st.caption(f"Perfil: {st.session_state.get('perfil_demo', 'Analista de calidad')}")

    if st.button("Cerrar sesión"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.divider()
    st.header("Parámetros de monitoreo")

    contamination = st.slider(
        "Sensibilidad del modelo (contamination)",
        min_value=0.01,
        max_value=0.20,
        value=CONFIG.default_contamination,
        step=0.01,
        help="Valores más altos hacen al modelo más sensible a posibles anomalías.",
    )

    mostrar_limites = st.checkbox(
        "Mostrar media y límites de control",
        value=True,
        help="Agrega las líneas de control tipo Levey-Jennings a la serie temporal.",
    )

    source = st.radio(
        "Origen de datos",
        options=[
            "Dataset de ejemplo",
            "Subir CSV",
            "Generar dataset sintético",
            "LIS simulado",
        ],
        index=0,
    )

    uploaded_file = None
    synthetic_params = {}
    lis_params = {}

    if source == "Subir CSV":
        uploaded_file = st.file_uploader("Carga un archivo CSV", type=["csv"])
        st.info("Se recomienda usar columnas como: fecha, resultado, analito, equipo.")

    elif source == "Generar dataset sintético":
        st.markdown("**Parámetros del dataset sintético**")
        synthetic_params["n"] = st.slider("Número de registros", 60, 365, 240, 10)
        synthetic_params["analito"] = st.text_input("Analito", value="Glucosa control")
        synthetic_params["equipo"] = st.text_input("Equipo", value="Analizador A")
        synthetic_params["lote"] = st.text_input("Lote", value="L001")
        synthetic_params["nivel_control"] = st.selectbox("Nivel de control", ["Nivel 1", "Nivel 2", "Nivel 3"])
        synthetic_params["media_objetivo"] = st.number_input("Media objetivo", value=100.0, step=0.5)
        synthetic_params["ds_objetivo"] = st.number_input("Desviación estándar objetivo", value=2.0, step=0.1)
        synthetic_params["incluir_shift"] = st.checkbox("Incluir shift", value=True)
        synthetic_params["incluir_tendencia"] = st.checkbox("Incluir tendencia", value=True)
        synthetic_params["incluir_imprecision"] = st.checkbox("Incluir imprecisión alta", value=True)
        synthetic_params["num_outliers"] = st.slider("Número de outliers", 0, 12, 6, 1)
        synthetic_params["seed"] = st.number_input("Semilla", value=42, step=1)

    elif source == "LIS simulado":
        st.markdown("**Consulta simulada a LIS**")
        lis_params["analito"] = st.selectbox(
            "Analito",
            ["Glucosa control", "Serología control", "Química clínica control"],
        )
        lis_params["equipo"] = st.selectbox(
            "Equipo",
            ["Analizador A", "Analizador B", "Analizador C"],
        )
        lis_params["lote"] = st.selectbox(
            "Lote",
            ["L001", "L002", "L003"],
        )
        lis_params["nivel_control"] = st.selectbox(
            "Nivel de control",
            ["Nivel 1", "Nivel 2", "Nivel 3"],
        )
        lis_params["fecha_inicio"] = st.date_input("Fecha inicial", value=pd.to_datetime("2025-01-01"))
        lis_params["fecha_fin"] = st.date_input("Fecha final", value=pd.to_datetime("2025-08-28"))

        consultar_lis = st.button("Consultar datos LIS simulados", use_container_width=True)
        if consultar_lis:
            st.session_state["lis_query_ready"] = True
        elif "lis_query_ready" not in st.session_state:
            st.session_state["lis_query_ready"] = False


# =========================
# Carga y proceso
# =========================
try:
    if source == "Dataset de ejemplo":
        raw_df = load_example_data()

    elif source == "Subir CSV":
        if uploaded_file is None:
            st.warning("Carga un archivo CSV para continuar.")
            st.stop()
        raw_df = pd.read_csv(uploaded_file)

    elif source == "Generar dataset sintético":
        raw_df = generate_synthetic_qc_data(
            n=synthetic_params["n"],
            analito=synthetic_params["analito"],
            equipo=synthetic_params["equipo"],
            lote=synthetic_params["lote"],
            nivel_control=synthetic_params["nivel_control"],
            media_objetivo=synthetic_params["media_objetivo"],
            ds_objetivo=synthetic_params["ds_objetivo"],
            incluir_shift=synthetic_params["incluir_shift"],
            incluir_tendencia=synthetic_params["incluir_tendencia"],
            incluir_imprecision=synthetic_params["incluir_imprecision"],
            num_outliers=synthetic_params["num_outliers"],
            seed=int(synthetic_params["seed"]),
        )
        st.success(f"Dataset sintético generado correctamente: {len(raw_df)} registros.")

    elif source == "LIS simulado":
        if not st.session_state.get("lis_query_ready", False):
            st.info("Configura los filtros y presiona 'Consultar datos LIS simulados'.")
            st.stop()

        raw_df = load_lis_simulated_data(
            analito=lis_params["analito"],
            equipo=lis_params["equipo"],
            lote=lis_params["lote"],
            nivel_control=lis_params["nivel_control"],
            fecha_inicio=lis_params["fecha_inicio"],
            fecha_fin=lis_params["fecha_fin"],
        )
        st.success(f"Consulta completada. Se recuperaron {len(raw_df)} registros del LIS simulado.")

    df, date_col, value_col = preprocess_data(raw_df)
    df = apply_simple_westgard_proxy(df, value_col=value_col)
    df = run_isolation_forest(df, contamination=contamination, value_col=value_col)
    summary = build_summary(df, date_col=date_col, value_col=value_col)

except Exception as e:
    st.error(f"No fue posible procesar la información: {e}")
    st.stop()


# =========================
# Encabezado principal
# =========================
try:
    logo = Image.open(LOGO_PATH)
except Exception:
    logo = None

analito_info = str(df["analito"].iloc[0]) if "analito" in df.columns else "N/D"
equipo_info = str(df["equipo"].iloc[0]) if "equipo" in df.columns else "N/D"
lote_info = str(df["lote"].iloc[0]) if "lote" in df.columns else "N/D"
nivel_info = str(df["nivel_control"].iloc[0]) if "nivel_control" in df.columns else "N/D"
periodo_info = f"{df[date_col].min().date()} a {df[date_col].max().date()}"

total_alertas_estadisticas = int(df["wg_1_3s"].sum()) + int(df["wg_2_2s"].sum())

if total_alertas_estadisticas == 0 and int(df["anomalia_if"].sum()) == 0:
    estado_proceso = "🟢 En control"
elif total_alertas_estadisticas <= 5:
    estado_proceso = "🟡 Revisión recomendada"
else:
    estado_proceso = "🔴 Fuera de control"

header_left, header_right = st.columns([1.1, 4])

with header_left:
    if logo is not None:
        st.image(logo, width=280)
    else:
        st.markdown("## 🧪")

with header_right:
    st.title("SmartQC AI")
    st.caption(
        "Detección proactiva de anomalías, monitoreo de reglas estadísticas y generación de hallazgos para apoyo a la toma de decisiones en control de calidad."
    )

    b1, b2 = st.columns(2)
    b1.success(f"Estado actual: {estado_proceso}")
    b2.info(f"Origen de datos: {source}")

quick1, quick2, quick3 = st.columns(3)
quick1.info("📈 Monitoreo continuo del control")
quick2.info("🧠 Detección asistida por IA")
quick3.info("📄 Informe ejecutivo automatizado")

st.markdown("### Resumen operativo")
ctx1, ctx2, ctx3, ctx4, ctx5 = st.columns(5)
ctx1.metric("Analito", analito_info)
ctx2.metric("Equipo", equipo_info)
ctx3.metric("Lote", lote_info)
ctx4.metric("Nivel de control", nivel_info)
ctx5.metric("Periodo", periodo_info)

st.markdown("### Indicadores clave")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros analizados", f"{summary['total_registros']}")
col2.metric("Eventos detectados por IA", f"{summary['anomalias_if']}")
col3.metric("Alertas estadísticas", f"{summary['alertas_westgard']}")
col4.metric("Eventos confirmados", f"{summary['coincidencias']}")


# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Panel de monitoreo", "Gestión de alertas", "Trazabilidad", "Informe ejecutivo"]
)

with tab1:
    st.subheader("Serie temporal del control")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[value_col],
            mode="lines+markers",
            name="Resultado",
        )
    )

    if mostrar_limites and "media_control" in df.columns:
        fig.add_trace(go.Scatter(x=df[date_col], y=df["media_control"], mode="lines", name="Media objetivo"))
        fig.add_trace(go.Scatter(x=df[date_col], y=df["ls_2s"], mode="lines", name="Límite superior 2s"))
        fig.add_trace(go.Scatter(x=df[date_col], y=df["li_2s"], mode="lines", name="Límite inferior 2s"))
        fig.add_trace(go.Scatter(x=df[date_col], y=df["ls_3s"], mode="lines", name="Límite superior 3s"))
        fig.add_trace(go.Scatter(x=df[date_col], y=df["li_3s"], mode="lines", name="Límite inferior 3s"))

    anom_df = df[df["anomalia_if"] == 1]
    fig.add_trace(
        go.Scatter(
            x=anom_df[date_col],
            y=anom_df[value_col],
            mode="markers+text",
            name="Anomalía IA",
            marker=dict(size=10, symbol="x"),
            text=["Anomalía" for _ in range(len(anom_df))],
            textposition="top center",
        )
    )

    wg_df = df[(df["wg_1_3s"] == 1) | (df["wg_2_2s"] == 1)]
    fig.add_trace(
        go.Scatter(
            x=wg_df[date_col],
            y=wg_df[value_col],
            mode="markers+text",
            name="Alerta regla base",
            marker=dict(size=10, symbol="diamond"),
            text=["Regla" for _ in range(len(wg_df))],
            textposition="bottom center",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Reglas 1_3s", int(df["wg_1_3s"].sum()))
    c2.metric("Reglas 2_2s", int(df["wg_2_2s"].sum()))
    c3.metric("Anomalías IA", int(df["anomalia_if"].sum()))

    st.subheader("Distribución de resultados")
    hist = px.histogram(df, x=value_col, nbins=25)
    st.plotly_chart(hist, use_container_width=True)


with tab2:
    st.subheader("Comparación entre regla base e IA")
    comparison = df[
        [date_col, value_col, "wg_1_2s", "wg_1_3s", "wg_2_2s", "anomalia_if", "score_if"]
    ].copy()
    st.dataframe(comparison, use_container_width=True)

    scatter = px.scatter(
        df,
        x=date_col,
        y="score_if",
        symbol="anomalia_if",
        title="Score del modelo en el tiempo",
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.markdown(
        "**Interpretación sugerida:** los puntos marcados por IA y las reglas de control representan eventos a revisar. La coincidencia entre ambos enfoques fortalece la evidencia de una posible desviación; cuando no coinciden, puede tratarse de patrones más sutiles detectados por el modelo."
    )

    st.subheader("Eventos priorizados")
    eventos = df[
        (df["anomalia_if"] == 1) | (df["wg_1_3s"] == 1) | (df["wg_2_2s"] == 1)
    ].copy()
    if not eventos.empty:
        st.dataframe(
            eventos[
                [date_col, value_col, "wg_1_2s", "wg_1_3s", "wg_2_2s", "anomalia_if", "score_if"]
            ],
            use_container_width=True,
        )
    else:
        st.info("No se identificaron eventos priorizados con la configuración actual.")


with tab3:
    st.subheader("Vista de datos procesados")
    st.dataframe(df, use_container_width=True)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Descargar resultados en CSV",
        data=csv_buffer.getvalue(),
        file_name="resultados_smartqc_ai.csv",
        mime="text/csv",
    )


with tab4:
    st.subheader("Reporte automático")
    analito = str(df["analito"].iloc[0]) if "analito" in df.columns else "analito piloto"
    report = generate_report_text(summary, analito=analito)

    if "wg_2_2s" in df.columns:
        report += (
            f"\n\nDe forma complementaria, la regla simplificada 2_2s identificó "
            f"{int(df['wg_2_2s'].sum())} eventos, lo cual aporta una referencia adicional "
            f"de comportamiento fuera de control."
        )

    st.text_area("Resumen ejecutivo", value=report, height=260)

    st.download_button(
        label="Descargar reporte TXT",
        data=report,
        file_name="reporte_smartqc_ai.txt",
        mime="text/plain",
    )


st.divider()
with st.expander("Guía rápida para tu tesis"):
    st.markdown(
        """
        - Usa capturas de esta aplicación como evidencia del prototipo funcional.
        - Muestra la pestaña de serie temporal, la tabla de anomalías y el reporte automático.
        - Incluye como valor agregado la generación de datasets sintéticos y la simulación de consulta a LIS.
        - Este prototipo puede desplegarse en Streamlit Community Cloud enlazándolo a un repositorio en GitHub.
        """
    )


# =========================
# Pie de página institucional
# =========================
try:
    unir_logo = Image.open(UNIR_LOGO_PATH)
except Exception:
    unir_logo = None

st.divider()

f1, f2 = st.columns([1, 4])

with f1:
    if unir_logo is not None:
        st.image(unir_logo, width=130)

with f2:
    st.info(
        "SmartQC AI © 2026\n\n"
        "Prototipo académico desarrollado en el contexto de la Maestría en Inteligencia Artificial\n\n"
        "Universidad Internacional de La Rioja (UNIR)"
    )