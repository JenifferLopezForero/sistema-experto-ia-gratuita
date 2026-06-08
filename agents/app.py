from agents.base_agent_universal import ExpertAgent
import streamlit as st
import os
import pandas as pd

st.set_page_config(
    page_title="Sistema Experto IA Gratuita",
    layout="wide"
)

with st.sidebar:
    st.title("⚙️ Configuración")
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Proveedor: Groq (LLaMA 3)")
    elif os.getenv("GEMINI_API_KEY"):
        st.success("✅ Proveedor: Google Gemini Flash")
    elif os.getenv("HF_TOKEN"):
        st.info("ℹ️ Proveedor: HuggingFace")
    else:
        st.error("❌ Sin proveedor configurado")
    st.markdown("---")
    st.markdown("**Links útiles:**")
    st.markdown("- [Groq (gratis)](https://console.groq.com)")
    st.markdown("- [Google Gemini](https://aistudio.google.com)")
    st.markdown("- [HuggingFace](https://huggingface.co)")

RULES = [
    {
        "id": "R1",
        "cond": lambda h: h.get("fiebre", 0) > 38 and h.get("tos", False),
        "conclusion": "Posible infección respiratoria"
    },
    {
        "id": "R2",
        "cond": lambda h: h.get("fiebre", 0) > 39,
        "conclusion": "Fiebre alta — requiere atención médica"
    },
    {
        "id": "R3",
        "cond": lambda h: h.get("dolor_cabeza", False) and h.get("fiebre", 0) > 37.5,
        "conclusion": "Posible síndrome gripal"
    },
    {
        "id": "R4",
        "cond": lambda h: not h.get("fiebre", 0) > 37.5 and h.get("tos", False),
        "conclusion": "Tos sin fiebre — posible alergia o irritación"
    },
    {
        "id": "R5",
        "cond": lambda h: h.get("presion_alta", False),
        "conclusion": "Hipertensión detectada — revisar medicación"
    },
]

SYSTEM_PROMPT = (
    "Eres un asistente médico experto. "
    "Recibirás hechos clínicos y conclusiones de un motor de reglas. "
    "Genera un diagnóstico claro, recomendaciones y advertencias. "
    "Responde siempre en español. Sé conciso pero completo."
)

DATASET_PATH = "data/pacientes.csv"
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATASET_PATH):
    pd.DataFrame({
        "id":           [1, 2, 3],
        "nombre":       ["Paciente A", "Paciente B", "Paciente C"],
        "fiebre":       [38.5, 37.0, 39.5],
        "tos":          [True, True, False],
        "dolor_cabeza": [True, False, True],
        "presion_alta": [False, False, True]
    }).to_csv(DATASET_PATH, index=False)

@st.cache_resource
def cargar_agente():
    return ExpertAgent(rules=RULES, dataset_path=DATASET_PATH)

try:
    agente = cargar_agente()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

st.title("🧠 Sistema Experto con IA Gratuita")
st.markdown(
    f"_Sin pagos · Sin tarjeta · "
    f"Backend: **{agente.backend.upper()}** · Modelo: **{agente.model}**_"
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Ingresar Hechos del Caso")
    fiebre  = st.slider("Temperatura corporal (°C)", 35.0, 42.0, 37.0, 0.1)
    tos     = st.checkbox("¿Presenta tos?")
    dolor   = st.checkbox("¿Dolor de cabeza?")
    presion = st.checkbox("¿Presión arterial alta?")
    hechos  = {
        "fiebre":       fiebre,
        "tos":          tos,
        "dolor_cabeza": dolor,
        "presion_alta": presion
    }
    analizar = st.button("🔍 Analizar con Sistema Experto", type="primary")

with col2:
    st.subheader("📊 Dataset Cargado")
    st.dataframe(agente.df, use_container_width=True)

if analizar:
    st.divider()
    with st.spinner("Ejecutando motor de reglas e IA..."):
        conclusiones, fired = agente.forward_chain(hechos)
        respuesta = agente.query_agent(hechos, conclusiones, SYSTEM_PROMPT)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⚙️ Motor de Reglas")
        if fired:
            st.success(f"Reglas activadas: {', '.join(fired)}")
            for c in conclusiones:
                st.write(f"✅ {c}")
        else:
            st.info("Ninguna regla fue activada.")
    with c2:
        st.subheader("🤖 Diagnóstico IA")
        st.markdown(respuesta)

    with st.expander("🔎 Ver hechos ingresados"):
        st.json(hechos)
