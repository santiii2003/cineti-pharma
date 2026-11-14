import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="CinetiPharma AI - UdeA", layout="wide")
st.title("🧪 Herramienta Inteligente de Análisis Cinético y Estabilidad Farmacéutica")
st.caption("Fisicoquímica CIFAL - UdeA 2026-2 | Método Integral y Predicción de Arrhenius")

# --- BARRA LATERAL: CONFIGURACIÓN E IA ---
with st.sidebar:
    st.header("⚙️ Configuración General")
    unidad_tiempo = st.selectbox("Unidad de tiempo", ["horas", "días", "meses", "minutos"])
    unidad_conc = st.selectbox("Unidad de concentración", ["mg/mL", "%", "mol/L", "µg/mL"])
    
    st.divider()
    st.header("🤖 Módulo de IA (Conclusiones)")
    api_key = st.text_input("API Key de Gemini (opcional)", type="password", 
                            help="Si se deja vacío, la app usará el motor experto local integrado.")

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2 = st.tabs(["1️⃣ Análisis Cinético (Método Integral)", "2️⃣ Módulo Arrhenius y Predicción"])

# ==========================================
# PESTAÑA 1: DETERMINACIÓN DE ORDEN CINÉTICO
# ==========================================
with tab1:
    st.subheader("Ingreso de Datos Experimentales")
    datos_ejemplo = pd.DataFrame({
        "Tiempo": [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0],
        "Concentracion": [100.0, 94.1, 88.5, 83.3, 78.4, 69.4, 48.2]
    })
    
    df_input = st.data_editor(datos_ejemplo, num_rows="dynamic", use_container_width=True)
    df = df_input.dropna().sort_values(by="Tiempo")
    
    if len(df) >= 3 and (df["Concentracion"] > 0).all():
        t = df["Tiempo"].values
        A = df["Concentracion"].values
        lnA = np.log(A)
        invA = 1.0 / A
        
        # Regresiones lineales para los 3 órdenes
        reg0 = linregress(t, A)
        reg1 = linregress(t, lnA)
        reg2 = linregress(t, invA)
        
        modelos = {
            "Orden Cero": {"reg": reg0, "y": A, "ylabel": f"[A] ({unidad_conc})", "k": -reg0.slope, "A0": reg0.intercept},
            "Orden Uno":  {"reg": reg1, "y": lnA, "ylabel": "ln[A]", "k": -reg1.slope, "A0": np.exp(reg1.intercept)},
            "Orden Dos":  {"reg": reg2, "y": invA, "ylabel": f"1/[A] (1/{unidad_conc})", "k": reg2.slope, "A0": 1.0 / reg2.intercept}
        }
        
        # Determinación automática del mejor orden según R^2
        mejor_orden = max(modelos.keys(), key=lambda m: modelos[m]["reg"].rvalue**2)
        mejor_r2 = modelos[mejor_orden]["reg"].rvalue**2
        
        st.success(f"🎯 **Modelo Cinético Más Probable:** {mejor_orden} ($R^2 = {mejor_r2:.5f}$)")
        
        # Generación automática de los 3 gráficos
        cols = st.columns(3)
        for idx, (nombre, info) in enumerate(modelos.items()):
            reg = info["reg"]
            r2 = reg.rvalue**2
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=info["y"], mode='markers', name='Experimental', marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=t, y=reg.intercept + reg.slope * t, mode='lines', name='Ajuste Lineal'))
            fig.update_layout(
                title=f"{nombre}<br><sup>m = {reg.slope:.5f} | b = {reg.intercept:.4f} | R² = {r2:.5f}</sup>",
                xaxis_title=f"Tiempo ({unidad_tiempo})",
                yaxis_title=info["ylabel"],
                height=350, margin=dict(l=20, r=20, t=60, b=20), showlegend=False
            )
            cols[idx].plotly_chart(fig, use_container_width=True)
            
        # Cálculo de parámetros cinéticos
        st.subheader("📊 Parámetros Cinéticos Calculados")
        orden_seleccionado = st.selectbox("Orden para cálculo de tiempos (seleccionado automáticamente):", 
                                          list(modelos.keys()), index=list(modelos.keys()).index(mejor_orden))
        
        k_val = modelos[orden_seleccionado]["k"]
        # Se usa [A]0 experimental si t=0 existe, de lo contrario el intercepto del modelo
        A0_val = A[0] if t[0] == 0 else modelos[orden_seleccionado]["A0"]
        
        if k_val > 0:
            if orden_seleccionado == "Orden Cero":
                t50 = (0.50 * A0_val) / k_val
                t98 = (0.02 * A0_val) / k_val
                t95 = (0.05 * A0_val) / k_val
                t90 = (0.10 * A0_val) / k_val
            elif orden_seleccionado == "Orden Uno":
                t50 = np.log(2) / k_val
                t98 = -np.log(0.98) / k_val
                t95 = -np.log(0.95) / k_val
                t90 = -np.log(0.90) / k_val
            else: # Orden Dos
                t50 = 1.0 / (k_val * A0_val)
                t98 = ((1.0 / 0.98) - 1.0) / (k_val * A0_val)
                t95 = ((1.0 / 0.95) - 1.0) / (k_val * A0_val)
                t90 = ((1.0 / 0.90) - 1.0) / (k_val * A0_val)
                
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Constante k", f"{k_val:.5f}")
            m2.metric(f"t50 ({unidad_tiempo})", f"{t50:.2f}")
            m3.metric(f"t98 ({unidad_tiempo})", f"{t98:.2f}")
            m4.metric(f"t95 ({unidad_tiempo})", f"{t95:.2f}")
            m5.metric(f"t90 - Vida útil ({unidad_tiempo})", f"{t90:.2f}")
            
            # Guardar en sesión para usar en pestaña 2
            st.session_state["orden_activo"] = orden_seleccionado
            st.session_state["A0_activo"] = A0_val
        else:
            st.warning("La constante cinética k resultó negativa o cero. Verifique que la concentración disminuya con el tiempo.")
    else:
        st.error("Ingrese al menos 3 puntos con concentraciones mayores a cero.")

# ==========================================
# PESTAÑA 2: MÓDULO ARRHENIUS Y PREDICCIÓN
# ==========================================
with tab2:
    st.subheader("Análisis de Dependencia Térmica (Ecuación de Arrhenius)")
    
    datos_arrhenius = pd.DataFrame({
        "Temperatura_C": [25.0, 40.0, 50.0, 60.0],
        "k": [0.0045, 0.0210, 0.0560, 0.1420]
    })
    
    df_arr_in = st.data_editor(datos_arrhenius, num_rows="dynamic", use_container_width=True)
    df_arr = df_arr_in.dropna()
    
    if len(df_arr) >= 2 and (df_arr["k"] > 0).all():
        df_arr["T_K"] = df_arr["Temperatura_C"] + 273.15
        df_arr["inv_T"] = 1.0 / df_arr["T_K"]
        df_arr["ln_k"] = np.log(df_arr["k"])
        
        st.dataframe(df_arr[["Temperatura_C", "T_K", "inv_T", "k", "ln_k"]].style.format({
            "Temperatura_C": "{:.1f}", "T_K": "{:.2f}", "inv_T": "{:.6f}", "k": "{:.6f}", "ln_k": "{:.4f}"
        }), use_container_width=True)
        
        reg_arr = linregress(df_arr["inv_T"], df_arr["ln_k"])
        R_const = 8.314 # J/(mol*K)
        Ea_J = -reg_arr.slope * R_const
        Ea_kJ = Ea_J / 1000.0
        A_factor = np.exp(reg_arr.intercept)
        r2_arr = reg_arr.rvalue**2
        
        col_g, col_p = st.columns([1.2, 1])
        with col_g:
            fig_arr = go.Figure()
            fig_arr.add_trace(go.Scatter(x=df_arr["inv_T"], y=df_arr["ln_k"], mode='markers', name='Experimental', marker=dict(size=9)))
            fig_arr.add_trace(go.Scatter(x=df_arr["inv_T"], y=reg_arr.intercept + reg_arr.slope * df_arr["inv_T"], mode='lines', name='Ajuste Arrhenius'))
            fig_arr.update_layout(
                title=f"Gráfico de Arrhenius: ln(k) vs 1/T<br><sup>Pendiente = {reg_arr.slope:.2f} K | R² = {r2_arr:.5f}</sup>",
                xaxis_title="1/T (K⁻¹)", yaxis_title="ln(k)", height=360
            )
            st.plotly_chart(fig_arr, use_container_width=True)
            
        with col_p:
            st.metric("Energía de Activación (Ea)", f"{Ea_kJ:.2f} kJ/mol", f"{Ea_J:.1f} J/mol")
            st.metric("Factor Preexponencial (A)", f"{A_factor:.4e}")
            st.metric("Linealidad Arrhenius (R²)", f"{r2_arr:.5f}")
            
        st.divider()
        st.subheader("🔮 Predicción de Estabilidad a Otras Temperaturas")
        
        c1, c2, c3 = st.columns(3)
        temp_pred = c1.number_input("Temperatura a predecir (°C)", value=5.0, step=1.0)
        orden_pred = c2.selectbox("Orden cinético aplicable", ["Orden Cero", "Orden Uno", "Orden Dos"],
                                  index=["Orden Cero", "Orden Uno", "Orden Dos"].index(st.session_state.get("orden_activo", "Orden Uno")))
        A0_pred = c3.number_input(f"Concentración inicial [A]0 ({unidad_conc})", value=float(st.session_state.get("A0_activo", 100.0)))
        
        T_pred_K = temp_pred + 273.15
        k_pred = A_factor * np.exp(-Ea_J / (R_const * T_pred_K))
        
        if orden_pred == "Orden Cero":
            t50_p, t98_p, t95_p, t90_p = (0.50*A0_pred)/k_pred, (0.02*A0_pred)/k_pred, (0.05*A0_pred)/k_pred, (0.10*A0_pred)/k_pred
        elif orden_pred == "Orden Uno":
            t50_p, t98_p, t95_p, t90_p = np.log(2)/k_pred, -np.log(0.98)/k_pred, -np.log(0.95)/k_pred, -np.log(0.90)/k_pred
        else:
            t50_p = 1.0 / (k_pred * A0_pred)
            t98_p = ((1.0 / 0.98) - 1.0) / (k_pred * A0_pred)
            t95_p = ((1.0 / 0.95) - 1.0) / (k_pred * A0_pred)
            t90_p = ((1.0 / 0.90) - 1.0) / (k_pred * A0_pred)
            
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric(f"k predicha ({temp_pred}°C)", f"{k_pred:.6f}")
        p2.metric(f"t50 ({unidad_tiempo})", f"{t50_p:.2f}")
        p3.metric(f"t98 ({unidad_tiempo})", f"{t98_p:.2f}")
        p4.metric(f"t95 ({unidad_tiempo})", f"{t95_p:.2f}")
        p5.metric(f"t90 ({unidad_tiempo})", f"{t90_p:.2f}")
        
        # Generación automática de conclusiones mediante IA
        st.subheader("🧠 Conclusiones Automáticas de Estabilidad (IA)")
        if st.button("Generar Análisis y Recomendaciones"):
            sensibilidad = "alta sensibilidad térmica (requiere estricto control de cadena de frío)" if Ea_kJ > 85 else \
                           "sensibilidad térmica moderada (típica de reacciones hidrolíticas/oxidativas en solución)" if Ea_kJ >= 40 else \
                           "baja barrera energética (degradación rápida incluso a bajas temperaturas)"
            
            conclusion_base = (
                f"**Diagnóstico Cinético y Térmico:**\n"
                f"* **Ajuste del Modelo:** La degradación sigue una cinética de **{orden_pred}** con una energía de activación de **{Ea_kJ:.2f} kJ/mol** ($R^2 = {r2_arr:.4f}$), lo que indica una {sensibilidad}.\n"
                f"* **Proyección a {temp_pred} °C:** La constante de velocidad estimada es $k = {k_pred:.6f}\\text{{ {unidad_tiempo}}}^{{-1}}$, alcanzando el límite de especificación farmacéutica ($t_{{90}}$) en **{t90_p:.2f} {unidad_tiempo}** y su vida media ($t_{{50}}$) en **{t50_p:.2f} {unidad_tiempo}**.\n"
                f"* **Recomendación Operacional:** Mantener el producto protegido de excursiones térmicas por encima de {temp_pred} °C y verificar compatibilidad de pH y excipientes para minimizar la velocidad de degradación."
            )
            
            if api_key:
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    prompt = (
                        f"Actúa como experto en estabilidad farmacéutica. Redacta 3 conclusiones técnicas concisas sobre un fármaco "
                        f"con cinética de {orden_pred}, Ea = {Ea_kJ:.2f} kJ/mol, A = {A_factor:.3e}, y que a {temp_pred}°C presenta "
                        f"k = {k_pred:.6f} {unidad_tiempo}^-1, t90 = {t90_p:.2f} {unidad_tiempo} y t50 = {t50_p:.2f} {unidad_tiempo}."
                    )
                    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.info(resp.text)
                except Exception as e:
                    st.warning(f"No se pudo conectar con la API externa ({e}). Mostrando diagnóstico local:")
                    st.info(conclusion_base)
            else:
                st.info(conclusion_base)
