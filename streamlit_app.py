# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# 1. Tytuł
st.title("💡 Fraud Classification App")

# 2. Załaduj model
MODEL_PATH = "model.pkl"
if not os.path.isfile(MODEL_PATH):
    st.error(f"Model not found — umieść '{MODEL_PATH}' w katalogu aplikacji")
    st.stop()

model = joblib.load(MODEL_PATH)

# 3. Wczytywanie danych
st.sidebar.header("Wczytaj dane")
upload = st.sidebar.file_uploader("Plik CSV/Excel z danymi do predykcji", type=['csv', 'xlsx'])

if upload:
    try:
        if upload.name.endswith('.csv'):
            df = pd.read_csv(upload)
        else:
            df = pd.read_excel(upload)
    except Exception as e:
        st.error(f"Błąd odczytu pliku: {e}")
        st.stop()

    st.write("🧾 Dane wejściowe:")
    st.dataframe(df.head())

    # Jeśli są kolumny niezgodne z modelem — alert
    expected = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if expected is not None:
        missing = set(expected) - set(df.columns)
        if missing:
            st.warning(f"Brakuje oczekiwanych cech: {missing}")
            st.stop()
        X = df[expected]
    else:
        X = df  # zakładamy poprawny format

    # 4. Predykcja
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    df['predicted_label'] = y_pred
    if y_prob is not None:
        df['probability_not_fraud'] = y_prob[:, 0]
        df['probability_fraud'] = y_prob[:, 1]

    st.write("🎯 Wyniki predykcji:")
    st.dataframe(df)

else:
    st.write("📌 Załaduj dane przez menu po lewej (plik CSV lub Excel).")

# 5. Ręczne wprowadzanie pojedynczego rekordu
st.sidebar.header("Lub wprowadź pojedynczy rekord")
manual = {}
if expected is not None:
    for feat in expected:
        manual[feat] = st.sidebar.text_input(feat, value="")
    if st.sidebar.button("Predict single record"):
        try:
            row = pd.DataFrame([manual], columns=expected).astype(float)
            pred = model.predict(row)[0]
            result = f"⚠️ Fraud" if pred == 1 else "✅ Not Fraud"
            st.sidebar.write(f"Predykcja: **{result}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(row)[0]
                st.sidebar.write(f"Prawdopodobieństwo: not_fraud = {probs[0]:.3f}, fraud = {probs[1]:.3f}")
        except Exception as e:
            st.sidebar.error(f"Błąd: {e}")
