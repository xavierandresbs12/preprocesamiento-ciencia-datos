import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def limpiar_datos(df):
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    return df

def normalizar_datos(df, columnas):
    scaler = MinMaxScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df

def codificar_categorias(df, columnas):
    encoder = LabelEncoder()
    for col in columnas:
        df[col] = encoder.fit_transform(df[col])
    return df

print("Preprocesamiento completado correctamente")