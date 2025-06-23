# ds.py
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Leitura do dataset
df = pd.read_csv('data/Chicago_Crimes_2001_to_2004.csv', on_bad_lines='skip', low_memory=False)

# 2. Conversão de tipos e limpeza
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

# 3. Criação de colunas úteis
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Hour'] = df['Date'].dt.hour
df['Weekday'] = df['Date'].dt.day_name()

# 4. Análise Exploratória
# Crimes mais comuns
top_crimes = df['Primary Type'].value_counts().head(10)
plt.figure(figsize=(10, 5))
top_crimes.plot(kind='bar', title='Top 10 tipos de crimes (2001–2004)')
plt.ylabel('Ocorrências')
plt.tight_layout()
plt.savefig('figures/crimes_mais_comuns.png')
plt.close()

# Crimes por hora
plt.figure(figsize=(10, 5))
df['Hour'].value_counts().sort_index().plot(kind='line', title='Crimes por hora do dia')
plt.xlabel('Hora')
plt.ylabel('Ocorrências')
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/crimes_por_hora.png')
plt.close()

# Crimes por dia da semana
dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['Weekday'] = pd.Categorical(df['Weekday'], categories=dias, ordered=True)
plt.figure(figsize=(10, 5))
df['Weekday'].value_counts().sort_index().plot(kind='bar', title='Crimes por dia da semana')
plt.ylabel('Ocorrências')
plt.tight_layout()
plt.savefig('figures/crimes_por_dia.png')
plt.close()

# 5. Mapa de calor
map_data = df[['Latitude', 'Longitude']].dropna().sample(10000)
mapa = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
HeatMap(map_data.values.tolist()).add_to(mapa)
mapa.save('figures/mapa_calor.html')

# 6. Modelagem preditiva
# Previsão do tipo de crime com features simples
model_df = df.dropna(subset=['Primary Type', 'Hour', 'Arrest', 'Domestic', 'District'])
model_df = model_df.copy()
model_df['District'] = model_df['District'].astype(str)

X = pd.get_dummies(model_df[['Hour', 'Arrest', 'Domestic', 'District']])
y = model_df['Primary Type']

crime_counts = y.value_counts()
classes_validas = crime_counts[crime_counts >= 10].index
X = X[y.isin(classes_validas)]
y = y[y.isin(classes_validas)]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Relatório do modelo
with open("figures/classificacao.txt", "w") as f:
    f.write("Relatório de Classificação - RandomForestClassifier\n\n")
    f.write(classification_report(y_test, y_pred))

joblib.dump(clf, './models/modelo_rf.pkl')