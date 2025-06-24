import pandas as pd
import folium
from folium.plugins import HeatMap
import joblib
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# 1. Leitura e pré-processamento dos dados
df = pd.read_csv(
    'data/Chicago_Crimes_2001_to_2004.csv',
    on_bad_lines='skip',
    low_memory=False
)
df1 = pd.read_csv(
    'data/Chicago_Crimes_2005_to_2007.csv',
    on_bad_lines='skip',
    low_memory=False
)

df = pd.concat([df, df1], ignore_index=True)

df['Date'] = pd.to_datetime(
    df['Date'],
    format='%m/%d/%Y %I:%M:%S %p',
    errors='coerce'
)
df = df.dropna(subset=['Date', 'Latitude', 'Longitude', 'Primary Type', 'District'])
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Hour'] = df['Date'].dt.hour
df['Weekday'] = pd.Categorical(
    df['Date'].dt.day_name(),
    categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    ordered=True
)

# Dados para predição
model_df = df.dropna(subset=['Primary Type', 'Hour', 'Arrest', 'Domestic', 'District']).copy()
model_df['District'] = model_df['District'].astype(str)
X_full = pd.get_dummies(model_df[['Hour','Arrest','Domestic','District']])
y_full = model_df['Primary Type']
valid_classes = y_full.value_counts()[lambda x: x>=10].index
y_full = y_full[y_full.isin(valid_classes)]
X_full = X_full.loc[y_full.index]

# 2. Carrega o modelo
clf = joblib.load('models/modelo_rf.pkl')

# 3. Inicializa app Dash
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    title="Chicago Crimes Dashboard",
    suppress_callback_exceptions=True
)

# 4. Abas
tabs = dbc.Tabs([
    dbc.Tab(label="Visão Geral", tab_id="tab-overview"),
    dbc.Tab(label="Predição", tab_id="tab-predict")
], id="tabs", active_tab="tab-overview")

overview_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Chicago Crimes (2001–2007)", className="mb-4 text-center"), width=12)),
    dbc.Row([
        dbc.Col([html.Label("Ano:"), dcc.Dropdown(
            id='year-filter',
            options=[{'label': y, 'value': y} for y in sorted(df['Year'].unique())],
            value=df['Year'].min(), clearable=False
        )], width=3),
        dbc.Col([html.Label("Distrito:"), dcc.Dropdown(
            id='district-filter',
            options=[{'label': d, 'value': d} for d in sorted(df['District'].unique())],
            multi=True
        )], width=4)
    ], className="mb-4"),
    dbc.Row([dbc.Col(dcc.Graph(id='top-crimes-bar'), md=6), dbc.Col(dcc.Graph(id='crimes-by-hour'), md=6)]),
    dbc.Row([dbc.Col(dcc.Graph(id='crimes-by-weekday'), md=6),
             dbc.Col(html.Iframe(id='heatmap', srcDoc=None, className="heatmap-frame"), md=6)])
], fluid=True)

predict_layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Previsão de Tipo de Crime", className="mb-4"), width=12)),
    dbc.Row([
        dbc.Col([dbc.Label("Hora (0–23):"), dbc.Input(id='input-hour', type='number', min=0, max=23, value=12)], width=3),
        dbc.Col([dbc.Label("Prisão?"), dbc.Checklist(options=[{'label':'Sim','value':1}], value=[], id='input-arrest', inline=True)], width=3),
        dbc.Col([dbc.Label("Caso Doméstico?"), dbc.Checklist(options=[{'label':'Sim','value':1}], value=[], id='input-domestic', inline=True)], width=3),
        dbc.Col([dbc.Label("Distrito:"), dcc.Dropdown(
            id='input-district',
            options=[{'label': d, 'value': d} for d in sorted(df['District'].unique())],
            value=df['District'].iloc[0], clearable=False
        )], width=3)
    ], className="mb-4"),
    dbc.Row(dbc.Col(dbc.Button("Prever Crime", id='btn-predict', color='primary'), width=2)),
    dbc.Row(dbc.Col(html.Div(id='prediction-output', className="mt-4"), width=12))
], fluid=True)

app.layout = html.Div([tabs, html.Div(id="tab-content")] )

# 5. Callbacks
@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(tab):
    return predict_layout if tab=="tab-predict" else overview_layout

@app.callback(
    Output('top-crimes-bar','figure'), Output('crimes-by-hour','figure'),
    Output('crimes-by-weekday','figure'), Output('heatmap','srcDoc'),
    Input('year-filter','value'), Input('district-filter','value')
)
def update_overview(year, districts):
    dff = df[df['Year']==year]
    if districts:
        dff = dff[dff['District'].isin(districts)]
    top10 = dff['Primary Type'].value_counts().nlargest(10)
    fig1 = px.bar(x=top10.index, y=top10.values, labels={'x':'Crime','y':'Ocorrências'}, title='Top 10 Crimes')
    hour_counts = dff['Hour'].value_counts().sort_index()
    fig2 = px.line(x=hour_counts.index, y=hour_counts.values, labels={'x':'Hora','y':'Ocorrências'}, title='Crimes por Hora')
    wk_counts = dff['Weekday'].value_counts().sort_index()
    fig3 = px.bar(x=wk_counts.index, y=wk_counts.values, labels={'x':'Dia da Semana','y':'Ocorrências'}, title='Crimes por Dia')
    m = folium.Map(location=[41.8781,-87.6298], zoom_start=11)
    heat_data = dff[['Latitude','Longitude']].dropna().sample(min(len(dff),5000)).values.tolist()
    HeatMap(heat_data, radius=12).add_to(m)
    return fig1, fig2, fig3, m._repr_html_()

@app.callback(
    Output('prediction-output','children'), Input('btn-predict','n_clicks'),
    State('input-hour','value'), State('input-arrest','value'),
    State('input-domestic','value'), State('input-district','value')
)
def predict_crime(n_clicks, hour, arrest, domestic, district):
    if not n_clicks:
        return ''
    arrest_flag = 1 if arrest else 0
    domestic_flag = 1 if domestic else 0
    sample = pd.DataFrame([{'Hour':hour,'Arrest':arrest_flag,'Domestic':domestic_flag,'District':district}])
    X_sample = pd.get_dummies(sample).reindex(columns=X_full.columns, fill_value=0)
    pred = clf.predict(X_sample)[0]
    return dbc.Alert(f"Tipo de crime previsto: {pred}", color='info')

if __name__ == '__main__':
    app.run(debug=True)