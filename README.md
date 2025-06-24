# 🔍 Crime Analysis in Chicago (2001–2017)

Análise exploratória, predição e dashboard interativo de crimes na cidade de Chicago, a partir de dados públicos de 2001 a 2017.

---

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como parte da disciplina **CCC269 – Data Science** da UPF, sob orientação do Prof. Dr. Carlos Amaral Hölbig. O pipeline inclui:

1. Aquisição e limpeza de dados  
2. Análise exploratória  
3. Modelagem preditiva (RandomForestClassifier e LightGBM)  
4. Dashboard interativo com Dash  

---

## 🗃️ Dataset

Os dados originais estão em [Kaggle – Crimes in Chicago (currie32)](https://www.kaggle.com/datasets/currie32/crimes-in-chicago).  
Arquivos utilizados (diretório `data/`):

- `Chicago_Crimes_2001_to_2004.csv`  
- `Chicago_Crimes_2005_to_2007.csv`  
- `Chicago_Crimes_2008_to_2011.csv`  
- `Chicago_Crimes_2012_to_2017.csv`  
- `Chicago_Crimes_All.csv` (gerado pelo notebook `Source.ipynb` concatenando todos os anteriores)

### Principais variáveis

| Coluna                   | Descrição                                                    |
|--------------------------|--------------------------------------------------------------|
| `Date`                   | Data e hora do registro (MM/DD/YYYY HH:MM:SS AM/PM)          |
| `Primary Type`           | Categoria principal do crime                                 |
| `Description`            | Descrição detalhada                                          |
| `Location Description`   | Tipo de local (e.g. STREET, RESIDENCE, SCHOOL)               |
| `Arrest`                 | Indicador de prisão (True/False)                             |
| `Domestic`               | Indicador de crime doméstico (True/False)                    |
| `District`               | Distrito policial                                            |
| `Latitude`, `Longitude`  | Coordenadas geográficas                                      |

---

## 🛠️ Tecnologias

- **Python 3.10**  
- Bibliotecas:  
  - **Dados & ML**: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`  
  - **Visualização**: `matplotlib`, `seaborn`, `folium`, `plotly`, `dash`, `dash-bootstrap-components`

---

## 📊 Pipeline

### 1. Pré-processamento (`Source.ipynb`)

- Leitura de múltiplos CSVs com `parse_dates=['Date']` e `on_bad_lines='skip'`  
- Conversão de tipos (`Date`, `Latitude`, `Longitude`)  
- Geração de features temporais: `Year`, `Month`, `Hour`, `Weekday`  
- Saída: `data/Chicago_Crimes_All.csv`  

### 2. Análise Exploratória (`ds.py` / Source.ipynb)

- Top 10 tipos de crime  
- Distribuição por hora do dia  
- Ocorrências por dia da semana  
- Mapa de calor georreferenciado (Folium)  
- **Saída em** `figures/`:  
  - `crimes_mais_comuns.png`  
  - `crimes_por_hora.png`  
  - `crimes_por_dia.png`  
  - `mapa_calor.html`  

### 3. Modelagem Preditiva

#### 3.1 LightGBM (`lgb.ipynb`)

- **Problema**: classificação binária de `Arrest`  
- **Pré-processamento**: colunas categóricas convertidas para `category`  
- **Treinamento**:  
  - Split estratificado 80/20  
  - Parâmetros:  
    - `objective='binary'`  
    - `metric='binary_logloss'`  
    - `learning_rate=0.05`  
    - `num_leaves=31`  
    - `seed=42`  
  - Early stopping (50 rounds)  
  - Log a cada 100 iterações  
- **Avaliação**: acurácia, precisão, recall, F1-score e matriz de confusão exibidas no notebook  
- **Exportação**: `models/modelo_lgb.pkl`  

#### 3.2 RandomForestClassifier (`ds.py`)

- **Problema**: classificação multi-classe de `Primary Type`  
- **Pré-processamento**:  
  - Remover classes com <10 ocorrências  
  - Features: `Hour`, `Arrest`, `Domestic`, `District` (one-hot encoding)  
  - Split estratificado 70/30  
- **Treinamento**:  
  - `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`  
- **Validação**: relatório em `figures/classificacao.txt`  
- **Exportação**: `models/modelo_rf.pkl`  

### 4. Dashboard Interativo (`app.py`)

- **Framework**: Dash + Bootstrap (tema LUX)  
- **Abas**:  
  - **Visão Geral**: filtros por ano e distrito, gráficos e mapa de calor  
  - **Predição**: formulário para entrada de variáveis e previsão do tipo de crime  
- **Execução**:  
  ```bash
  python app.py

---

## 🚀 Como Executar

1. **Clone o repositório:**
   ```sh
   git clone https://github.com/daltonadiers/dataScienceProject.git
   cd dataScienceProject
   ```

2. **Instale as dependências:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Adicione o arquivo de dados em `data/`**  
   Baixe o arquivo `Chicago_Crimes_2001_to_2004.csv` do [Kaggle](https://www.kaggle.com/datasets/currie32/crimes-in-chicago) e coloque na pasta `data/`.

4. **Execute o script principal:**
   ```sh
   python ds.py
   ```

5. **Confira os resultados na pasta `figures/` e se o modelo foi gerado em `modules\`.**

6. **Execute o script para iniciar o servidor com os dashboards**
   ```sh
   python app.py
   ```

7. **Acesse a url**
   ```
   http://127.0.0.1:8050/
   ```

---

## 👥 Autores

- [Dalton Oberdan Adiers](https://github.com/daltonadiers)
- [Henrique Linck Poerschke](https://github.com/OtavioFicagna)
- [Lucas Tomasini Muliterno Friedric](https://github.com/lucasfriedrichh)
- [Otávio Augusto Ficagna](https://github.com/HPoerschke)

---

## 📄 Licença

Este projeto é apenas para fins acadêmicos.

---

## 📚 Referências

- [Crimes in Chicago – Kaggle](https://www.kaggle.com/datasets/currie32/crimes-in-chicago)

---

> Projeto desenvolvido para a disciplina de Data Science – CCC269 – UPF.
