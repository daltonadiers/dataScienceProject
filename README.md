# 🔍 Crime Analysis in Chicago (2001–2004)

Análise exploratória e predição de crimes na cidade de Chicago, com base em dados públicos registrados pela polícia entre 2001 e 2004.

---

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como parte do curso **CCC269 – Data Science**, orientado pelo **Prof. Dr. Carlos Amaral Hölbig**. O objetivo foi aplicar um pipeline completo de Ciência de Dados, desde a aquisição e limpeza dos dados até a modelagem preditiva e visualização.

**Tema livre** escolhido: **Crime Prediction and Analysis in Chicago**

---

## 🗃️ Dataset

**Origem oficial:**  
🔗 [Crimes in Chicago – Kaggle (por currie32)](https://www.kaggle.com/datasets/currie32/crimes-in-chicago)

**Arquivo utilizado neste projeto:**  
- `Chicago_Crimes_2001_to_2004.csv`

**Total de registros:** 1.923.515 crimes reportados

### 📄 Principais variáveis do dataset:

| Coluna                | Descrição |
|-----------------------|-----------|
| `Date`                | Data e hora do crime |
| `Primary Type`        | Tipo principal de crime (ex: THEFT, BATTERY) |
| `Description`         | Descrição detalhada |
| `Location Description`| Tipo de local (rua, casa, escola, etc.) |
| `Arrest`              | Houve prisão? (True/False) |
| `Domestic`            | Foi um crime doméstico? (True/False) |
| `District`            | Número do distrito policial |
| `Latitude`, `Longitude` | Coordenadas do local |
| `Community Area`, `Ward`, `FBI Code`, `Beat`, etc. | Outras características administrativas e geográficas |

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**
- `pandas`, `numpy` – Manipulação de dados
- `matplotlib`, `seaborn` – Visualizações
- `folium` – Mapas e mapas de calor
- `scikit-learn` – Modelagem preditiva
- `Jupyter` ou execução via script (`ds.py`)

---

## 📊 Etapas do Projeto

### ✅ 1. Carregamento e limpeza dos dados
- Leitura do CSV com `on_bad_lines='skip'`
- Conversão de colunas para tipos corretos
- Remoção de valores nulos
- Criação de colunas derivadas: `Year`, `Month`, `Hour`, `Weekday`

### ✅ 2. Análise Exploratória
- Top 10 crimes mais comuns (`Primary Type`)
- Distribuição de crimes por hora do dia
- Dias da semana com mais ocorrências
- Localização geográfica dos crimes com mapa de calor

### ✅ 3. Visualizações geradas:
- 📊 `figures/crimes_mais_comuns.png`
- 🕒 `figures/crimes_por_hora.png`
- 📆 `figures/crimes_por_dia.png`
- 🗺️ `figures/mapa_calor.html`

### ✅ 4. Modelagem preditiva
- **Objetivo:** Prever o tipo de crime com base em:
  - Hora do dia
  - Distrito policial
  - Se houve prisão
  - Se foi um crime doméstico

- **Modelo usado:** `RandomForestClassifier` (Scikit-learn)

- **Pré-processamento adicional:**
  - Classes com menos de 10 ocorrências foram removidas
  - Dados foram divididos em treino e teste (70/30), com `stratify`

- **Resultado:** Relatório salvo em `figures/classificacao.txt` com métricas de desempenho

---

## 🚀 Como Executar

1. **Clone o repositório:**
   ```
   git clone https://github.com/daltonadiers/dataScienceProject.git
   cd dataScienceProject
   ```

2. **Instale as dependências:**
   ```
   pip install pandas numpy matplotlib seaborn folium scikit-learn
   ```

3. **Adicione o arquivo de dados em `data/`**  
   Baixe o arquivo `Chicago_Crimes_2001_to_2004.csv` do [Kaggle](https://www.kaggle.com/datasets/currie32/crimes-in-chicago) e coloque na pasta `data/`.

4. **Execute o script principal:**
   ```
   python ds.py
   ```

5. **Confira os resultados na pasta `figures/`.**

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
