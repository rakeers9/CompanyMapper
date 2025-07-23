# 📊 AI-Augmented Knowledge Graph and Event Monitoring for Company Impact Analysis

This project uses LLMs to build a structured knowledge graph of a company (e.g., Apple Inc.) from its official documents and monitors real-time news events to assess their impact.

It combines document parsing, knowledge graph construction, interactive visualization, and real-time news + research agents — all through a Streamlit interface.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/rakeers9/CompanyMapper.git
cd CompanyMapper
```

### 2. Set Up Your Environment

Make sure you have Python 3.8+ and a virtual environment.

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Add Your OpenAI Key

Set your OpenAI API key in a `.env` file or export it in your shell:

```bash
export OPENAI_API_KEY=sk-...
```

The code will access this key using:

```python
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

---

## 💡 Using the Streamlit App

Run the app with:

```bash
streamlit run streamlit_app.py
```

Once launched:

### 🧠 Knowledge Graph View
- You will immediately see the **generated company knowledge graph**, parsed from official documents (10-Ks, sustainability reports, supply chain disclosures).
- The graph is fully interactive — click nodes to view metadata, explore relationships, etc.

### 🗞️ Enhanced News Monitoring

1. Open the **left sidebar** and select **Display Mode** > **Enhanced News Monitoring**.
2. Navigate to the **Enhanced Pipeline** tab.

#### Initialization Steps:
- (Optional) Paste your [NewsAPI](https://newsapi.org/) key to improve coverage.
- Click **Initialize Enhanced System**.

#### To Run the News Stream:
1. Set your pipeline parameters (topics, filters, etc.).
2. Add your free API keys:
   - **NewsAPI key** (for real-time news)
   - **Alpha Vantage key** (for financial data enrichment)
3. Press **Enter**, then click **Run Enhanced Stream**.

Once running, the system will:
- Fetch news headlines
- Match them against your knowledge graph
- Trigger a research agent for relevant events
- Output summaries of potential **financial, operational, or reputational** impact

---

## 🔐 Environment Variables Required

You need the following keys in your `.env` file or shell environment:

```dotenv
OPENAI_API_KEY=your_openai_key_here
NEWSAPI_KEY=your_newsapi_key_here      # optional but recommended
ALPHA_VANTAGE_KEY=your_alpha_key_here  # optional but recommended
```

The code automatically loads your OpenAI key using:

```python
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

**Never commit your `.env` file to GitHub.**

---

## 📁 Project Structure

```plaintext
├── knowledge_graph.py         # Extracts entities & builds the graph
├── news_pipeline.py           # Downloads, cleans, and embeds news
├── relevance_detection.py     # Matches news to graph entities
├── research_agent.py          # Triggers detailed research
├── streamlit_app.py           # Interactive frontend UI
├── requirements.txt           # Python dependencies
├── .gitignore                 # Prevents keys, venv, etc. from being tracked
├── README.md                  # This file
```

---

## 📌 Status

✅ Core Milestones Implemented:
- LLM-powered knowledge graph extraction  
- Streamlit-based graph visualization  
- News relevance detection pipeline  
- Triggerable research agent with external APIs  
- JSON + GEXF graph output formats  

---

Feel free to fork, experiment, or extend the system — contributions welcome!
