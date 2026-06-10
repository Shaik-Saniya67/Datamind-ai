# 🧠 DataMind AI — AI Data Insights Chatbot

> FlowZint AI Hackathon 2026 | Open Innovation Category

## 📌 About
DataMind AI is an intelligent data analysis assistant that lets anyone upload a CSV/Excel file and instantly get AI-powered insights, visualizations, and SQL query results — all in plain English.

## 🚀 Features
- 📂 Upload CSV / Excel files
- 💬 Ask questions in plain English
- 📊 Auto-generated charts (histogram, bar, scatter, pie, heatmap, line)
- 🗃 Data Explorer with statistics
- 🔢 SQL Query runner (SQLite)
- 🧠 Powered by Grog AI

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | UI Framework |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Plotly | Interactive charts |
| SQLite | SQL query engine |
| Grog API | AI insights |

## ⚙️ Setup & Run

### 1. Clone / Download the project

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get Grog API Key
- Go to GrogCloud Console
- Sign in with Google
- Click "Get API Key" → "Create API Key"
- Copy the key

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open browser
- Go to http://localhost:8501
- Upload any CSV file
- Start asking questions!

## 🌐 Deploy on Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Deploy — done!

## 📁 Project Structure
```
ai_data_insights/
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
├── utils/
│   ├── data_processor.py  # CSV loading, SQL queries
│   ├── ai_engine.py       # Grog AI integration
│   └── visualizer.py      # Auto chart generation
└── README.md
```

## 👨‍💻 Built for FlowZint AI Hackathon 2026
