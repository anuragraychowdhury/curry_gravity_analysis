# Project Structure

## 📁 Directory Organization

```
.
├── retrieval/                    # BM25 + Neural re-ranking search system
│   ├── __init__.py              # Module exports
│   ├── retrieval_system.py      # Two-stage retrieval (BM25 + Neural)
│   └── evaluation.py            # Evaluation metrics (Precision, Recall, nDCG)
│
├── analysis/                     # Curry teammate efficiency analysis
│   ├── __init__.py
│   ├── curry_effect.py          # Core stats/regression analysis
│   └── analysis_app.py          # Streamlit UI for analysis
│
├── data/                        # Data directory
│   ├── corpus.json              # Passage dataset (for retrieval)
│   ├── nba_per_game_processed.csv   # Per-game stats (main analysis input)
│   └── nba_2022-23_all_stats.csv    # Advanced stats (optional)
│
├── fetch_data.py                # Retrieval: fetches text data from Reddit/NewsAPI
├── simple_app.py                # Retrieval: Streamlit search interface
├── run_project.py               # Retrieval: automated data fetch + smoke test
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
└── PROJECT_STRUCTURE.md         # This file
```

## 🎯 Component Overview

### `retrieval/` - Retrieval System Module

**Purpose**: Contains the two-stage retrieval system (BM25 + Neural re-ranking)

**Files**:
- `retrieval_system.py`: Main retrieval class combining BM25 and neural re-ranking
- `evaluation.py`: Evaluation metrics (Precision@5, Recall@10, nDCG@10)
- `__init__.py`: Module exports for easy importing

**Usage**:
```python
from retrieval import RetrievalSystem, load_data

passages = load_data()
system = RetrievalSystem(passages)
results = system.search("query", top_k=10)
```

### Root Level Files

- **`fetch_data.py`**: Retrieval – fetches text data from Reddit and NewsAPI  
- **`simple_app.py`**: Retrieval – Streamlit web interface for searching  
- **`run_project.py`**: Retrieval – automated script to fetch data and test system  
- **`requirements.txt`**: Python package dependencies  
- **`README.md`**: Complete project documentation  
- **`PROJECT_STRUCTURE.md`**: This structure overview  

## 🔄 Data Flow

```
1. fetch_data.py
   ↓
   [Reddit API] + [NewsAPI]
   ↓
   data/corpus.json (185 passages)
   ↓
2. retrieval/retrieval_system.py
   ↓
   [BM25] → Top 100
   ↓
   [Neural Re-ranking] → Top 10
   ↓
3. simple_app.py
   ↓
   User sees results in web interface
```

## 📦 Dependencies

All dependencies are in `requirements.txt`:
- `rank-bm25`: BM25 ranking algorithm
- `sentence-transformers`: Neural re-ranking
- `streamlit`: Web interface
- `numpy`: Numerical operations
- `requests`: API calls

## 🚀 Quick Start

1. **Install**: `pip install -r requirements.txt`
2. **Fetch Data**: `python fetch_data.py` or `python run_project.py`
3. **Run App**: `streamlit run simple_app.py`

## 📝 Key Features

✅ **Modular Design**: Retrieval system in separate folder  
✅ **Clean Structure**: Organized by functionality  
✅ **Easy Imports**: Simple `from retrieval import ...`  
✅ **Comprehensive Docs**: README + API guide  

