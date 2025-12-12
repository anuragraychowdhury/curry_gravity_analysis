# Stephen Curry Performance Retrieval System

[Presentation Link
](https://docs.google.com/presentation/d/1oU1maW551ph-DnMaDQuyAxFhyrY2CVimrG7nTvOFcXg/edit?usp=sharing)

A two-stage text retrieval system that identifies and ranks passages describing Stephen Curry's notable game performances from sports articles and discussions. The system combines traditional keyword-based retrieval (BM25) with neural semantic re-ranking to provide more relevant results.

## 🎯 Overview

This project implements a hybrid retrieval system that:
1. **Stage 1 (BM25)**: Retrieves top 100 passages using keyword matching
2. **Stage 2 (Neural Re-ranking)**: Re-ranks results using semantic similarity
3. **Evaluation**: Measures performance using Precision@5, Recall@10, and nDCG@10

## 📁 Project Structure

```
.
├── retrieval/                  # Retrieval system module (search)
│   ├── __init__.py
│   ├── retrieval_system.py     # Two-stage retrieval (BM25 + Neural)
│   ├── evaluation.py           # Evaluation metrics for retrieval
│   ├── fetch_data.py           # Fetches text data from Reddit + NewsAPI
│   ├── search_app.py           # Streamlit web interface for search
│   └── run_project.py          # Automated data fetch + quick test
│
├── analysis/                   # Teammate efficiency analysis
│   ├── __init__.py
│   ├── curry_effect.py         # Core stats + regression analysis (Curry teammates)
│   └── analysis_app.py         # Streamlit UI for the analysis
│
├── data/                       # Data directory
│   ├── corpus.json             # Text passages for retrieval system
│   ├── nba_per_game_processed.csv      # Per-game stats for analysis
│   └── nba_2022-23_all_stats.csv       # Advanced stats (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Fetch Data

The system fetches real data from Reddit and NewsAPI:

```bash
python retrieval/fetch_data.py
```

**Options:**
- **Reddit**: No API key needed (press 'y')
- **NewsAPI**: Enter your API key (get free key at https://newsapi.org/)

Or use the automated script with your key already configured:

```bash
python retrieval/run_project.py
```

### 3. Run the Web Interface

```bash
streamlit run retrieval/search_app.py
```

This opens a browser where you can:
- Enter search queries (e.g., "Curry clutch performance")
- Toggle neural re-ranking on/off
- View top-ranked passages with similarity scores

## 🔍 How It Works

### Two-Stage Retrieval Pipeline

1. **BM25 Retrieval (Stage 1)**
   - Fast keyword-based initial retrieval
   - Uses BM25 ranking algorithm
   - Returns top 100 candidate passages

2. **Neural Re-ranking (Stage 2)**
   - Uses sentence transformers for semantic similarity
   - Re-ranks top 100 results from BM25
   - Returns top-k most semantically relevant passages

### Example Query Flow

```
Query: "Curry clutch performance"
    ↓
[BM25] → Top 100 keyword matches
    ↓
[Neural Re-ranking] → Top 10 semantic matches
    ↓
Results with similarity scores
```

## 📊 Features

- ✅ **BM25 Retrieval**: Fast keyword-based initial retrieval
- ✅ **Neural Re-ranking**: Semantic similarity using sentence transformers
- ✅ **Evaluation Metrics**: Precision@5, Recall@10, nDCG@10
- ✅ **Web Interface**: User-friendly Streamlit app
- ✅ **API Integration**: Automatic data fetching from Reddit and NewsAPI
- ✅ **Toggle Comparison**: Compare BM25 vs Neural re-ranking results

## 🛠️ Usage

### Command Line

```python
from retrieval.retrieval_system import RetrievalSystem, load_data

# Load data
passages = load_data('data/corpus.json')

# Initialize system
system = RetrievalSystem(passages)

# Search
results = system.search("Curry clutch performance", top_k=10)

# Display results
for passage, score in results:
    print(f"{score:.3f}: {passage}")
```

### Web Interface

```bash
streamlit run retrieval/search_app.py
```

### Evaluation

```python
from retrieval.evaluation import evaluate_retrieval

# Define relevant passages for a query
relevant = {2, 5, 8}  # Passage indices that are relevant
retrieved = [0, 2, 1, 5, 3, 8, 4, 6, 7, 9]  # Retrieved indices

# Evaluate
metrics = evaluate_retrieval(relevant, retrieved)
print(metrics)
# {'precision@5': 0.4, 'precision@10': 0.3, 'recall@10': 1.0, 'ndcg@10': 0.876}
```

## 📈 Evaluation Metrics

- **Precision@5**: Fraction of top 5 results that are relevant
- **Recall@10**: Fraction of all relevant results found in top 10
- **nDCG@10**: Normalized Discounted Cumulative Gain at 10 (accounts for ranking quality)

## 🔧 Configuration

### NewsAPI Key

Your NewsAPI key is configured in `run_project.py`. To update it:

```python
NEWSAPI_KEY = "your-api-key-here"
```

Get a free key at: https://newsapi.org/register

### Data Sources

- **Reddit**: Fetches from r/nba subreddit (no API key needed)
  - Historical discussions and game threads
  - No date restrictions
- **NewsAPI**: Fetches recent news articles (free tier: 100 requests/day)
  - Recent articles only (last month)
  - Get free key at: https://newsapi.org/register
  - Note: Free tier doesn't allow historical data (2022-23 season)

## 📝 Data Format

Data is stored in `data/corpus.json`:

```json
{
  "passages": [
    "Stephen Curry scored 50 points in a thrilling overtime victory...",
    "Curry's clutch performance in Game 7...",
    ...
  ],
  "metadata": {
    "total_passages": 185,
    "description": "Stephen Curry game performance passages"
  }
}
```

## 🎓 Technical Details

### BM25 Algorithm
- Uses `rank-bm25` library
- Simple tokenization (space-based splitting)
- Fast keyword matching for initial retrieval

### Neural Re-ranking
- Model: `all-MiniLM-L6-v2` (sentence transformer)
- Encodes query and passages into embeddings
- Computes cosine similarity for semantic matching
- Re-ranks top 100 BM25 results

### Why Two-Stage?
- **BM25**: Fast, good for keyword matching
- **Neural**: Slower but captures semantic meaning
- **Combined**: Best of both worlds - fast initial retrieval + semantic re-ranking

## 🐛 Troubleshooting

### "No data found"
- Run `python retrieval/fetch_data.py` to fetch data
- Check that `data/corpus.json` exists

### NewsAPI errors
- Free tier only allows recent articles (last month)
- Historical data (2022-23) not available on free tier
- Reddit data works for historical content

### Streamlit not opening
- Install: `pip install streamlit`
- Check terminal for URL (usually http://localhost:8501)

## 📚 Dependencies

- `rank-bm25`: BM25 ranking algorithm
- `sentence-transformers`: Neural re-ranking
- `streamlit`: Web interface
- `numpy`: Numerical operations
- `requests`: API calls

## 🎯 Project Requirements Met

✅ Two-stage retrieval system (BM25 + Neural re-ranking)  
✅ Evaluation metrics (Precision@5, Recall@10, nDCG@10)  
✅ Streamlit web interface  
✅ API-based data collection (Reddit + NewsAPI)  
✅ Comparison with BM25 baseline (toggle in UI)  

## 📄 License

This project is for educational and demonstration purposes.

---

**Ready to search?** Run `streamlit run simple_app.py` and start querying! 🏀
