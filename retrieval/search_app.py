"""
Streamlit app for the retrieval (search) system.
"""
import os
import sys

import streamlit as st

# Ensure project root is on sys.path so `retrieval` package can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.retrieval_system import RetrievalSystem, load_data

st.set_page_config(page_title="Curry Search", layout="wide")

st.title("🏀 Stephen Curry Performance Search")
st.markdown("**Two-stage retrieval:** BM25 → Neural Re-ranking")

# Load data
@st.cache_resource
def load_system():
    passages = load_data('data/corpus.json')
    if passages:
        return RetrievalSystem(passages), passages
    return None, []

system, passages = load_system()

if not system:
    st.warning("⚠️ No data found! Please create `data/corpus.json`")
    st.code("""
{
  "passages": [
    "Stephen Curry scored 50 points...",
    "Curry's clutch performance...",
    ...
  ]
}
    """)
else:
    st.success(f"✓ Loaded {len(passages)} passages")
    
    # Search interface
    query = st.text_input("Enter your search query", placeholder="e.g., Curry clutch performance")
    
    col1, col2 = st.columns(2)
    with col1:
        use_reranking = st.checkbox("Use Neural Re-ranking", value=True)
    with col2:
        top_k = st.slider("Number of results", 5, 20, 10)
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            results = system.search(query, use_reranking=use_reranking, top_k=top_k)
            
            st.subheader(f"Top {len(results)} Results")
            
            for i, (passage, score) in enumerate(results, 1):
                st.markdown(f"**{i}. Score: {score:.3f}**")
                st.write(passage)
                st.divider()


