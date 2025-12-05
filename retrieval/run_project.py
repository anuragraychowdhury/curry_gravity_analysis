"""
Quick script to fetch data and test the retrieval system.
"""
import os
import sys

# Ensure project root is on sys.path so `retrieval` package can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.fetch_data import main as fetch_main
from retrieval.retrieval_system import RetrievalSystem, load_data

# Your NewsAPI key (get free key at https://newsapi.org/register)
# Replace this with your own key, or set it as an environment variable
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "YOUR_API_KEY_HERE")

def setup_and_test():
    """Fetch data and test the retrieval system."""
    print("=" * 60)
    print("STEP 1: Fetching Data from Reddit and NewsAPI")
    print("=" * 60)
    
    # Fetch data from both APIs
    passages = fetch_main(reddit=True, newsapi_key=NEWSAPI_KEY)
    
    if not passages:
        print("\n❌ No data fetched. Please check your API key or internet connection.")
        return
    
    print("\n" + "=" * 60)
    print("STEP 2: Testing the Retrieval System")
    print("=" * 60)
    
    # Load the saved data
    passages = load_data('data/corpus.json')
    print(f"\n✓ Loaded {len(passages)} passages from data/corpus.json")
    
    # Initialize the retrieval system
    print("\nInitializing retrieval system...")
    system = RetrievalSystem(passages)
    print("✓ System ready!")
    
    # Test queries
    test_queries = [
        "Curry clutch performance",
        "Curry three-point shooting",
        "Curry scoring record"
    ]
    
    print("\n" + "=" * 60)
    print("STEP 3: Testing Search Queries")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        
        # Search with re-ranking
        results = system.search(query, use_reranking=True, top_k=3)
        
        for i, (passage, score) in enumerate(results, 1):
            print(f"\n{i}. [Score: {score:.3f}]")
            print(f"   {passage[:150]}..." if len(passage) > 150 else f"   {passage}")
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the web interface: streamlit run retrieval/search_app.py")

if __name__ == "__main__":
    setup_and_test()

