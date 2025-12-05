"""
Fetch real NBA/Curry data from APIs instead of hardcoded data.
"""
import json
import requests
import time
from typing import List

def fetch_from_newsapi(query: str, api_key: str, from_date: str = None, to_date: str = None) -> List[str]:
    """
    Fetch articles from NewsAPI.
    
    Get free API key at: https://newsapi.org/
    Free tier: 100 requests/day, only last month of articles
    
    Args:
        query: Search query (e.g., "Stephen Curry NBA")
        api_key: Your NewsAPI key
        from_date: Start date (YYYY-MM-DD) - defaults to 30 days ago
        to_date: End date (YYYY-MM-DD) - defaults to today
    
    Returns:
        List of article descriptions/excerpts
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': api_key,
        'pageSize': 100  # Max per request
    }
    
    # Only add dates if provided (free tier has date restrictions)
    if from_date and to_date:
        params['from'] = from_date
        params['to'] = to_date
    
    passages = []
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        for article in data.get('articles', []):
            # Extract relevant text
            description = article.get('description', '')
            content = article.get('content', '')
            
            # Filter for Curry-related content
            text = description or content
            if text and ('curry' in text.lower() or 'warriors' in text.lower()):
                # Split into sentences (simple approach)
                sentences = text.split('. ')
                for sentence in sentences:
                    if len(sentence) > 50 and 'curry' in sentence.lower():
                        passages.append(sentence.strip())
        
        print(f"Fetched {len(passages)} passages from NewsAPI")
    except Exception as e:
        print(f"Error fetching from NewsAPI: {e}")
    
    return passages

def fetch_from_reddit(subreddit: str = "nba", query: str = "Stephen Curry", limit: int = 100) -> List[str]:
    """
    Fetch posts from Reddit (no API key needed for basic use).
    
    Uses Reddit's public JSON API (no authentication required for read-only).
    
    Args:
        subreddit: Subreddit name (default: "nba")
        query: Search query
        limit: Number of posts to fetch
    
    Returns:
        List of post texts/comments
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        'q': query,
        'limit': limit,
        'sort': 'relevance',
        'restrict_sr': '1'
    }
    headers = {'User-Agent': 'CurryRetrievalBot/1.0'}
    
    passages = []
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        for post in data.get('data', {}).get('children', []):
            post_data = post.get('data', {})
            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '')
            
            text = f"{title}. {selftext}"
            if 'curry' in text.lower() and len(text) > 50:
                # Split into sentences
                sentences = text.split('. ')
                for sentence in sentences:
                    if len(sentence) > 50 and 'curry' in sentence.lower():
                        passages.append(sentence.strip())
        
        print(f"Fetched {len(passages)} passages from Reddit")
        time.sleep(2)  # Be respectful with rate limiting
    except Exception as e:
        print(f"Error fetching from Reddit: {e}")
    
    return passages

def save_passages(passages: List[str], output_file: str = 'data/corpus.json'):
    """Save passages to JSON file."""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    data = {
        'passages': passages,
        'metadata': {
            'total_passages': len(passages),
            'description': 'Stephen Curry game performance passages from 2022-23 NBA season'
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(passages)} passages to {output_file}")

def main(reddit=True, newsapi_key=None):
    """
    Main function to fetch data from Reddit and NewsAPI.
    
    Args:
        reddit: Whether to fetch from Reddit (default: True)
        newsapi_key: NewsAPI key (optional)
    """
    print("Fetching NBA/Curry Data from APIs")
    print("=" * 50)
    
    all_passages = []
    
    # Option 1: Reddit (no API key needed)
    if reddit:
        print("\n1. Fetching from Reddit...")
        passages = fetch_from_reddit("nba", "Stephen Curry", limit=100)
        all_passages.extend(passages)
        print(f"✓ Got {len(passages)} passages from Reddit\n")
    
    # Option 2: NewsAPI (requires API key)
    if newsapi_key:
        print("2. Fetching from NewsAPI...")
        passages = fetch_from_newsapi("Stephen Curry NBA Warriors", newsapi_key)
        all_passages.extend(passages)
        print(f"✓ Got {len(passages)} passages from NewsAPI\n")
        time.sleep(1)  # Rate limiting
    
    # Remove duplicates
    unique_passages = list(dict.fromkeys(all_passages))  # Preserves order
    
    if unique_passages:
        save_passages(unique_passages)
        print(f"\n✓ Total unique passages: {len(unique_passages)}")
        print(f"✓ Data saved to data/corpus.json")
        return unique_passages
    else:
        print("\n⚠ No passages fetched.")
        return []

if __name__ == "__main__":
    main()


