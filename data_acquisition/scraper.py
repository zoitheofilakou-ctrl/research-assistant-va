import requests
import time
import json
import os
from dotenv import load_dotenv
load_dotenv()
#ROLE: Data Acquisition Group
#PURPOSE: Automated Metadata Collection from Semantic Scholar API

#Global Configuration
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
BASE_URL = "https://api.semanticscholar.org/graph/v1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if API_KEY is None:
    raise ValueError("SEMANTIC_SCHOLAR_API_KEY not set")


def fetch_rehabilitation_papers(search_query, result_limit=10):
    """
    Retrieves research paper metadata based on specific keywords.
    Filters for recent publications (2020-2024) and extracts key fields for RAG processing.
    """
    search_endpoint = f"{BASE_URL}/paper/search"
    
    #Define search parameters to narrow down from 650k+ generic results
    params = {
        'query': search_query,
        'limit': result_limit,
        'year': '2020-2024',
        #Requesting specific fields required for LLM Screening and RAG Integration
        'fields': 'title,abstract,year,externalIds,url,citationCount'
    }
    
    headers = {'x-api-key': API_KEY}
    
    print(f"[*] Initializing search for query: '{search_query}'")
    
    try:
        response = requests.get(search_endpoint, params=params, headers=headers)
        
        # The approved API key allows 1 request per second.
        # We use 1.1s delay to avoid '429 Too Many Requests' errors.
        time.sleep(1.1) 
        
        if response.status_code == 200:
            search_results = response.json()
            papers_found = search_results.get('data', [])
            print(f"[+] Successfully retrieved {len(papers_found)} papers.")
            return papers_found
        else:
            print(f"[!] API Request Failed. Status Code: {response.status_code}")
            print(f"[!] Error Message: {response.text}")
            return []
            
    except requests.exceptions.RequestException as error:
        print(f"[!] An unexpected system error occurred: {error}")
        return []

if __name__ == "__main__":
    #1.Define three different topics to make the database more diverse
    #Each topic will fetch 20 papers to reach a total of 60.
    search_tasks = [
    "AI assistant healthcare professionals",
    "large language model clinical practice",
    "RAG system medical literature",
    "LLM clinical decision support",
    "evidence-based practice AI researchers",
    "clinical decision support systems",
    "systematic review automation",
    "retrieval augmented generation medical",
    "human oversight AI healthcare",
    "information overload healthcare",
]
    
    all_collected_papers = []
    
    print("=== STARTING MULTI-TOPIC DATA HARVESTING ===")
    
    for keyword in search_tasks:
        #Run the fetch process for each keyword
        papers = fetch_rehabilitation_papers(keyword, result_limit=20)
        all_collected_papers.extend(papers)

    # Deduplication by paperId
    seen_ids = set()
    unique_papers = []
    for paper in all_collected_papers:
        pid = paper.get("paperId")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_papers.append(paper)

    print(f"[*] After dedup: {len(unique_papers)} unique papers (from {len(all_collected_papers)} total)")
    all_collected_papers = unique_papers
        
    #2.Export the combined data to the V3 version JSON file
    if all_collected_papers:
        output_path = os.path.join(BASE_DIR, '..', 'data')
        os.makedirs(output_path, exist_ok=True)

        output_filename = os.path.join(output_path, 'hybrede_metadata_v5.json')
        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(all_collected_papers, json_file, ensure_ascii=False, indent=4)
        
        print("-" * 50)
        print(f"SUCCESS: Total {len(all_collected_papers)} papers exported to '{output_filename}'")
        print("PURPOSE: 10 queries x 20 papers, deduplicated for RAG screening pipeline.")
        print("ACTION: Ready for LLM screening stage.")
        print("-" * 50)
    else:
        print("[!] No data collected. Please check API status or network.")