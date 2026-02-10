import requests
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()

#ROLE: Data Acquisition Group
#PURPOSE: Automated Metadata Collection from Semantic Scholar API

#Global Configuration
API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = "https://api.semanticscholar.org/graph/v1"

def fetch_rehabilitation_papers(search_query, result_limit=20):
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
            
    except Exception as error:
        print(f"[!] An unexpected system error occurred: {error}")
        return []

if __name__ == "__main__":
    # Define a high-relevance query to align with HybReDe project goals
    target_keyword = "clinical knowledge support healthcare professionals"
    
    # Execute the fetch process
    collected_papers = fetch_rehabilitation_papers(target_keyword, result_limit=20)

    # Export the Metadata to a JSON file for the LLM Screening Group
    if collected_papers:
        output_filename = 'data/hybrede_metadata_v2.json'
        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(collected_papers, json_file, ensure_ascii=False, indent=4)
        
        print("-" * 50)
        print(f"SUCCESS: Metadata exported to '{output_filename}'")
        print("ACTION: This file can now be processed by the LLM Reasoning Team.")
        print("-" * 50)