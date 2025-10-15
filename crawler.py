import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import time
import joblib
import os
import re

class SmartCrawler:
    def __init__(self, model_path='classifier.pkl', vectorizer_path='vectorizer.pkl'):
        # Bypasses Cloudflare blocks
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
        )
        self.model = None
        self.vectorizer = None
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("ML Intelligence Loaded Successfully.")
        else:
            print(f"Warning: Model files not found. Run trainer.py first.")

    def predict_relevance(self, text):
        if not self.model: return True
        # Clean the text to match training format
        cleaned = re.sub(r'\s+', ' ', text).strip()
        vec = self.vectorizer.transform([cleaned])
        return bool(self.model.predict(vec)[0])

    def crawl(self, seed_url, max_pages=10):
        to_visit = [seed_url]
        visited = set()
        results = []

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited: continue
            
            try:
                print(f"Requesting: {url}")
                res = self.scraper.get(url, timeout=20)
                if res.status_code != 200: 
                    print(f" -> Failed (Status {res.status_code})")
                    continue

                soup = BeautifulSoup(res.text, 'html.parser')
                
                # Target the actual problem text to avoid "menu/sidebar" noise
                problem_div = soup.find('div', class_='problem-statement')
                text_to_analyze = problem_div.get_text() if problem_div else soup.get_text()

                if self.predict_relevance(text_to_analyze):
                    print(" -> Match found! Relevant content saved.")
                    visited.add(url)
                    results.append({"url": url, "snippet": text_to_analyze[:300].strip()})
                    
                    # Look for more links
                    for a in soup.find_all('a', href=True):
                        link = a['href']
                        if link.startswith('/problemset/problem/'):
                            full_link = "https://codeforces.com" + link
                            if full_link not in visited:
                                to_visit.append(full_link)
                else:
                    print(" -> Irrelevant content. Skipping.")
                    visited.add(url)

            except Exception as e:
                print(f"Error at {url}: {e}")
            
            print("Waiting 7 seconds (Politeness Delay)...")
            time.sleep(7) 
        
        return results

if __name__ == "__main__":
    crawler = SmartCrawler()
    # Using first page of cf problemset as the starting point
    seed = "https://codeforces.com/problemset/page/1"
    crawled_data = crawler.crawl(seed, max_pages=50)
    
    if crawled_data:
        pd.DataFrame(crawled_data).to_csv("crawled_results.csv", index=False)
        print(f"\nSuccess! Saved {len(crawled_data)} pages to 'crawled_results.csv'")
    else:
        print("\nCrawl complete, but no new relevant pages were discovered.")