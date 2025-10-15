from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from crawler import SmartCrawler
from bs4 import BeautifulSoup
import json
import asyncio

app = FastAPI()

# Crucial: Allows your Vercel frontend to talk to your Render backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

crawler = SmartCrawler()

@app.get("/crawl")
async def start_crawl(url: str = Query(...), limit: int = 10):
    async def event_generator():
        to_visit = [url]
        visited = set()
        found_count = 0

        while to_visit and found_count < limit:
            curr_url = to_visit.pop(0)
            if curr_url in visited: continue
            visited.add(curr_url)

            try:
                # Use a separate thread for the blocking scraper call
                res = await asyncio.to_thread(crawler.scraper.get, curr_url, timeout=15)
                if res.status_code != 200: continue

                soup = BeautifulSoup(res.text, 'html.parser')
                prob_div = soup.find('div', class_='problem-statement')
                text = prob_div.get_text() if prob_div else soup.get_text()

                is_relevant = crawler.predict_relevance(text)
                
                # Send update to frontend
                yield f"data: {json.dumps({'url': curr_url, 'relevant': is_relevant, 'snippet': text[:150]})}\n\n"

                if is_relevant:
                    found_count += 1
                    # Extract new links
                    for a in soup.find_all('a', href=True):
                        if '/problemset/problem/' in a['href']:
                            full = "https://codeforces.com" + a['href'].split('?')[0]
                            if full not in visited: to_visit.append(full)

                # Wait between requests to avoid 403 blocks
                await asyncio.sleep(7)
                
            except Exception as e:
                yield f"data: {json.dumps({'url': curr_url, 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")