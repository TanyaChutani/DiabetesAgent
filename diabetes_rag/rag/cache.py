import json
import os

CACHE_FILE = "pubmed_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}

def get_cache(query):
    return cache.get(query)

def set_cache(query, result):
    cache[query] = result
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)