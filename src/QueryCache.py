import os
import pickle
import hashlib
from collections import OrderedDict


class QueryCache:
    def __init__(self, max_size=1000, cache_file=None):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.cache_file = cache_file

        # Load existing cache if file exists
        if cache_file and os.path.exists(cache_file):
            self.load_from_disk()

    def _hash_query(self, query):
        """Create unique hash for query"""
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query):
        """Get cached result if exists, otherwise return None"""
        key = self._hash_query(query)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, query, result):
        """Store result in cache"""
        key = self._hash_query(query)

        if key in self.cache:
            # Update existing entry
            self.cache.move_to_end(key)
        elif len(self.cache) > self.max_size:
            # Remove least recently used item
            self.cache.popitem(last=False)

        self.cache[key] = result

    def save_to_disk(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(dict(self.cache), f)
        except Exception as e:
            print(f"Warning: Failed to save cache to disk: {e}")

    def load_from_disk(self):
        """Load cache from disk"""
        try:
            with open(self.cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)
                self.cache = OrderedDict(loaded_cache)
                print(f"Loaded cache with {len(self.cache)} entries from disk")
        except Exception as e:
            print(f"Warning: Failed to load cache from disk: {e}")
