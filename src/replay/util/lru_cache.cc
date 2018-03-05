#include "replay/util/lru_cache.h"

#include <unordered_map>
#include <vector>

namespace replay {

template <class Key, class Value>
LruCache<Key, Value>::LruCache(int cache_size) : cache_size_(cache_size) {
  cache_.reserve(cache_size);
}

template <class Key, class Value>
bool LruCache<Key, Value>::Exists(const Key& key) {
  return (cache_.find(key) != cache_.end());
}

template <class Key, class Value>
void LruCache<Key, Value>::Insert(const Key& key, Value& value) {
  if (!Exists(key)) {
    // If at capacity, remove the LRU object
    cache_.erase(accesses_[0]);
    accesses_.erase(accesses_.begin());
  }
  // Insert the new object, update the accesses
  cache_[key] = value;
  auto found_key = std::find(accesses_.begin(), accesses_.end(), key);
  if (found_key != accesses_.end()) {
    accesses_.erase(found_key);
  }
  accesses_.push_back(key);
}

template <class Key, class Value>
Value& LruCache<Key, Value>::Get(const Key& key) {
  DCHECK(Exists(key)) << "Requested key doesn't exist.";
  return cache_[key];
}
}
