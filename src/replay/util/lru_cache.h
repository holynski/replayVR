#pragma once

#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace replay {

template <class Key, class Value>
class LruCache {
 public:
  LruCache(const int cache_size);
  bool Exists(const Key& key) const;
  void Insert(const Key& key, Value& value);
  const Value& Get(const Key& key) const;

 private:
  int cache_size_;
  std::unordered_map<Key, Value> cache_;
  std::vector<Key> accesses_;
};

template <class Key, class Value>
LruCache<Key, Value>::LruCache(int cache_size) : cache_size_(cache_size) {
  cache_.reserve(cache_size);
}

template <class Key, class Value>
bool LruCache<Key, Value>::Exists(const Key& key) const {
  return (cache_.find(key) != cache_.end());
}

template <class Key, class Value>
void LruCache<Key, Value>::Insert(const Key& key, Value& value) {
  if (!Exists(key)) {
    // If at capacity, remove the LRU object
    if (cache_.size() > cache_size_) {
      cache_.erase(accesses_[0]);
      accesses_.erase(accesses_.begin());
    }
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
const Value& LruCache<Key, Value>::Get(const Key& key) const {
  DCHECK(Exists(key)) << "Requested key doesn't exist.";
  return cache_.at(key);
}

}  // namespace replay
