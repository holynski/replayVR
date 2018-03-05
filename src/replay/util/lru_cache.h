#include <unordered_map>
#include <vector>

namespace replay {

  template <class Key, class Value>
  class LruCache {
    public:
    LruCache(const int cache_size);
    bool Exists(const Key& key);
    void Insert(const Key& key, Value& value);
    Value& Get(const Key& key);
    private:
    int cache_size_;
    std::unordered_map<Key, Value> cache_;
    std::vector<Key> accesses_;
  };

}
