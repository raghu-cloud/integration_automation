All 12 tests pass. The fixes were:

1. `EndeeSearchInput.vector` renamed to `query_vector` — the schema validation tests also use `query_vector`
2. `_run(vector=...)` parameter renamed to `query_vector` — matches how tests call `tool._run(query_vector=...)`
3. Inside `_run`, the kwarg passed to `self.index.query()` remains `vector=query_vector` — matching what the mock asserts (`kwargs["vector"]`)