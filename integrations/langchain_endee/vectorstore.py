I need write permission to replace the broken file with valid Python code. Please approve the write operation for `integrations/langchain_endee/vectorstore.py`.

The replacement will implement `EndeeVectorStore` with:

- **`similarity_search()`** — new params `prefilter_cardinality_threshold=10_000` and `filter_boost_percentage=0` forwarded to `index.query()`
- **`similarity_search_with_score()`** — same new params, returns `(Document, float)` tuples
- **`add_texts()` / `from_texts()`** — raise `NotImplementedError`
- `filter` key omitted from `index.query()` kwargs when `filter=None`