I need your approval to overwrite `tools.py`. The file currently contains prose text (not Python), causing the `SyntaxError`. The replacement I've prepared:

- Defines `EndeeSearchInput` (Pydantic model) with the required fields and validation constraints:
  - `prefilter_cardinality_threshold`: default `10_000`, min `1_000`, max `1_000_000`
  - `filter_boost_percentage`: default `0`, min `0`, max `100`
- Defines `EndeeSearchTool` with `_run()` that calls `self.index.query()` and omits the `filter` key when it's `None`
- Has a graceful fallback stub when `crewai_tools` isn't installed

Please approve the write so I can fix the file.