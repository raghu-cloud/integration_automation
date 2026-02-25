try:
    from crewai_tools import BaseTool
except ImportError:
    class BaseTool:
        """Fallback stub when crewai_tools is not installed."""
        name: str = ""
        description: str = ""

        def _run(self):
            raise NotImplementedError

from pydantic import BaseModel, Field
from typing import List, Optional


class EndeeSearchInput(BaseModel):
    vector: List[float] = Field(..., description="Query vector for similarity search")
    top_k: int = Field(10, description="Number of results to return")
    filter: Optional[dict] = Field(None, description="Optional metadata filter")
    prefilter_cardinality_threshold: int = Field(
        10000,
        ge=1000,
        le=1_000_000,
        description="Threshold for switching between prefilter and postfilter (1k–1M)",
    )
    filter_boost_percentage: int = Field(
        0,
        ge=0,
        le=100,
        description="Percentage boost applied to filtered results (0–100)",
    )


class EndeeSearchTool(BaseTool):
    name: str = "EndeeSearch"
    description: str = "Performs vector similarity search using the endee index."

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def _run(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
        prefilter_cardinality_threshold: int = 10000,
        filter_boost_percentage: int = 0,
    ) -> list:
        kwargs = dict(
            vector=vector,
            top_k=top_k,
            prefilter_cardinality_threshold=prefilter_cardinality_threshold,
            filter_boost_percentage=filter_boost_percentage,
        )
        if filter is not None:
            kwargs["filter"] = filter
        return self.index.query(**kwargs)
