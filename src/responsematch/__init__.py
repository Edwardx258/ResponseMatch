"""ResponseMatch: match emergency / 311 service requests to volunteers.

A LangGraph workflow parses each request, retrieves handbook guidance from
ChromaDB, scores candidates on two independent tracks (rules and an LLM
judge), and only recommends volunteers both tracks agree on.
"""

__version__ = "0.1.0"
