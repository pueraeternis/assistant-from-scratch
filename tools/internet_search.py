# tools/internet_search.py

from typing import Any

from ddgs import DDGS

from core.base_tool import BaseTool


class InternetSearchTool(BaseTool):
    """
    A tool to search the internet using DuckDuckGo.
    Ideal for finding real-time or recent information.
    """

    name = "InternetSearch"
    description = (
        "Use this tool to search the internet for current information. "
        "Provide a clear and concise search query as input. "
        "For example, to find the latest news, you could input 'latest technology news'."
    )

    def _run(self, **kwargs: Any) -> str:
        """
        Performs an internet search using the provided query and returns
        a formatted string of the top search results.
        """
        query = kwargs.get("query")
        if not query or not isinstance(query, str):
            return "Error: The 'query' argument is missing or is not a string for InternetSearchTool."

        ddgs_client = DDGS()
        # We'll take the top 3-5 results to avoid overloading the LLM context
        results = list(ddgs_client.text(query, max_results=4))  # pyright: ignore[reportAttributeAccessIssue]
        if not results:
            return "No relevant results found for your query."

        # Format the results into a string that's easy for the LLM to parse
        formatted_results = []
        for i, res in enumerate(results, 1):
            snippet = (
                f"Result [{i}]:\n" f"Title: {res['title']}\n" f"Snippet: {res['body']}\n" f"Source: {res['href']}\n"
            )
            formatted_results.append(snippet)

        return "---\n".join(formatted_results)
