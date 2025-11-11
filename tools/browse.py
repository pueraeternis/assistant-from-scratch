# tools/browse.py

from typing import Any

import httpx
from bs4 import BeautifulSoup

from core.base_tool import BaseTool


class BrowseTool(BaseTool):
    """
    A tool to browse a webpage and extract its textual content.
    """

    name = "BrowseWebpage"
    description = (
        "Takes a URL as input and returns the clean, textual content of the webpage. "
        "Use this tool when you have a URL from a search result and need to read the full article "
        "to find specific details that are not in the search snippet."
    )

    def _run(self, **kwargs: Any) -> str:
        """
        Fetches the content of a URL (passed as query) and extracts clean text.
        """
        url = kwargs.get("url")
        if not url or not isinstance(url, str):
            return "Error: The 'url' argument is missing or is not a string for BrowseWebpage."
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = httpx.get(url, follow_redirects=True, headers=headers, timeout=30.0)

            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script_or_style.decompose()

            text_chunks = [p.get_text(strip=True) for p in soup.find_all("p")]
            full_text = "\n".join(chunk for chunk in text_chunks if chunk)

            if not full_text:
                return "Could not extract meaningful text from the page. It might be a video, a PDF, or a page without paragraphs."

            max_length = 4000
            return full_text[:max_length] + "..." if len(full_text) > max_length else full_text

        except httpx.RequestError as e:
            return f"Error fetching the URL: {e}"
        except Exception as e:
            return f"An error occurred while browsing the page: {e}"
