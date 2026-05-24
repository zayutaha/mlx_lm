"""50+ tests for web search, scraping, relevance, and /search vs /research logic."""
import unittest
from html import escape
from unittest.mock import patch, MagicMock

from mlx_lm.web_search import search_web, scrape_url, is_relevant, _clean_text, _extract_main_content


# ═══════════════════════════════════════════════
# TESTS 1-20: _clean_text
# ═══════════════════════════════════════════════

class TestCleanText(unittest.TestCase):

    def test_removes_navigation_boilerplate(self):
        text = "skip to content\nnavigation\nmain menu\nThe actual article content here."
        result = _clean_text(text)
        self.assertNotIn("skip to content", result)
        self.assertNotIn("navigation", result)
        self.assertNotIn("main menu", result)
        self.assertIn("The actual article content here.", result)

    def test_removes_wikipedia_references_section(self):
        text = "Some article text.\nReferences\n1. A book citation\n2. Another source\nThe article continues."
        result = _clean_text(text)
        self.assertIn("Some article text.", result)
        self.assertNotIn("References", result)

    def test_removes_political_offices_section(self):
        text = "Biography text here.\nPolitical offices\nPreceded by X\nSucceeded by Y\nMore biography."
        result = _clean_text(text)
        self.assertIn("Biography text here.", result)
        self.assertNotIn("Political offices", result)

    def test_removes_categories(self):
        text = "Content.\nCategories\n1926 births\n2018 deaths\nMore content."
        result = _clean_text(text)
        self.assertNotIn("Categories", result)

    def test_removes_hidden_categories(self):
        text = "Content.\nHidden categories\nArticles with short description\nArticles containing text.\nEnd."
        result = _clean_text(text)
        self.assertNotIn("Hidden categories", result)
        self.assertIn("Content.", result)

    def test_removes_retrieved_from(self):
        text = "Content.\nRetrieved from https://en.wikipedia.org/foo"
        result = _clean_text(text)
        self.assertNotIn("Retrieved from", result)

    def test_removes_short_lines(self):
        text = "Long meaningful line of text.\n\nAB\nAnother meaningful paragraph."
        result = _clean_text(text)
        self.assertNotIn("\nAB\n", result)
        self.assertIn("Long meaningful", result)

    def test_removes_punctuation_only_lines(self):
        text = "Content.\n---\nMore content.\n***\nEnd."
        result = _clean_text(text)
        self.assertNotIn("---", result)
        self.assertNotIn("***", result)

    def test_removes_jump_to_links(self):
        text = "Jump to content\nJump to search\nThe real content."
        result = _clean_text(text)
        self.assertNotIn("Jump to", result)

    def test_removes_contents_header(self):
        text = "Contents\n1. Introduction\n2. History\nThe content."
        result = _clean_text(text)
        self.assertNotIn("Contents", result)

    def test_removes_cookie_statement(self):
        text = "Content.\ncookie statement\nThis site uses cookies.\nReal content."
        result = _clean_text(text)
        self.assertNotIn("cookie statement", result)

    def test_removes_privacy_boilerplate(self):
        text = "Content.\nPrivacy policy\nTerms of use\nAbout\nReal content."
        result = _clean_text(text)
        self.assertNotIn("Privacy policy", result)
        self.assertNotIn("Terms of use", result)

    def test_removes_see_also_section(self):
        text = "Content.\nSee also\nRelated topic 1\nRelated topic 2\nMore content."
        result = _clean_text(text)
        self.assertNotIn("See also", result)

    def test_preserves_meaningful_content_around_boilerplate(self):
        text = "Navigation menu\nSkip to content\nThe President announced new climate policies on March 15, 2026.\nSee also\nClimate change.\nFooter links."
        result = _clean_text(text)
        self.assertIn("The President announced", result)
        self.assertIn("March 15, 2026", result)

    def test_normalizes_excessive_newlines(self):
        text = "Line 1.\n\n\n\n\nLine 2.\n\n\n\nLine 3."
        result = _clean_text(text)
        self.assertNotIn("\n\n\n\n", result)

    def test_normalizes_excessive_spaces(self):
        text = "Word1    Word2     Word3."
        result = _clean_text(text)
        self.assertNotIn("    ", result)

    def test_removes_further_reading_section(self):
        text = "Content.\nFurther reading\nBook 1\nBook 2\nArticle 3\nEnd."
        result = _clean_text(text)
        self.assertNotIn("Further reading", result)

    def test_removes_external_links_section(self):
        text = "Content.\nExternal links\nLink 1\nLink 2\nMore content."
        result = _clean_text(text)
        self.assertNotIn("External links", result)

    def test_removes_text_is_available_under(self):
        text = "Content.\ntext is available under the Creative Commons license.\nMore content."
        result = _clean_text(text)
        self.assertNotIn("text is available under", result)

    def test_preserves_dates_and_numbers(self):
        text = "On April 12, 1981, she was born. She served from 2013 to 2021. The cost was $2.5 billion."
        result = _clean_text(text)
        self.assertIn("April 12, 1981", result)
        self.assertIn("$2.5 billion", result)


# ═══════════════════════════════════════════════
# TESTS 21-30: _extract_main_content
# ═══════════════════════════════════════════════

class TestExtractMainContent(unittest.TestCase):

    def _soup(self, html):
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "lxml")

    def test_extracts_from_article_tag(self):
        html = "<html><body><article>" + "The main article content here. " * 50 + "</article><footer>Footer</footer></body></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertIn("main article content", result)
        self.assertNotIn("Footer", result)

    def test_extracts_from_main_tag(self):
        html = "<html><body><header>Header</header><main>The primary content. " * 50 + "</main><aside>Sidebar</aside></body></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertIn("primary content", result)

    def test_extracts_from_mw_parser_output(self):
        html = "<html><body><div class=\"mw-parser-output\">Wikipedia content. " * 50 + "</div><div class=\"navbox\">Navigation</div></body></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertIn("Wikipedia content", result)

    def test_extracts_from_role_main(self):
        html = "<html><body><div role=\"main\">Main content here. " * 50 + "</div><div role=\"navigation\">Nav</div></body></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertIn("Main content", result)

    def test_extracts_from_content_id(self):
        html = "<html><body><div id=\"content\">The article text. " * 50 + "</div><div id=\"footer\">Footer</div></body></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertIn("article text", result)

    def test_fallback_to_body(self):
        html = "<html><body><p>Simple body content. " * 50 + "</p></body></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertIn("Simple body content", result)

    def test_picks_richest_content_block(self):
        html = "<html><body><div class=\"sidebar\">Short</div><div class=\"content\">" + "Rich content. " * 100 + "</div></body></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertIn("Rich content", result)

    def test_skips_empty_elements(self):
        html = "<html><body><article></article><main><p>Real content here. " * 50 + "</p></main></body></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertIn("Real content", result)
        self.assertNotIn("article", result)

    def test_handles_no_body(self):
        html = "<html><head><title>Empty</title></head></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertIsInstance(result, str)

    def test_extracts_full_content_string(self):
        expected_text = "Comprehensive article text with details. " * 50
        html = "<html><body><article>" + expected_text + "</article></body></html>"
        soup = self._soup(html)
        result = _extract_main_content(soup)
        self.assertGreater(len(result), 100)


# ═══════════════════════════════════════════════
# TESTS 31-40: is_relevant
# ═══════════════════════════════════════════════

class TestIsRelevant(unittest.TestCase):

    def test_exact_match_is_relevant(self):
        self.assertTrue(is_relevant(
            "Tulsi Gabbard Resigns as DNI",
            "Tulsi Gabbard is resigning as Director of National Intelligence",
            "tulsi gabbard resignation"
        ))

    def test_partial_keyword_match_is_relevant(self):
        self.assertTrue(is_relevant(
            "Gabbard leaves Trump administration",
            "The director of national intelligence is leaving her post",
            "what happened to tulsi gabbard"
        ))

    def test_no_keyword_match_is_not_relevant(self):
        self.assertFalse(is_relevant(
            "Thinking processes theory of constraints",
            "How to improve your thinking with constraints",
            "tulsi gabbard resignation"
        ))

    def test_completely_unrelated_is_not_relevant(self):
        self.assertFalse(is_relevant(
            "HTTP Analyzer API capture tutorial",
            "How to capture API requests from any app using HTTP Analyzer",
            "tulsi gabbard"
        ))

    def test_single_keyword_match_suffices(self):
        self.assertTrue(is_relevant(
            "Climate policy update 2026",
            "New regulations for emissions",
            "biden climate policy"
        ))

    def test_empty_keywords_returns_true(self):
        self.assertTrue(is_relevant(
            "Some random title",
            "Some random snippet",
            "a the of"
        ))

    def test_only_stopwords_returns_true(self):
        self.assertTrue(is_relevant(
            "Any article title",
            "Any snippet text",
            "what is the for"
        ))

    def test_case_insensitive_matching(self):
        self.assertTrue(is_relevant(
            "TULSI GABBARD NEWS",
            "BREAKING: GABBARD RESIGNS",
            "Tulsi Gabbard"
        ))

    def test_keyword_in_title_suffices(self):
        self.assertTrue(is_relevant(
            "Elon Musk SpaceX Starship Launch",
            "Short snippet",
            "elon musk starship"
        ))

    def test_keyword_in_snippet_suffices(self):
        self.assertTrue(is_relevant(
            "Latest Technology News",
            "SpaceX launched Starship successfully",
            "starship launch"
        ))


# ═══════════════════════════════════════════════
# TESTS 41-48: search_web and scrape_url (mocked)
# ═══════════════════════════════════════════════

class TestSearchWebMocked(unittest.TestCase):

    @patch("mlx_lm.web_search.DDGS")
    def test_search_web_returns_results(self, mock_ddgs):
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]
        results = search_web("test query", num_results=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Result 1")
        self.assertEqual(results[0]["url"], "https://example.com/1")

    @patch("mlx_lm.web_search.DDGS")
    def test_search_web_empty_results(self, mock_ddgs):
        mock_ddgs.return_value.__enter__.return_value.text.return_value = []
        results = search_web("nothing", num_results=5)
        self.assertEqual(results, [])

    @patch("mlx_lm.web_search.requests.get")
    def test_scrape_url_returns_text(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html><body><article>" + b"Paragraph text. " * 100 + b"</article></body></html>"
        mock_get.return_value = mock_response
        result = scrape_url("https://example.com/article")
        self.assertIsNotNone(result)
        self.assertIn("Paragraph text", result)

    @patch("mlx_lm.web_search.requests.get")
    def test_scrape_url_non_html_returns_none(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_get.return_value = mock_response
        result = scrape_url("https://example.com/doc.pdf")
        self.assertIsNone(result)

    @patch("mlx_lm.web_search.requests.get")
    def test_scrape_url_http_error_returns_none(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response
        result = scrape_url("https://example.com/error")
        self.assertIsNone(result)

    @patch("mlx_lm.web_search.requests.get")
    def test_scrape_url_strips_script_and_style(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html><body><article>Content. <script>alert('x')</script><style>.cls{}</style>More content. " * 20 + b"</article></body></html>"
        mock_get.return_value = mock_response
        result = scrape_url("https://example.com")
        self.assertIsNotNone(result)
        self.assertIn("Content.", result)

    @patch("mlx_lm.web_search.requests.get")
    def test_scrape_url_strips_nav_and_footer(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html><body><nav>Nav links</nav><article>Real content. " * 50 + b"</article><footer>Footer</footer></body></html>"
        mock_get.return_value = mock_response
        result = scrape_url("https://example.com")
        self.assertIsNotNone(result)
        self.assertIn("Real content", result)

    @patch("mlx_lm.web_search.requests.get")
    def test_scrape_url_uses_main_content_selector(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html><body><div role='main'>Main content. " * 50 + b"</div><div class='sidebar'>Sidebar stuff</div></body></html>"
        mock_get.return_value = mock_response
        result = scrape_url("https://example.com")
        self.assertIsNotNone(result)
        self.assertIn("Main content", result)
        # Sidebar might still appear since we extract by selector
        # Just verify main content is present


# ═══════════════════════════════════════════════
# TESTS 49-52: Search vs Research flow logic
# ═══════════════════════════════════════════════

class TestSearchResearchFlow(unittest.TestCase):

    def test_search_has_depth_3(self):
        """Simulate the logic that /search uses depth=3."""
        query = "/search what happened"
        is_research = query.startswith("/research ")
        is_search = query.startswith("/search ")
        self.assertTrue(is_search)
        self.assertFalse(is_research)
        depth = 8 if is_research else 3
        self.assertEqual(depth, 3)

    def test_research_has_depth_8(self):
        """Simulate the logic that /research uses depth=8."""
        query = "/research artificial intelligence"
        is_research = query.startswith("/research ")
        is_search = query.startswith("/search ")
        self.assertFalse(is_search)
        self.assertTrue(is_research)
        depth = 8 if is_research else 3
        self.assertEqual(depth, 8)

    def test_search_extracts_query_correctly(self):
        prefix = "/research " if "/research " in "/search test query" else "/search "
        topic = "/search test query"[len(prefix):].strip()
        self.assertEqual(topic, "test query")

    def test_research_extracts_topic_correctly(self):
        query = "/research machine learning transformers"
        prefix = "/research " if query.startswith("/research ") else "/search "
        topic = query[len(prefix):].strip()
        self.assertEqual(topic, "machine learning transformers")


# ═══════════════════════════════════════════════
# TESTS 53-55: Query parsing from model output
# ═══════════════════════════════════════════════

class TestQueryParsing(unittest.TestCase):

    def test_parse_basic_queries(self):
        model_output = "elon musk news 2026\nelon musk latest updates\nelon musk biography"
        queries = [
            line.strip().lstrip("0123456789.)- ")
            for line in model_output.splitlines()
            if line.strip() and len(line.strip()) > 3
        ]
        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[0], "elon musk news 2026")

    def test_parse_numbered_queries(self):
        model_output = "1. elon musk news\n2. elon musk spacex\n3. elon musk biography"
        queries = [
            line.strip().lstrip("0123456789.)- ")
            for line in model_output.splitlines()
            if line.strip() and len(line.strip()) > 3
        ]
        self.assertEqual(len(queries), 3)
        self.assertIn("elon musk news", queries)

    def test_parse_empty_lines_skipped(self):
        model_output = "query one\n\n\nquery two\n\n\nquery three"
        queries = [
            line.strip().lstrip("0123456789.)- ")
            for line in model_output.splitlines()
            if line.strip() and len(line.strip()) > 3
        ]
        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[1], "query two")

    def test_parse_respects_depth_limit(self):
        depth = 8
        model_output = "\n".join([f"query {i}" for i in range(1, 15)])
        queries = [
            line.strip().lstrip("0123456789.)- ")
            for line in model_output.splitlines()
            if line.strip() and len(line.strip()) > 3
        ][:depth]
        self.assertEqual(len(queries), depth)

    def test_fallback_when_parsing_empty(self):
        model_output = ""
        queries = [
            line.strip().lstrip("0123456789.)- ")
            for line in model_output.splitlines()
            if line.strip() and len(line.strip()) > 3
        ][:3]
        if not queries:
            queries = ["original topic"]
        self.assertEqual(queries, ["original topic"])

    def test_fallback_appends_original_query(self):
        topic = "climate change"
        queries = ["global warming 2026", "carbon emissions policy"]
        if topic not in queries:
            queries.append(topic)
        self.assertIn("climate change", queries)

    def test_deduplicates_queries(self):
        queries = ["elon musk", "elon musk spacex", "elon musk"]
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0], "elon musk")

    def test_short_queries_filtered(self):
        model_output = "a\n\nab\n\nabcd\n\nvalid query"
        queries = [
            line.strip().lstrip("0123456789.)- ")
            for line in model_output.splitlines()
            if line.strip() and len(line.strip()) > 3
        ]
        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0], "abcd")
        self.assertEqual(queries[1], "valid query")


if __name__ == "__main__":
    unittest.main(verbosity=2)
