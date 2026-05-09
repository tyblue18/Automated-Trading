# Article finder - scrapes google to find articles about a specified topic from a specified source, stops when a year has no articles

# Imports
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

from config import get_config, configure_cli_logging

logger = logging.getLogger(__name__)
configure_cli_logging()

# Find first occurance of a char in a string
def find_first(s, c):
  for i in range(len(s)):
    if(s[i] == c): return i
  return -1

# Build Google News search url according to search string, year, and page
def buildSearch(searchString, year, pageNum=0):
  base = "https://www.google.com/search?q="
  search = searchString.replace(" ", "+")
  date = "+before:" + str(year+1) + "-01-01+after:" + str(year) + "-01-01"
  tags = "&tbs=sbd:1&tbm=nws&start=" + str(pageNum * 10)
  return base + search + date + tags

# Iterate over multiple google search pages
def full_search(searchString, year):
  logger.info("Searching in year %d...", year)
  extracted_links = []
  max_search = 99

  for i in range(max_search):
    # Build search url
    url = buildSearch(searchString, year, i)

    # Get html from url
    response = requests.get(url)
    raw = response.text
    soup = BeautifulSoup(response.content, 'html.parser')

    # Check to make sure request isn't blocked by CAPTCHA
    is_bad_response = re.match(r'<!DOCTYPE [^>]+>', raw)
    if is_bad_response:
      logger.warning("Bad HTML response — possible CAPTCHA block")
      logger.debug("%s", soup.prettify())
      break;

    # Find all elements containing Forbes news article links
    links = soup.find_all(href=re.compile("https://www.forbes.com/sites"))

    # Extract news links and titles
    for link in links:
        href = link.get("href")
        href = href[find_first(href, '=')+1: find_first(href, '&')]
        text = link.get_text()
        if href and href.split('/')[5] == str(year): # Ensure the href attribute exists and is from the correct year
            extracted_links.append(href)

    # Print status
    logger.debug("Extracted %d links from search page %d", len(links), i + 1)
    if len(links) == 0:
      logger.info("Page runout — ending search for year %d", year)
      logger.info("Found %d links for year %d", len(extracted_links), year)
      return extracted_links
    time.sleep(1)

_cfg = get_config()
_SEARCH_QUERY = _cfg["article_search"]["query_template"].format(
    company=_cfg["target_company"]["name"]
)

# Iterate over years to find articles
year = 2025
links = []

while True:
  new_links = full_search(_SEARCH_QUERY, year)
  if len(new_links) > 0: # Add links to list and move on to previous year
    links.extend(new_links)
    year -= 1
  else: # End search if there are no articles found for the year
    logger.info("Year runout — ending search")
    break

# # Find and remove duplicates
# num_links = len(extracted_links)
# extracted_links = set(extracted_links)
# num_dup = num_links - len(extracted_links)

# Print results
for link_data in links:
  # print(link_data['text'])
  logger.debug("%s", link_data)
  # print("")
logger.info("Found %d links total", len(links))

# Export links as CSV
df_final = pd.DataFrame(links, columns=['Link'])
df_final.to_csv('forbes_search.csv', index=True)
df_final.head(10)