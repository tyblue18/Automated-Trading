# Article scraper - scrapes forbes articles for article title, author, publication date, and content

# Imports
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random

logger = logging.getLogger(__name__)

# Get critical elements from article webpage
def scrape(url, wait_times=[1,5,30], printErrors = True):
  # Run until scrape is successful
  completed = False
  errors = 0
  while(not completed):
    # Get html and scrape elements
    try:
      response = requests.get(url)
      soup = BeautifulSoup(response.content, 'html.parser')
      title = soup.find_all('h1')[0].text.strip()
      date = soup.find_all('time')[0].text.strip()
      p = [elem.text.strip() for elem in soup.find_all('p')]
      author = p[0][2:-1]
      body = "\n".join(p[2:])
      completed = True

    # Print html and wait a bit if there's an error
    except Exception as e:
      errors += 1
      logger.exception(f"({errors}) Scraping error")
      if(printErrors): print(soup.prettify())
      completed = False
      if(len(wait_times) > 1): time.sleep(wait_times[errors])
      else: time.sleep(wait_times[0])

      # Skip article if the article cannot be scraped
      if errors >= len(wait_times)-1:
        print("(!) Skipping article...")
        return "0", "0", "0", "0"

  time.sleep(wait_times[0])
  return title, date, author, body

# Open CSV containing links
df = pd.DataFrame(pd.read_csv("forbes_search.csv"))
links = df['Link'].tolist()
numLinks = len(links)

data = [["0"] for i in range(numLinks)]

# Loop through articles until all articles have been scraped
passNum = 1
while(True):
  for i in range(numLinks):
    if data[i][0] == "0":
      print(f"Scraping article {i+1} of {numLinks}")
      dataAquired = False
      data[i] = scrape(links[i], [random.uniform(1,3)], False)

  # Post-pass
  passNum += 1
  aquired = 0
  for i in range(numLinks):
    if data[i][0] != "0": aquired += 1
  print(f"Scraped {aquired} out of {numLinks} articles")
  if aquired == numLinks: break
  print(f"Waiting before starting pass {passNum}...")
  time.sleep(5)
print("Scraping Complete")

# Export data as CSV
df_final = pd.DataFrame(data, columns=['Title', 'Time', 'Author', 'Body'])
df_final.insert(0, 'Link', df['Link'])
df_final.to_csv('forbes_articles.csv', index=True)
df_final.head(10)