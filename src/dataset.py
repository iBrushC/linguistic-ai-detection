# Collects the dataset
import json
import requests
from bs4 import BeautifulSoup

# Sample Essays
essays = [
    "https://publicdomainreview.org/essay/jumbos-ghost/",
    "https://publicdomainreview.org/essay/william-wells-brown-wildcat-banker/",
    "https://publicdomainreview.org/essay/race-and-the-white-elephant-war-of-1884/",
    "https://publicdomainreview.org/essay/george-washington-at-the-siamese-court/",
    "https://publicdomainreview.org/essay/the-tale-of-beatrix-potter/",
    "https://publicdomainreview.org/essay/time-and-place-eric-ravilious-1903-1942/",
    "https://publicdomainreview.org/essay/seeing-joyce/",
    "https://publicdomainreview.org/essay/in-search-of-true-color/",
    "https://publicdomainreview.org/essay/the-kept-and-the-killed/",
    "https://publicdomainreview.org/essay/the-mark-of-the-beast-georgian-britains-anti-vaxxer-movement/",
    "https://publicdomainreview.org/essay/the-city-that-fell-off-a-cliff/",
    "https://publicdomainreview.org/essay/the-secret-history-of-holywell-street-home-to-victorian-london-s-dirty-book-trade/",
    "https://publicdomainreview.org/essay/the-lost-world-of-the-london-coffeehouse/",
    "https://publicdomainreview.org/essay/pods-pots-and-potions-putting-cacao-to-paper-in-early-modern-europe/",
    "https://publicdomainreview.org/essay/when-chocolate-was-medicine-colmenero-wadsworth-and-dufour/",
    "https://publicdomainreview.org/essay/mother-gooses-french-birth-1697-and-british-afterlife-1729/"
]

# Fetching essays from links
def fetch_essay(url: str) -> dict:
    r = requests.get(url)
    soup = BeautifulSoup(r.content)

    essay = {}

    # Header
    essay_header = soup.select_one("div.essay-header")
    essay_header_parts = list(essay_header.children)
    essay["title"] = essay_header_parts[1].text
    essay["author"] = essay_header_parts[2].text

    # Intro
    essay_intro = soup.select_one("div.essay-intro")
    essay["intro"] = essay_intro.text[:essay_intro.text.find("Published")]

    # Body
    essay_body = soup.select("div.essay__text-block")
    body_text = "\n".join([b.text for b in essay_body])
    essay["body"] = body_text

    return essay


if __name__ == "__main__":
    # Create the dataset
    extracted_essays = [
        fetch_essay(e) for e in essays
    ]

    # Clean up the data to remove licensing
    for e in extracted_essays:
        e["body"] = e["body"].replace("The text of this essay is published under a CC BY-SA license, see here for details.", "")

    with open("essays.json", "w+") as f:
        f.write(json.dumps(extracted_essays))