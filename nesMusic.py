import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Used chatgpt to generate a scraper

# STEP 1: Set up base URL and output folder
base_url = "https://www.vgmusic.com/music/console/nintendo/nes/"  # replace this with actual NES MIDI page
output_dir = "nes_midis"
os.makedirs(output_dir, exist_ok=True)

# STEP 2: Get page HTML
response = requests.get(base_url)

soup = BeautifulSoup(response.text, "html.parser")

# STEP 3: Find all .mid/.midi links
midi_links = []
for link in soup.find_all("a", href=True):
    href = link["href"]
    if href.endswith(".mid") or href.endswith(".midi"):
        midi_links.append(urljoin(base_url, href))

# STEP 4: Limit to first 2000 links
midi_links = midi_links[:2000]

# STEP 5: Download the files
for i, url in enumerate(midi_links):
    filename = os.path.join(output_dir, f"{i+1:03d}_{os.path.basename(url)}")
    try:
        print(f"Downloading {url}...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
