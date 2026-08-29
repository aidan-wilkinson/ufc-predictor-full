from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from pathlib import Path

url = "http://ufcstats.com/statistics/events/completed?page=all"

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

output_path = data_dir / "event_links.txt"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(15000)

    html = page.content()

    browser.close()

soup = BeautifulSoup(html, "lxml")

event_links = [
    link["href"]
    for link in soup.find_all(
        "a",
        class_="b-link b-link_style_black"
    )
]

print(f"{len(event_links)} events found")

for link in event_links[:3]:
    print(link)

# Save the links
with open(output_path, "w") as f:
    for link in event_links:
        f.write(link + "\n")

print(f"Saved event links to {output_path}")