from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import threading
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

links_path = data_dir / "event_links.txt"
events_output_path = data_dir / "event_details.csv"
fight_links_output_path = data_dir / "fight_links.txt"

MAX_WORKERS = 6  # each worker runs its own browser instance — keep modest, this is memory-heavier than a requests-based pool


# ============================================================
# LOAD LINKS + EXISTING DATA
# ============================================================

with open(links_path, "r") as f:
    event_links = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(event_links)} event links")

if events_output_path.exists():
    existing_events_df = pd.read_csv(events_output_path)
    scraped_event_ids = set(existing_events_df["event_id"].astype(str))
    print(f"Found {len(existing_events_df)} existing fight records across previously-scraped events")
else:
    existing_events_df = pd.DataFrame()
    scraped_event_ids = set()

if fight_links_output_path.exists():
    with open(fight_links_output_path, "r") as f:
        existing_fight_links = set(line.strip() for line in f if line.strip())
    print(f"Found {len(existing_fight_links)} existing fight links")
else:
    existing_fight_links = set()

events_to_scrape = [
    (idx, link) for idx, link in enumerate(event_links)
    if link[-16:] not in scraped_event_ids
]

print(f"{len(events_to_scrape)} events remaining to scrape")


# ============================================================
# SHARED STATE (thread-safe accumulation)
# ============================================================

winner_names = []
new_fight_links_all = []
_lock = threading.Lock()


# ============================================================
# PER-THREAD BROWSER MANAGEMENT
# ============================================================
# Playwright's sync API requires all calls for a given browser/page to
# happen on the thread that created them. Rather than sharing one browser
# across threads (which breaks), each worker thread lazily starts its own
# Playwright instance + browser + page on first use, and reuses it for
# every event that thread processes.

_thread_local = threading.local()
_active_playwrights = []  # tracked for cleanup at the end
_active_lock = threading.Lock()


def get_thread_page():
    if not hasattr(_thread_local, "page"):
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        _thread_local.pw = pw
        _thread_local.browser = browser
        _thread_local.page = page

        with _active_lock:
            _active_playwrights.append((pw, browser))

    return _thread_local.page


def cleanup_all_browsers():
    for pw, browser in _active_playwrights:
        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass


# ============================================================
# EVENT SCRAPER
# ============================================================

def get_event_data(idx, link):
    """
    Returns (event_fight_rows, fight_links) for this event, or (None, None) on failure.
    """
    try:
        page = get_thread_page()

        page.goto(link, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)  # let UFCStats finish loading dynamic content

        html = page.content()
        soup = BeautifulSoup(html, "lxml")

        date_loc_list = soup.find_all("li", class_="b-list__box-list-item")

        if len(date_loc_list) < 2:
            print(f"FAILED [{idx}] - Title: {soup.title.text if soup.title else 'None'} HTML: {len(html)}")
            return None, None

        event_id = link[-16:]
        date = date_loc_list[0].text.replace("Date:", "").strip()
        location = date_loc_list[1].text.replace("Location:", "").strip()

        fight_rows = soup.find_all(
            "tr",
            class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"
        )

        event_records = []
        event_fight_links = []

        for fight in fight_rows:
            winner_name = None
            winner_id = None

            flag = fight.find("i", class_="b-flag__text")
            w_l_d = flag.text.strip() if flag else None

            fight_id = fight["data-link"][-16:]

            if w_l_d == "win":
                players = fight.find("td", class_="b-fight-details__table-col l-page_align_left")
                if players:
                    players = players.find_all("a", class_="b-link b-link_style_black")
                    if players:
                        winner_name = players[0].text.strip()
                        winner_id = players[0]["href"][-16:]

            event_records.append({
                "event_id": event_id,
                "fight_id": fight_id,
                "date": date,
                "location": location,
                "winner": winner_name,
                "winner_id": winner_id
            })

            event_fight_links.append(fight["data-link"])

        return event_records, event_fight_links

    except Exception as e:
        print(f"FAILED [{idx}] {link}")
        print(f"{type(e).__name__}: {e}")
        return None, None


# ============================================================
# SAVE HELPERS — always merge with existing data, never overwrite
# ============================================================

def save_progress(final=False):
    if winner_names:
        new_events_df = pd.DataFrame(winner_names)
        combined_events_df = pd.concat([existing_events_df, new_events_df], ignore_index=True)
        combined_events_df = combined_events_df.drop_duplicates(subset="fight_id", keep="last")
        combined_events_df.to_csv(events_output_path, index=False)
    else:
        combined_events_df = existing_events_df

    combined_fight_links = existing_fight_links | set(new_fight_links_all)
    with open(fight_links_output_path, "w") as f:
        for link in sorted(combined_fight_links):
            f.write(link + "\n")

    tag = "FINAL" if final else "Checkpoint"
    print(f"{tag} saved: {len(combined_events_df)} fight records, {len(combined_fight_links)} fight links")


# ============================================================
# RUN CONCURRENTLY
# ============================================================

print(f"\nScraping {len(events_to_scrape)} events with {MAX_WORKERS} concurrent browser workers...")

try:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_event_data, idx, link): (idx, link)
            for idx, link in events_to_scrape
        }

        completed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Events"):
            idx, link = futures[future]
            try:
                records, links = future.result()
            except Exception as e:
                print(f"FAILED [{idx}] {link} — {type(e).__name__}: {e}")
                records, links = None, None

            if records:
                with _lock:
                    winner_names.extend(records)
                    new_fight_links_all.extend(links)

            completed += 1
            if completed % 20 == 0:
                with _lock:
                    save_progress()
finally:
    cleanup_all_browsers()


# ============================================================
# FINAL SAVE
# ============================================================

save_progress(final=True)

print()
print("--------------------------------")
print("SCRAPING COMPLETE")
print("--------------------------------")
print(f"Events processed this run: {len(events_to_scrape)}")
print(f"New fight records: {len(winner_names)}")
print(f"New fight links: {len(new_fight_links_all)}")
print(f"Saved to: {events_output_path} and {fight_links_output_path}")
print("--------------------------------")