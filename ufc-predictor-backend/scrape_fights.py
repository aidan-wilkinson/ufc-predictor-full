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

links_path = data_dir / "fight_links.txt"
output_path = data_dir / "fight_details.csv"

CHECKPOINT_EVERY = 100    # rows between incremental saves


# ============================================================
# LOAD LINKS + EXISTING DATA
# ============================================================

with open(links_path, "r") as f:
    fight_links = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(fight_links)} fight links")

if output_path.exists():
    existing_df = pd.read_csv(output_path)
    scraped_ids = set(existing_df["fight_id"].astype(str))
    print(f"Found {len(scraped_ids)} existing fights — will skip these and append new ones")
else:
    existing_df = pd.DataFrame()
    scraped_ids = set()

links_to_scrape = [
    (idx, link) for idx, link in enumerate(fight_links)
    if link[-16:] not in scraped_ids
]

print(f"{len(links_to_scrape)} fights remaining to scrape")


# ============================================================
# SHARED STATE
# ============================================================

fight_details = []          # accumulates NEW rows from this run only
failed_links = []           # (idx, link) pairs that need Playwright fallback


def save_progress(final=False):
    """
    Always combines existing_df (data present before this run) with
    whatever has been scraped so far in fight_details. Never overwrites
    with new-data-only, so partial/interrupted runs never lose old rows.
    """
    if not fight_details:
        return

    new_df = pd.DataFrame(fight_details)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="fight_id", keep="last")
    combined_df.to_csv(output_path, index=False)

    tag = "FINAL" if final else "Checkpoint"
    print(f"{tag} saved: {len(combined_df)} total fights ({len(new_df)} new this run)")


# ============================================================
# HELPERS (unchanged parsing logic)
# ============================================================

def percentage(landed, attempted):
    try:
        return int(round(landed / attempted, 2) * 100)
    except Exception:
        return None


def parse_stat_pair(text):
    """
    UFCStats usually gives stats like:
        10 of 20
        5 of 10
    Returns: red_landed, red_attempted, blue_landed, blue_attempted
    """
    values = text.split()
    return (
        int(values[0]),
        int(values[2]),
        int(values[3]),
        int(values[5])
    )


def parse_accuracy_pair(text):
    """
    Parses: 50% 40%   or   --- 40%
    """
    values = text.split()
    red = None if values[0] == "---" else int(values[0].replace("%", ""))
    blue = None if values[1] == "---" else int(values[1].replace("%", ""))
    return red, blue


def parse_control_pair(text):
    """
    Parses: 1:30 2:15
    """
    values = text.split()

    def convert(value):
        if value == "--":
            return None
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + int(seconds)

    return convert(values[0]), convert(values[1])


# ============================================================
# PARSE + RECORD (shared by both fetch paths)
# ============================================================

def parse_fight_html(idx, link, html):
    """
    Parses fetched HTML into a fight record dict. Returns None on failure.
    """
    fight_id = link[-16:]

    soup = BeautifulSoup(html, "lxml")

    # ---- BASIC PAGE VALIDATION ----
    if "Loading" in soup.title.text if soup.title else True:
        print(f"FAILED [{idx}] - Title: {soup.title.text if soup.title else 'None'} HTML: {len(html)}")
        return None

    # ---- EVENT ----
    event_link = soup.find("a", class_="b-link")
    if not event_link:
        print(f"FAILED [{idx}] - Event link not found")
        return None

    event_name = event_link.text.strip()
    event_id = event_link["href"][-16:]

    # ---- FIGHTERS ----
    fighter_names = soup.find_all("a", class_="b-link b-fight-details__person-link")
    if len(fighter_names) < 2:
        print(f"FAILED [{idx}] - Fighter names not found")
        return None

    r_name = fighter_names[0].text.strip()
    b_name = fighter_names[1].text.strip()
    r_id = fighter_names[0]["href"][-16:]
    b_id = fighter_names[1]["href"][-16:]

    # ---- DIVISION / TITLE ----
    division_element = soup.find("i", class_="b-fight-details__fight-title")
    if division_element:
        division_info = division_element.text.lower().strip()
        is_title_fight = 1 if "title" in division_info else 0
        division_info = division_info.replace("ufc", "").replace("title", "").replace("bout", "").strip()
    else:
        division_info = None
        is_title_fight = 0

    # ---- METHOD ----
    method_element = soup.find("i", style="font-style: normal")
    method = method_element.text.strip() if method_element else None

    # ---- FIGHT DETAILS ----
    fight_details_element = soup.find("p", class_="b-fight-details__text")
    if not fight_details_element:
        print(f"FAILED [{idx}] - Fight detail section not found")
        return None

    fight_details_list = fight_details_element.find_all("i", class_="b-fight-details__text-item")

    finish_round = int(fight_details_list[0].text.lower().replace("round:", "").strip())

    match_timestamp = fight_details_list[1].text.lower().replace("time:", "").strip()
    minutes, seconds = match_timestamp.split(":")
    match_time_sec = int(minutes) * 60 + int(seconds)

    total_rounds = fight_details_list[2].text.lower().replace("time format:", "").strip()
    total_rounds = None if total_rounds == "no time limit" else int(total_rounds[0])

    referee = fight_details_list[3].text.replace("Referee:", "").strip()

    # ---- DEFAULT ALL STATS TO NONE ----
    r_kd = b_kd = None
    r_sig_str_landed = b_sig_str_landed = None
    r_sig_str_atmpted = b_sig_str_atmpted = None
    r_sig_str_acc = b_sig_str_acc = None
    r_total_str_landed = b_total_str_landed = None
    r_total_str_atmpted = b_total_str_atmpted = None
    r_total_str_acc = b_total_str_acc = None
    r_td_landed = b_td_landed = None
    r_td_atmpted = b_td_atmpted = None
    r_td_acc = b_td_acc = None
    r_sub_att = b_sub_att = None
    r_ctrl = b_ctrl = None
    r_head_landed = b_head_landed = None
    r_head_atmpted = b_head_atmpted = None
    r_head_acc = b_head_acc = None
    r_body_landed = b_body_landed = None
    r_body_atmpted = b_body_atmpted = None
    r_body_acc = b_body_acc = None
    r_leg_landed = b_leg_landed = None
    r_leg_atmpted = b_leg_atmpted = None
    r_leg_acc = b_leg_acc = None
    r_dist_landed = b_dist_landed = None
    r_dist_atmpted = b_dist_atmpted = None
    r_dist_acc = b_dist_acc = None
    r_clinch_landed = b_clinch_landed = None
    r_clinch_atmpted = b_clinch_atmpted = None
    r_clinch_acc = b_clinch_acc = None
    r_ground_landed = b_ground_landed = None
    r_ground_atmpted = b_ground_atmpted = None
    r_ground_acc = b_ground_acc = None
    r_rev = b_rev = None

    # ---- TABLES ----
    tables = soup.find_all("table", style="width: 745px")

    if len(tables) >= 2:
        table1 = tables[0]
        td_1_list = table1.find_all("td", class_="b-fight-details__table-col")

        kd_players = td_1_list[1].text.split()
        r_kd = int(kd_players[0])
        b_kd = int(kd_players[1])

        r_sig_str_landed, r_sig_str_atmpted, b_sig_str_landed, b_sig_str_atmpted = parse_stat_pair(td_1_list[2].text)
        r_sig_str_acc, b_sig_str_acc = parse_accuracy_pair(td_1_list[3].text)

        r_total_str_landed, r_total_str_atmpted, b_total_str_landed, b_total_str_atmpted = parse_stat_pair(td_1_list[4].text)
        r_total_str_acc = percentage(r_total_str_landed, r_total_str_atmpted)
        b_total_str_acc = percentage(b_total_str_landed, b_total_str_atmpted)

        r_td_landed, r_td_atmpted, b_td_landed, b_td_atmpted = parse_stat_pair(td_1_list[5].text)
        r_td_acc, b_td_acc = parse_accuracy_pair(td_1_list[6].text)

        sub_att = td_1_list[7].text.split()
        r_sub_att = int(sub_att[0])
        b_sub_att = int(sub_att[1])

        rev = td_1_list[8].text.split()
        r_rev = int(rev[0])
        b_rev = int(rev[1])

        r_ctrl, b_ctrl = parse_control_pair(td_1_list[9].text)

        table2 = tables[1]
        td_2_list = table2.find_all("td", class_="b-fight-details__table-col")

        r_head_landed, r_head_atmpted, b_head_landed, b_head_atmpted = parse_stat_pair(td_2_list[3].text)
        r_head_acc = percentage(r_head_landed, r_head_atmpted)
        b_head_acc = percentage(b_head_landed, b_head_atmpted)

        r_body_landed, r_body_atmpted, b_body_landed, b_body_atmpted = parse_stat_pair(td_2_list[4].text)
        r_body_acc = percentage(r_body_landed, r_body_atmpted)
        b_body_acc = percentage(b_body_landed, b_body_atmpted)

        r_leg_landed, r_leg_atmpted, b_leg_landed, b_leg_atmpted = parse_stat_pair(td_2_list[5].text)
        r_leg_acc = percentage(r_leg_landed, r_leg_atmpted)
        b_leg_acc = percentage(b_leg_landed, b_leg_atmpted)

        r_dist_landed, r_dist_atmpted, b_dist_landed, b_dist_atmpted = parse_stat_pair(td_2_list[6].text)
        r_dist_acc = percentage(r_dist_landed, r_dist_atmpted)
        b_dist_acc = percentage(b_dist_landed, b_dist_atmpted)

        r_clinch_landed, r_clinch_atmpted, b_clinch_landed, b_clinch_atmpted = parse_stat_pair(td_2_list[7].text)
        r_clinch_acc = percentage(r_clinch_landed, r_clinch_atmpted)
        b_clinch_acc = percentage(b_clinch_landed, b_clinch_atmpted)

        r_ground_landed, r_ground_atmpted, b_ground_landed, b_ground_atmpted = parse_stat_pair(td_2_list[8].text)
        r_ground_acc = percentage(r_ground_landed, r_ground_atmpted)
        b_ground_acc = percentage(b_ground_landed, b_ground_atmpted)

    # ---- STRIKE DISTRIBUTION ----
    r_landed_head_per = r_landed_body_per = r_landed_leg_per = None
    r_landed_dist_per = r_landed_clinch_per = r_landed_ground_per = None
    b_landed_head_per = b_landed_body_per = b_landed_leg_per = None
    b_landed_dist_per = b_landed_clinch_per = b_landed_ground_per = None

    try:
        red = soup.find_all("i", class_="b-fight-details__charts-num b-fight-details__charts-num_style_red b-fight-details__charts-num_pos_left js-red")
        r_landed_head_per = int(red[0].text.strip().replace("%", ""))
        r_landed_dist_per = int(red[1].text.strip().replace("%", ""))

        blue = soup.find_all("i", class_="b-fight-details__charts-num b-fight-details__charts-num_style_blue b-fight-details__charts-num_pos_right js-blue")
        b_landed_head_per = int(blue[0].text.strip().replace("%", ""))
        b_landed_dist_per = int(blue[1].text.strip().replace("%", ""))
    except Exception:
        pass

    try:
        red = soup.find_all("i", class_="b-fight-details__charts-num b-fight-details__charts-num_style_dark-red b-fight-details__charts-num_pos_left js-red")
        r_landed_body_per = int(red[0].text.strip().replace("%", ""))
        r_landed_clinch_per = int(red[1].text.strip().replace("%", ""))

        blue = soup.find_all("i", class_="b-fight-details__charts-num b-fight-details__charts-num_style_dark-blue b-fight-details__charts-num_pos_right js-blue")
        b_landed_body_per = int(blue[0].text.strip().replace("%", ""))
        b_landed_clinch_per = int(blue[1].text.strip().replace("%", ""))
    except Exception:
        pass

    try:
        red = soup.find_all("i", class_="b-fight-details__charts-num b-fight-details__charts-num_style_light-red b-fight-details__charts-num_pos_left js-red")
        r_landed_leg_per = int(red[0].text.strip().replace("%", ""))
        r_landed_ground_per = int(red[1].text.strip().replace("%", ""))

        blue = soup.find_all("i", class_="b-fight-details__charts-num b-fight-details__charts-num_style_light-blue b-fight-details__charts-num_pos_right js-blue")
        b_landed_leg_per = int(blue[0].text.strip().replace("%", ""))
        b_landed_ground_per = int(blue[1].text.strip().replace("%", ""))
    except Exception:
        pass

    return {
        "event_name": event_name,
        "event_id": event_id,
        "fight_id": fight_id,
        "r_name": r_name,
        "r_id": r_id,
        "b_name": b_name,
        "b_id": b_id,
        "division": division_info,
        "title_fight": is_title_fight,
        "method": method,
        "finish_round": finish_round,
        "match_time_sec": match_time_sec,
        "total_rounds": total_rounds,
        "referee": referee,
        "r_kd": r_kd, "r_sig_str_landed": r_sig_str_landed, "r_sig_str_atmpted": r_sig_str_atmpted, "r_sig_str_acc": r_sig_str_acc,
        "r_total_str_landed": r_total_str_landed, "r_total_str_atmpted": r_total_str_atmpted, "r_total_str_acc": r_total_str_acc,
        "r_td_landed": r_td_landed, "r_td_atmpted": r_td_atmpted, "r_td_acc": r_td_acc,
        "r_sub_att": r_sub_att, "r_ctrl": r_ctrl,
        "r_head_landed": r_head_landed, "r_head_atmpted": r_head_atmpted, "r_head_acc": r_head_acc,
        "r_body_landed": r_body_landed, "r_body_atmpted": r_body_atmpted, "r_body_acc": r_body_acc,
        "r_leg_landed": r_leg_landed, "r_leg_atmpted": r_leg_atmpted, "r_leg_acc": r_leg_acc,
        "r_dist_landed": r_dist_landed, "r_dist_atmpted": r_dist_atmpted, "r_dist_acc": r_dist_acc,
        "r_clinch_landed": r_clinch_landed, "r_clinch_atmpted": r_clinch_atmpted, "r_clinch_acc": r_clinch_acc,
        "r_ground_landed": r_ground_landed, "r_ground_atmpted": r_ground_atmpted, "r_ground_acc": r_ground_acc,
        "r_landed_head_per": r_landed_head_per, "r_landed_body_per": r_landed_body_per, "r_landed_leg_per": r_landed_leg_per,
        "r_landed_dist_per": r_landed_dist_per, "r_landed_clinch_per": r_landed_clinch_per, "r_landed_ground_per": r_landed_ground_per,
        "b_kd": b_kd, "b_sig_str_landed": b_sig_str_landed, "b_sig_str_atmpted": b_sig_str_atmpted, "b_sig_str_acc": b_sig_str_acc,
        "b_total_str_landed": b_total_str_landed, "b_total_str_atmpted": b_total_str_atmpted, "b_total_str_acc": b_total_str_acc,
        "b_td_landed": b_td_landed, "b_td_atmpted": b_td_atmpted, "b_td_acc": b_td_acc,
        "b_sub_att": b_sub_att, "b_ctrl": b_ctrl,
        "b_head_landed": b_head_landed, "b_head_atmpted": b_head_atmpted, "b_head_acc": b_head_acc,
        "b_body_landed": b_body_landed, "b_body_atmpted": b_body_atmpted, "b_body_acc": b_body_acc,
        "b_leg_landed": b_leg_landed, "b_leg_atmpted": b_leg_atmpted, "b_leg_acc": b_leg_acc,
        "b_dist_landed": b_dist_landed, "b_dist_atmpted": b_dist_atmpted, "b_dist_acc": b_dist_acc,
        "b_clinch_landed": b_clinch_landed, "b_clinch_atmpted": b_clinch_atmpted, "b_clinch_acc": b_clinch_acc,
        "b_ground_landed": b_ground_landed, "b_ground_atmpted": b_ground_atmpted, "b_ground_acc": b_ground_acc,
        "b_landed_head_per": b_landed_head_per, "b_landed_body_per": b_landed_body_per, "b_landed_leg_per": b_landed_leg_per,
        "b_landed_dist_per": b_landed_dist_per, "b_landed_clinch_per": b_landed_clinch_per, "b_landed_ground_per": b_landed_ground_per,
    }


# ============================================================
# CONCURRENT PLAYWRIGHT SCRAPE
# ============================================================
# The requests-based fast path was tested and found to succeed 0% of the
# time on this site (every fight page requires JS rendering), so it's been
# removed entirely rather than paying its overhead on every single fight
# before falling through to Playwright anyway.
#
# Playwright's sync API requires all calls for a browser/page to happen on
# the thread that created them, so — same pattern as scrape_events.py —
# each worker thread lazily starts its own Playwright instance + browser +
# page on first use and reuses it for every fight that thread processes.

_thread_local = threading.local()
_active_playwrights = []
_active_lock = threading.Lock()
_result_lock = threading.Lock()

PLAYWRIGHT_WORKERS = 6  # each worker is a full browser instance — keep modest


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


def fetch_via_playwright(idx, link):
    """
    Returns (idx, link, record) — record is None on failure.
    """
    try:
        page = get_thread_page()
        page.goto(link, wait_until="domcontentloaded", timeout=15000)
        html = page.content()
    except Exception as e:
        print(f"FAILED [{idx}] {link} — Playwright could not load page: {type(e).__name__}")
        return idx, link, None

    record = parse_fight_html(idx, link, html)
    return idx, link, record


print(f"\nScraping {len(links_to_scrape)} fights with {PLAYWRIGHT_WORKERS} concurrent browser workers...")

try:
    with ThreadPoolExecutor(max_workers=PLAYWRIGHT_WORKERS) as executor:
        futures = {
            executor.submit(fetch_via_playwright, idx, link): (idx, link)
            for idx, link in links_to_scrape
        }

        completed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fights"):
            idx, link = futures[future]
            try:
                _, _, record = future.result()
            except Exception as e:
                print(f"FAILED [{idx}] {link} — {type(e).__name__}: {e}")
                record = None

            if record is not None:
                with _result_lock:
                    fight_details.append(record)
            else:
                failed_links.append((idx, link))

            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                with _result_lock:
                    save_progress()
finally:
    cleanup_all_browsers()

print(f"Scrape complete: {len(fight_details)} succeeded, {len(failed_links)} failed")


# ============================================================
# FINAL SAVE — always merges with existing_df, never overwrites it
# ============================================================

save_progress(final=True)

# Record which fighters were involved in THIS run's newly-scraped fights,
# so scrape_fighter_details.py knows whose career stats need refreshing
# (a fighter's stats change every time they have a new fight — without
# this, returning fighters would never get their stats updated after
# their first scrape).
new_fighter_ids_path = data_dir / "fighters_needing_refresh.txt"
new_fighter_ids = set()
for record in fight_details:
    new_fighter_ids.add(record["r_id"])
    new_fighter_ids.add(record["b_id"])

with open(new_fighter_ids_path, "w") as f:
    for fid in sorted(new_fighter_ids):
        f.write(fid + "\n")

print(f"Wrote {len(new_fighter_ids)} fighter IDs needing a stat refresh to {new_fighter_ids_path}")

print()
print("=" * 60)
print("FIGHT SCRAPING COMPLETE")
print("=" * 60)
print(f"New fights scraped this run: {len(fight_details)}")
print(f"Saved to: {output_path}")
print("=" * 60)

