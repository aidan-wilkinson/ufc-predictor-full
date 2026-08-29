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

fight_details_path = data_dir / "fight_details.csv"
output_path = data_dir / "fighter_details.csv"
refresh_ids_path = data_dir / "fighters_needing_refresh.txt"

base_url = "http://ufcstats.com/fighter-details/"

PLAYWRIGHT_WORKERS = 6
CHECKPOINT_EVERY = 100


# ============================================================
# LOAD EXISTING FIGHTER DATA
# ============================================================

if output_path.exists() and output_path.stat().st_size > 0:
    try:
        existing_fighter_df = pd.read_csv(output_path)

        scraped_fighter_ids = set(
            existing_fighter_df["id"].astype(str)
        )

        print(
            f"Found {len(scraped_fighter_ids)} existing fighter records"
        )

    except pd.errors.EmptyDataError:
        print("fighter_details.csv is empty — starting fresh")

        existing_fighter_df = pd.DataFrame()
        scraped_fighter_ids = set()

else:
    existing_fighter_df = pd.DataFrame()
    scraped_fighter_ids = set()


# ============================================================
# BUILD LIST OF FIGHTER IDS FROM fight_details.csv
# ============================================================

df_fight = pd.read_csv(fight_details_path)

r_fighter_id = df_fight['r_id'].unique()
b_fighter_id = df_fight['b_id'].unique()
all_ids = sorted(set(list(r_fighter_id) + list(b_fighter_id)))

print(f"{len(all_ids)} unique fighter IDs found in fight data")

# ------------------------------------------------------------
# WHO NEEDS SCRAPING?
# ------------------------------------------------------------
# A fighter's career stats (wins, losses, splm, sapm, etc.) change every
# time they have a new fight. Previously, any fighter already present in
# fighter_details.csv was skipped forever — meaning returning fighters'
# stats silently went stale after their first scrape. The correct set to
# (re-)scrape is:
#   1. Fighters who appear in this run's newly-scraped fights (their
#      stats just changed) — read from fighters_needing_refresh.txt,
#      written by scrape_fights.py.
#   2. Fighters never scraped before at all (not yet in fighter_details.csv).

never_scraped_ids = set(all_ids) - scraped_fighter_ids

if refresh_ids_path.exists() and refresh_ids_path.stat().st_size > 0:
    with open(refresh_ids_path, "r") as f:
        needs_refresh_ids = set(line.strip() for line in f if line.strip())
    print(f"Found {len(needs_refresh_ids)} fighters flagged for refresh (had a new fight this cycle)")
else:
    needs_refresh_ids = set()
    print("No fighters_needing_refresh.txt found — run scrape_fights.py first, or all fighters will be treated as new-only")

ids_to_scrape = sorted((never_scraped_ids | needs_refresh_ids) & set(all_ids))

print(f"Never scraped before: {len(never_scraped_ids)}")
print(f"Needing refresh from new fights: {len(needs_refresh_ids)}")
print(f"{len(ids_to_scrape)} fighters remaining to scrape")


# ============================================================
# SHARED STATE
# ============================================================

fighter_detail_data = []
failed_ids = []
_result_lock = threading.Lock()


# ============================================================
# PER-THREAD BROWSER MANAGEMENT
# ============================================================
# Same pattern as scrape_events.py / scrape_fights.py: Playwright's sync
# API requires all calls for a browser/page to happen on the thread that
# created it, so each worker thread lazily starts and reuses its own
# Playwright instance + browser + page.

_thread_local = threading.local()
_active_playwrights = []
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
# FIGHTER SCRAPER
# ============================================================

def get_fighter_data(fighter_id):
    """
    Returns a fighter record dict, or None on failure. Failures print the
    real exception (unlike the original bare except) so problems are
    actually diagnosable.
    """
    link = base_url + fighter_id
    try:
        page = get_thread_page()
        page.goto(link, wait_until="domcontentloaded", timeout=15000)
        html = page.content()

        soup = BeautifulSoup(html, "lxml")

        name_el = soup.find('span', class_='b-content__title-highlight')
        if not name_el:
            print(f"FAILED {link} — page did not contain expected fighter title (possibly blocked or not loaded)")
            return None

        fighter_name = name_el.text.strip()

        nick_el = soup.find('p', class_="b-content__Nickname")
        fighter_nick_name = nick_el.text.strip() if nick_el else None

        record_el = soup.find('span', class_="b-content__title-record")
        fighter_record = record_el.text.replace("Record:", "").strip().split('-')
        fighter_wins = int(fighter_record[0].split()[0])
        fighter_losses = int(fighter_record[1].split()[0])
        fighter_draws = int(fighter_record[2].split()[0])

        detail_list = soup.find_all('li', class_="b-list__box-list-item b-list__box-list-item_type_block")

        try:
            height = detail_list[0].text.replace("Height:", "").strip().replace("'", "").replace('"', '').split()
            height = round((int(height[0]) * 12 + int(height[1])) * 2.54, 2)
        except Exception:
            height = None

        try:
            weight = detail_list[1].text.replace("Weight:", "").strip().replace(" lbs", "")
            weight = round(float(weight) * 0.45359237, 2)
        except Exception:
            weight = None

        try:
            reach = detail_list[2].text.replace("Reach:", "").strip().replace('"', "")
            reach = round(int(reach) * 2.54, 2)
        except Exception:
            reach = None

        try:
            stance = detail_list[3].text.replace("STANCE:", "").strip()
            stance = stance if stance != "" else None
        except Exception:
            stance = None

        try:
            dob = detail_list[4].text.replace("DOB:", "").strip()
            dob = dob if dob != "--" else None
        except Exception:
            dob = None

        splm = float(detail_list[5].text.replace("SLpM:", "").strip())
        str_acc = int(detail_list[6].text.replace("Str. Acc.:", "").strip().replace("%", ""))
        sapm = float(detail_list[7].text.replace("SApM:", "").strip())
        str_def = int(detail_list[8].text.replace("Str. Def:", "").strip().replace("%", ""))
        td_avg = float(detail_list[10].text.replace("TD Avg.:", "").strip())
        td_acc = int(detail_list[11].text.replace("TD Acc.:", "").strip().replace("%", ""))
        td_def = int(detail_list[12].text.replace("TD Def.:", "").strip().replace("%", ""))
        sub_avg = float(detail_list[13].text.replace("Sub. Avg.:", "").strip())

        return {
            "id": fighter_id,
            "name": fighter_name,
            "nick_name": fighter_nick_name,
            "wins": fighter_wins,
            "losses": fighter_losses,
            "draws": fighter_draws,
            "height": height,
            "weight": weight,
            "reach": reach,
            "stance": stance,
            "dob": dob,
            "splm": splm,
            "str_acc": str_acc,
            "sapm": sapm,
            "str_def": str_def,
            "td_avg": td_avg,
            "td_avg_acc": td_acc,
            "td_def": td_def,
            "sub_avg": sub_avg,
        }

    except Exception as e:
        print(f"FAILED {link} — {type(e).__name__}: {e}")
        return None


# ============================================================
# SAVE HELPER — always merges with existing data, never overwrites
# ============================================================

def save_progress(final=False):
    if not fighter_detail_data:
        return
    new_df = pd.DataFrame(fighter_detail_data)
    combined_df = pd.concat([existing_fighter_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="id", keep="last")
    combined_df.to_csv(output_path, index=False)

    tag = "FINAL" if final else "Checkpoint"
    print(f"{tag} saved: {len(combined_df)} total fighters ({len(new_df)} new this run)")


# ============================================================
# RUN CONCURRENTLY
# ============================================================

print(f"\nScraping {len(ids_to_scrape)} fighters with {PLAYWRIGHT_WORKERS} concurrent browser workers...")

try:
    with ThreadPoolExecutor(max_workers=PLAYWRIGHT_WORKERS) as executor:
        futures = {executor.submit(get_fighter_data, fid): fid for fid in ids_to_scrape}

        completed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fighters"):
            fid = futures[future]
            try:
                record = future.result()
            except Exception as e:
                print(f"FAILED {base_url}{fid} — {type(e).__name__}: {e}")
                record = None

            if record is not None:
                with _result_lock:
                    fighter_detail_data.append(record)
            else:
                failed_ids.append(fid)

            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                with _result_lock:
                    save_progress()
finally:
    cleanup_all_browsers()


# ============================================================
# FINAL SAVE
# ============================================================

save_progress(final=True)

print()
print(f"Successfully scraped this run: {len(fighter_detail_data)}")
print(f"Failed: {len(failed_ids)}")

df_fighter = pd.read_csv(output_path)
print(f"Total fighter records on disk: {len(df_fighter)}")