import os

import sqlite3
from dateparser.search import search_dates
from dateparser import parse

DB_PATH = os.getenv("DB_PATH", "/app/data/ai_ignition_616.db")
DEFAULT_BUS_ID = int(os.getenv("DEFAULT_BUS_ID", 616))


def parse_date_range(date_text):
    if not date_text:
        return None, None
    # Use search_dates from dateparser.search
    dates = search_dates(date_text)
    if not dates:
        return None, None
    if len(dates) >= 2:
        # If two or more dates found, treat as a range
        return dates[0][1].date(), dates[1][1].date()
    # Only one date found: use it for both start and end
    d = dates[0][1].date()
    return d, d


def get_connection():
    return sqlite3.connect(DB_PATH)