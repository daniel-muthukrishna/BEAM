"""
Build tessbeam's sector table: for every stretch of time, which sector TESS is in and
where the spacecraft points.

    python -m tessbeam.build_sectors             # rebuild tessbeam/data/sectors.csv
    python -m tessbeam.build_sectors --check     # show what would change, write nothing
    python -m tessbeam.build_sectors -o my.csv
"""

import argparse
import csv
import os
from datetime import datetime, timedelta, timezone

import requests

MIT_URL = "https://tess.mit.edu/public/files/TESS_orbit_times.csv"
SCHEDULE_URL = "https://raw.githubusercontent.com/tessgi/tesswcs/main/src/tesswcs/data/pointings.csv"
MIT_HEADER = ["Sector", "Orbit", "Start of Orbit", "End of Orbit"]
SCHEDULE_HEADER = ["Cycle", "Sector", "RA", "Dec", "Roll", "Start", "End"]
COLUMNS = ["sector", "start_utc", "end_utc", "ra_deg", "dec_deg", "roll_deg", "planned", "problem"]
SNAPSHOT = os.path.join(os.path.dirname(__file__), "data", "sectors.csv")   # shipped with the package
TOL_DAYS = 2.0     # MIT and the schedule may differ by this much (the schedule is day-precision)


ATLAS_SECTOR = 1751
S99_PAUSE = "2026-01-15 05:42:37"
S1751_SPAN = ("2026-01-15 06:02:37", "2026-01-22 11:59:17")
S99_RESUME = "2026-01-22 12:15:57"


def fetch(url):
    """The text of one source file."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"could not download {url} ({e}); this needs internet access") from e
    return r.content.decode("utf-8-sig")       # the schedule file starts with a byte-order mark


def utc_now():
    """The "# built:" stamp for a table made now."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _utc(text):
    return datetime.strptime(text.strip().replace("T", " "), "%Y-%m-%d %H:%M:%S")


def _jd(jd):
    """JD (UTC), which is how the schedule gives its dates -> datetime, to the second."""
    return datetime(1858, 11, 17) + timedelta(seconds=round((float(jd) - 2400000.5) * 86400))


def _records(text, header, name):
    """The data rows of a source file (lists of strings), after checking its header."""
    lines = [l for l in text.lstrip("﻿").splitlines() if l.strip() and not l.startswith("#")]
    rows = [[c.strip() for c in r] for r in csv.reader(lines)]
    if not rows or rows[0] != header:
        raise ValueError(f"{name} does not begin with the header {','.join(header)}")
    if len(rows) == 1:
        raise ValueError(f"{name} has no data rows")
    return rows[1:]


def _mit_sectors(text):
    """MIT orbit times -> {sector: (start, end)}, each sector exactly as MIT labels its rows."""
    spans = {}
    for rec in _records(text, MIT_HEADER, "MIT's orbit-times file"):
        try:
            sector, start, end = int(rec[0]), _utc(rec[2]), _utc(rec[3])
        except (ValueError, IndexError):
            raise ValueError(f"MIT's orbit-times file has a row tessbeam can't read: {','.join(rec)}")
        first, last = spans.get(sector, (start, end))
        spans[sector] = (min(first, start), max(last, end))
    return spans


def _schedule(text):
    """Published schedule -> {sector: dict(ra, dec, roll, start, end)}."""
    plan = {}
    for rec in _records(text, SCHEDULE_HEADER, "the published schedule"):
        try:
            ra, dec, roll = (float(x) for x in rec[2:5])
            plan[int(rec[1])] = dict(ra=ra, dec=dec, roll=roll, start=_jd(rec[5]), end=_jd(rec[6]))
        except (ValueError, IndexError, OverflowError):
            raise ValueError(f"the published schedule has a row tessbeam can't read: {','.join(rec)}")
    return plan


def build(mit_text, schedule_text):
    """The text of the two source files -> the table rows (dicts keyed by COLUMNS), in time order.

    Pure: no network and no clock. Raises ValueError when either file or the resulting
    table looks wrong, so that a caller keeps the table it already has.
    """
    flown = _mit_sectors(mit_text)
    plan = _schedule(schedule_text)
    last = max(flown, key=lambda s: flown[s][1])          # MIT's last sector, usually only partly listed
    later = sorted((s for s in plan if s > last and s != ATLAS_SECTOR), key=lambda s: plan[s]["start"])
    tol = timedelta(days=TOL_DAYS)
    pause, resume = _utc(S99_PAUSE), _utc(S99_RESUME)

    def planned_end(s):
        # some planned sectors are published as ending 0.05 d after the next one starts
        nexts = [plan[n]["start"] for n in later if plan[n]["start"] > plan[s]["start"]]
        return min([plan[s]["end"]] + nexts)

    def row(sector, start, end, planned=0, problem=""):
        p = plan.get(sector, {})
        if not p:
            problem = (f"MIT's orbit-times file lists sector {sector} but the published schedule "
                       "does not, so tessbeam does not know where the spacecraft points.")
        return {"sector": sector, "start_utc": start.isoformat(), "end_utc": end.isoformat(),
                "ra_deg": p.get("ra"), "dec_deg": p.get("dec"), "roll_deg": p.get("roll"),
                "planned": planned, "problem": problem}

    rows = []
    for s, (start, end) in flown.items():                 # the sectors MIT lists, dates exact
        p, planned, problem = plan.get(s), 0, ""
        split = s == 99 and start <= pause and resume <= end
        if p and s == last:
            if p["end"] - end > tol:                      # only partly listed: the rest is still a plan
                end, planned = planned_end(s), 1
        elif p and (abs(start - p["start"]) > tol or abs(end - p["end"]) > tol):
            # a check, never a correction: say what disagrees and let the user decide
            problem = (f"MIT's orbit-times file lists sector {s} as {start:%Y-%m-%d %H:%M} to "
                       f"{end:%Y-%m-%d %H:%M} UTC but the published schedule has "
                       f"{p['start']:%Y-%m-%d} to {p['end']:%Y-%m-%d}; they disagree, so tessbeam "
                       "does not know the pointing for these dates.")
        if split:
            rows += [row(99, start, pause, 0, problem),
                     row(ATLAS_SECTOR, *map(_utc, S1751_SPAN)),
                     row(99, resume, end, planned, problem)]
        else:
            rows.append(row(s, start, end, planned, problem))

    for s in later:                                       # the sectors only the schedule has
        start = max(plan[s]["start"], _utc(rows[-1]["end_utc"]))
        rows.append(row(s, start, planned_end(s), planned=1))

    # the times are ISO strings of one fixed format, so they compare like the times themselves
    for r in rows:
        if r["end_utc"] <= r["start_utc"]:
            raise ValueError(f"sector {r['sector']} would end ({r['end_utc']}) before it starts "
                             f"({r['start_utc']})")
    for a, b in zip(rows, rows[1:]):
        if b["start_utc"] < a["end_utc"]:
            raise ValueError(f"the sectors are not in time order: sector {b['sector']} starts "
                             f"{b['start_utc']}, before sector {a['sector']} ends {a['end_utc']}")
    return rows


def write(rows, path, built):
    """Write a table file. Atomic (temporary file, then rename), so nobody reads half a table."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        f.write(f"# built: {built}\n")
        f.write(f"# sources: {MIT_URL} (dates of the sectors MIT lists), "
                f"{SCHEDULE_URL} (pointings; dates of planned sectors)\n")
        f.write("# one row per stretch of constant pointing; times are UTC; planned=1: the end "
                "date is the schedule's; problem: why tessbeam will not use the row\n")
        w = csv.DictWriter(f, COLUMNS, lineterminator="\n")   # csv quotes the commas in `problem`
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read(path):
    """A table file -> (rows, built stamp). The rows are the same dicts build() returns."""
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.read().splitlines()
    built = [l.split(":", 1)[1].strip() for l in lines if l.startswith("# built:")]
    reader = csv.DictReader([l for l in lines if not l.startswith("#")])
    if reader.fieldnames != COLUMNS or not built:
        raise ValueError(f"{path} is not a tessbeam sector table")
    num = lambda x: float(x) if x else None                # no pointing is written as an empty cell
    rows = [{"sector": int(r["sector"]), "start_utc": r["start_utc"], "end_utc": r["end_utc"],
             "ra_deg": num(r["ra_deg"]), "dec_deg": num(r["dec_deg"]), "roll_deg": num(r["roll_deg"]),
             "planned": int(r["planned"]), "problem": r["problem"]}
            for r in reader]
    if not rows:
        raise ValueError(f"{path} is an empty tessbeam sector table")
    return rows, built[0]


def summary(rows):
    """A few lines describing a table: row counts and the flagged sectors."""
    n_planned = sum(r["planned"] for r in rows)
    lines = [f"{len(rows)} rows: {len(rows) - n_planned} exact, {n_planned} planned"]
    flagged = {r["sector"]: r["problem"] for r in rows if r["problem"]}
    return lines + [f"flagged: sector {s}: {why}" for s, why in flagged.items()]


def changes(old, new):
    """What differs between two tables, as lines: sectors added, removed or changed."""
    by_sector = lambda rows: {s: [r for r in rows if r["sector"] == s] for s in {r["sector"] for r in rows}}
    show = lambda r: ", ".join(str(r[c]) for c in COLUMNS[1:7]) + (", problem" if r["problem"] else "")
    old, new = by_sector(old), by_sector(new)
    lines = [f"added: sector {s}" for s in sorted(set(new) - set(old))]
    lines += [f"removed: sector {s}" for s in sorted(set(old) - set(new))]
    for s in sorted(set(old) & set(new)):
        if old[s] != new[s]:
            lines += [f"changed: sector {s}"] + [f"    was {show(r)}" for r in old[s]]
            lines += [f"    now {show(r)}" for r in new[s]]
    return lines or ["no changes"]


def main():
    ap = argparse.ArgumentParser(description="Build tessbeam's sector table from MIT's orbit times "
                                             "and the published pointing schedule")
    ap.add_argument("--check", action="store_true", help="show the summary and the changes, write nothing")
    ap.add_argument("-o", "--output", default=SNAPSHOT, help="table file to write (default: the packaged one)")
    args = ap.parse_args()

    rows = build(fetch(MIT_URL), fetch(SCHEDULE_URL))
    print("\n".join(summary(rows)))
    if os.path.exists(args.output):
        old, built = read(args.output)
        print(f"compared with {args.output} (built {built}):")
        print("\n".join("  " + line for line in changes(old, rows)))
    if args.check:
        print("--check: nothing written")
    else:
        write(rows, args.output, utc_now())
        print("wrote", args.output)


if __name__ == "__main__":
    main()
