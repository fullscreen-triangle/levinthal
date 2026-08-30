#!/usr/bin/env python3
"""
Fetch real reaction/enzyme data from two heterogeneous public sources and
FREEZE it as a local fixture.

Why freeze.  The host system (HFQ) forbids network I/O in adapters by
construction: its claims are properties of the compiler, and a live service
can neither confirm nor refute them.  We honour that discipline.  The data
is real and fetched once; every experiment then runs against the frozen
snapshot, so results are reproducible and no claim depends on a service
being up.

Sources are genuinely heterogeneous -- different schemas, different
identifier spaces, different capabilities -- which is the setting a
federated query language exists for.
"""
from __future__ import annotations
import json, os, time, sys
from typing import Dict, List, Optional
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
KEGG = "https://rest.kegg.jp"
REACTOME = "https://reactome.org/ContentService"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ladder-federated-validation/1.0"})
TIMEOUT = 30


def _get(url: str, as_json: bool = False, tries: int = 3):
    for k in range(tries):
        try:
            r = SESSION.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json() if as_json else r.text
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def fetch_kegg_enzymes(limit: int = 400) -> List[Dict]:
    """EC entries with their reaction lists. Header is 'ENTRY  EC 1.1.1.1'."""
    listing = _get(f"{KEGG}/list/enzyme")
    if not listing:
        return []
    ids = [ln.split("\t")[0] for ln in listing.strip().split("\n") if ln.strip()]
    ids = [i.split(":")[-1] if ":" in i else i for i in ids][:limit]
    out = []
    for i in range(0, len(ids), 10):
        chunk = ids[i:i + 10]
        txt = _get(f"{KEGG}/get/{'+'.join('ec:'+c for c in chunk)}")
        if not txt:
            continue
        for rec in txt.split("\n///"):
            rec = rec.strip()
            if not rec:
                continue
            ent: Dict[str, object] = {"reactions": [], "substrates": [],
                                      "products": [], "cofactors": []}
            section = None
            for line in rec.split("\n"):
                if line[:1] not in (" ", "") and line.strip():
                    section = line.split()[0]
                    rest = line[len(section):].strip()
                else:
                    rest = line.strip()
                if section == "ENTRY":
                    toks = rest.split()
                    if toks and toks[0] == "EC" and len(toks) > 1:
                        ent["ec"] = toks[1]
                    elif toks:
                        ent["ec"] = toks[0]
                elif section == "ALL_REAC" and rest:
                    for t in rest.replace(";", " ").split():
                        if t.startswith("R") and t[1:].isdigit():
                            ent["reactions"].append(t)
                elif section == "SUBSTRATE" and rest:
                    ent["substrates"].append(rest.split("[")[0].strip())
                elif section == "PRODUCT" and rest:
                    ent["products"].append(rest.split("[")[0].strip())
                elif section == "COFACTOR" and rest:
                    ent["cofactors"].append(rest.split("[")[0].strip())
            if ent.get("ec"):
                out.append(ent)
        print(f"    kegg {min(i+10,len(ids))}/{len(ids)}  kept={len(out)}",
              flush=True)
    return out


def fetch_reactome(species: str = "Homo sapiens",
                   n_pathways: int = 30) -> List[Dict]:
    """Reaction participants from Reactome pathways."""
    paths = _get(f"{REACTOME}/data/pathways/top/9606", as_json=True)
    if not isinstance(paths, list):
        return []
    out: List[Dict] = []
    for p in paths[:n_pathways]:
        pid = p.get("stId")
        if not pid:
            continue
        evs = _get(f"{REACTOME}/data/pathway/{pid}/containedEvents",
                   as_json=True)
        if not isinstance(evs, list):
            continue
        for ev in evs[:60]:
            if not isinstance(ev, dict):
                continue
            if ev.get("className") != "Reaction":
                continue
            rid = ev.get("stId")
            if not rid:
                continue
            det = _get(f"{REACTOME}/data/query/{rid}", as_json=True)
            if not isinstance(det, dict):
                continue
            def names(key):
                v = det.get(key) or []
                return [e.get("displayName") for e in v
                        if isinstance(e, dict) and e.get("displayName")]
            out.append({
                "rid": rid,
                "pathway": p.get("displayName"),
                "name": det.get("displayName"),
                "inputs": names("input"),
                "outputs": names("output"),
                # The summary response carries `displayName` on the
                # catalystActivity entry itself and omits `physicalEntity`.
                # An earlier version required physicalEntity and therefore
                # extracted zero catalysts from a corpus that has them; the
                # emptiness was checked against the live API rather than
                # assumed, which is how the omission was found.
                "catalysts": [
                    c.get("displayName")
                    for c in (det.get("catalystActivity") or [])
                    if isinstance(c, dict) and c.get("displayName")
                ],
            })
        print(f"    reactome {p.get('displayName')[:38]:40s} total={len(out)}",
              flush=True)
        if len(out) >= 900:
            break
    return out


def main() -> int:
    os.makedirs(FIX, exist_ok=True)
    print("fetching KEGG enzymes ...", flush=True)
    kegg = fetch_kegg_enzymes()
    print(f"  -> {len(kegg)} enzyme records", flush=True)
    print("fetching Reactome reactions ...", flush=True)
    rx = fetch_reactome()
    print(f"  -> {len(rx)} reactions", flush=True)

    snap = {
        "snapshot": "live-" + time.strftime("%Y%m%d"),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {
            "kegg": {"endpoint": KEGG, "n": len(kegg),
                     "schema": "EC -> reactions, substrates, products"},
            "reactome": {"endpoint": REACTOME, "n": len(rx),
                         "schema": "stId -> inputs, outputs, catalysts"},
        },
        "kegg": kegg,
        "reactome": rx,
    }
    out = os.path.join(FIX, "sources.json")
    with open(out, "w") as fh:
        json.dump(snap, fh, indent=1)
    print(f"wrote {out}  ({os.path.getsize(out)} bytes)")
    return 0 if (kegg and rx) else 1


if __name__ == "__main__":
    sys.exit(main())
