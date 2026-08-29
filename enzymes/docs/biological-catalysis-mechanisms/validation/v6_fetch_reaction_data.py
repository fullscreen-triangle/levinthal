#!/usr/bin/env python3
"""
V6 --- Retrieve reaction and pathway data from public knowledgebases.

Sources:
  KEGG      https://rest.kegg.jp        (reactions, enzymes, pathways)
  Reactome  https://reactome.org        (pathways, reaction participants)

This script ONLY retrieves and caches.  It computes no framework quantity.
Analysis is done in v3/v4/v5 so that retrieval and interpretation are
separable and the cache can be inspected independently.

All retrieved payloads are written verbatim to results/cache/ so that any
number reported downstream can be traced to a raw record.
"""

from __future__ import annotations
import json
import os
import time
from typing import Dict, List, Optional

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")

KEGG = "https://rest.kegg.jp"
REACTOME = "https://reactome.org/ContentService"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "categorical-catalysis-validation/1.0"})
TIMEOUT = 30
PAUSE = 0.34          # be polite to public endpoints


def _get(url: str, accept_json: bool = False) -> Optional[object]:
    try:
        r = SESSION.get(url, timeout=TIMEOUT)
        time.sleep(PAUSE)
        if r.status_code != 200:
            return None
        return r.json() if accept_json else r.text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# KEGG
# ---------------------------------------------------------------------------
def fetch_kegg_enzyme_list() -> List[str]:
    """All EC numbers known to KEGG."""
    txt = _get(f"{KEGG}/list/enzyme")
    if not txt:
        return []
    ecs = []
    for line in txt.strip().split("\n"):
        parts = line.split("\t")
        if not parts or not parts[0]:
            continue
        # KEGG has served both "ec:1.1.1.1" and bare "1.1.1.1"; accept either.
        tok = parts[0][3:] if parts[0].startswith("ec:") else parts[0]
        # keep only fully specified EC numbers (four fields, no '-')
        fields = tok.split(".")
        if len(fields) == 4 and "-" not in fields:
            ecs.append(tok)
    return ecs


def fetch_kegg_reaction_list() -> List[Dict]:
    """All KEGG reactions with their definition strings."""
    txt = _get(f"{KEGG}/list/reaction")
    if not txt:
        return []
    out = []
    for line in txt.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append({"rn": parts[0].replace("rn:", ""),
                        "definition": parts[1]})
    return out


def fetch_kegg_entries(ids: List[str], batch: int = 10) -> Dict[str, str]:
    """
    Fetch full flat-file entries.  KEGG allows up to 10 ids per GET.
    Returns id -> raw entry text.
    """
    out: Dict[str, str] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        txt = _get(f"{KEGG}/get/{'+'.join(chunk)}")
        if not txt:
            continue
        # entries are separated by '///'; the header line is
        #   "ENTRY       EC 1.1.1.1                  Enzyme"
        # so the identifier is the token AFTER the type word ('EC'), not the
        # second token.
        for blob in txt.split("///"):
            blob = blob.strip()
            if not blob:
                continue
            toks = blob.split("\n")[0].split()
            key = None
            if len(toks) >= 3 and toks[0] == "ENTRY" and toks[1] == "EC":
                key = toks[2]
            elif len(toks) >= 2 and toks[0] == "ENTRY":
                key = toks[1]
            if key:
                out[key] = blob
    return out


def parse_kegg_enzyme_entry(entry: str) -> Dict:
    """Extract the fields relevant to aperture counting from a KEGG EC entry."""
    rec: Dict[str, object] = {
        "ec": None, "name": None, "sysname": None,
        "reaction_lines": [], "n_reactions": 0,
        "substrates": [], "products": [], "cofactors": [],
        "comment": "",
    }
    section = None
    for line in entry.split("\n"):
        if not line:
            continue
        if not line.startswith(" "):
            head = line.split()[0]
            section = head
            rest = line[len(head):].strip()
        else:
            rest = line.strip()

        if section == "ENTRY":
            toks = rest.split()
            # "EC 1.1.1.1  Enzyme"  ->  want "1.1.1.1"
            if toks and toks[0] == "EC" and len(toks) > 1:
                rec["ec"] = toks[1]
            elif toks:
                rec["ec"] = toks[0]
        elif section == "NAME" and rec["name"] is None:
            rec["name"] = rest.rstrip(";")
        elif section == "SYSNAME" and rec["sysname"] is None:
            rec["sysname"] = rest.rstrip(";")
        elif section == "REACTION":
            rec["reaction_lines"].append(rest)
        elif section == "SUBSTRATE":
            rec["substrates"].append(rest.rstrip(";"))
        elif section == "PRODUCT":
            rec["products"].append(rest.rstrip(";"))
        elif section == "COFACTOR":
            rec["cofactors"].append(rest.rstrip(";"))
        elif section == "COMMENT":
            rec["comment"] += " " + rest

    rec["n_reactions"] = len(rec["reaction_lines"])
    rec["n_substrates"] = len(rec["substrates"])
    rec["n_products"] = len(rec["products"])
    rec["n_cofactors"] = len(rec["cofactors"])
    return rec


# ---------------------------------------------------------------------------
# Reactome
# ---------------------------------------------------------------------------
def fetch_reactome_version() -> Optional[str]:
    v = _get(f"{REACTOME}/data/database/version")
    return v.strip() if isinstance(v, str) else None


def fetch_reactome_pathways(species: str = "9606", limit: int = 400) -> List[Dict]:
    """Top-level human pathways, then their contained events."""
    top = _get(f"{REACTOME}/data/pathways/top/{species}", accept_json=True)
    if not isinstance(top, list):
        return []
    out = []
    for p in top[:limit]:
        out.append({
            "stId": p.get("stId"),
            "displayName": p.get("displayName"),
            "speciesName": p.get("speciesName"),
        })
    return out


def fetch_reactome_pathway_events(st_id: str) -> List[Dict]:
    """Reaction-like events contained in a pathway (one level down)."""
    ev = _get(f"{REACTOME}/data/pathway/{st_id}/containedEvents",
              accept_json=True)
    if not isinstance(ev, list):
        return []
    out = []
    for e in ev:
        # the endpoint occasionally returns bare ids alongside objects
        if not isinstance(e, dict):
            continue
        out.append({
            "stId": e.get("stId"),
            "displayName": e.get("displayName"),
            "schemaClass": e.get("schemaClass"),
        })
    return out


def fetch_reactome_reaction_detail(st_id: str) -> Optional[Dict]:
    d = _get(f"{REACTOME}/data/query/{st_id}", accept_json=True)
    if not isinstance(d, dict):
        return None
    # Reactome sometimes returns bare dbIds instead of embedded objects in
    # these arrays; count them but only take names from real objects.
    inputs = d.get("input") or []
    outputs = d.get("output") or []
    cats = d.get("catalystActivity") or []

    def names(seq):
        return [x.get("displayName") for x in seq if isinstance(x, dict)][:12]

    return {
        "stId": d.get("stId"),
        "displayName": d.get("displayName"),
        "schemaClass": d.get("schemaClass"),
        "category": d.get("category"),
        "n_input": len(inputs),
        "n_output": len(outputs),
        "n_catalyst": len(cats),
        "isChimeric": d.get("isChimeric"),
        "input_names": names(inputs),
        "output_names": names(outputs),
    }


# ---------------------------------------------------------------------------
def main(n_enzymes: int = 400, n_pathways: int = 25,
         n_reactions_per_pathway: int = 40) -> Dict:
    os.makedirs(CACHE_DIR, exist_ok=True)

    manifest: Dict[str, object] = {
        "script": "v6_fetch_reaction_data.py",
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {},
    }

    # ---------------- KEGG ----------------
    kegg_enz_path = os.path.join(CACHE_DIR, "kegg_enzymes.json")
    kegg_rxn_path = os.path.join(CACHE_DIR, "kegg_reactions.json")
    if os.path.exists(kegg_enz_path) and os.path.exists(kegg_rxn_path):
        with open(kegg_enz_path) as fh:
            cached_parsed = json.load(fh)
        with open(kegg_rxn_path) as fh:
            cached_rns = json.load(fh)
        if len(cached_parsed) > 100:
            print(f"[V6] KEGG: using cache ({len(cached_parsed)} EC entries, "
                  f"{len(cached_rns)} reactions)")
            manifest["sources"]["kegg"] = {
                "endpoint": KEGG,
                "n_reactions_total": len(cached_rns),
                "n_ec_parsed": len(cached_parsed),
                "from_cache": True,
                "cache_files": ["cache/kegg_enzymes.json",
                                "cache/kegg_reactions.json"],
            }
            return _reactome_stage(manifest, n_pathways,
                                   n_reactions_per_pathway)

    print("[V6] KEGG: enzyme list ...")
    ecs = fetch_kegg_enzyme_list()
    print(f"      {len(ecs)} EC numbers")

    print("[V6] KEGG: reaction list ...")
    rns = fetch_kegg_reaction_list()
    print(f"      {len(rns)} reactions")

    # sample EC entries spread across all six top-level classes
    sample: List[str] = []
    if ecs:
        by_class: Dict[str, List[str]] = {}
        for ec in ecs:
            cls = ec.split(".")[0]
            by_class.setdefault(cls, []).append(ec)
        per = max(1, n_enzymes // max(1, len(by_class)))
        for cls in sorted(by_class):
            sample.extend(by_class[cls][:per])
    sample = sample[:n_enzymes]

    print(f"[V6] KEGG: fetching {len(sample)} EC entries ...")
    raw_entries = fetch_kegg_entries(sample)
    parsed = [parse_kegg_enzyme_entry(v) for v in raw_entries.values()]
    parsed = [p for p in parsed if p.get("ec")]
    print(f"      parsed {len(parsed)} entries")

    with open(os.path.join(CACHE_DIR, "kegg_enzymes.json"), "w") as fh:
        json.dump(parsed, fh, indent=2)
    with open(os.path.join(CACHE_DIR, "kegg_reactions.json"), "w") as fh:
        json.dump(rns, fh, indent=2)

    manifest["sources"]["kegg"] = {
        "endpoint": KEGG,
        "n_ec_total": len(ecs),
        "n_reactions_total": len(rns),
        "n_ec_sampled": len(sample),
        "n_ec_parsed": len(parsed),
        "cache_files": ["cache/kegg_enzymes.json", "cache/kegg_reactions.json"],
    }

    return _reactome_stage(manifest, n_pathways, n_reactions_per_pathway)


def _reactome_stage(manifest: Dict, n_pathways: int,
                    n_reactions_per_pathway: int) -> Dict:
    """Retrieve Reactome pathways and reaction details, then write manifest."""
    print("[V6] Reactome: version ...")
    version = fetch_reactome_version()
    print(f"      version {version}")

    print("[V6] Reactome: top-level human pathways ...")
    pathways = fetch_reactome_pathways()
    print(f"      {len(pathways)} pathways")

    reactions: List[Dict] = []
    for p in pathways[:n_pathways]:
        st = p.get("stId")
        if not st:
            continue
        events = fetch_reactome_pathway_events(st)
        rxn_events = [e for e in events
                      if e.get("schemaClass") in
                      ("Reaction", "BlackBoxEvent", "Polymerisation",
                       "Depolymerisation", "FailedReaction")]
        for e in rxn_events[:n_reactions_per_pathway]:
            if not e.get("stId"):
                continue
            d = fetch_reactome_reaction_detail(e["stId"])
            if d:
                d["pathway"] = p.get("displayName")
                d["pathway_stId"] = st
                reactions.append(d)
        print(f"      {str(p.get('displayName','?'))[:45]:45s} "
              f"{len(rxn_events):4d} events, cumulative {len(reactions)}")

    with open(os.path.join(CACHE_DIR, "reactome_reactions.json"), "w") as fh:
        json.dump(reactions, fh, indent=2)
    with open(os.path.join(CACHE_DIR, "reactome_pathways.json"), "w") as fh:
        json.dump(pathways, fh, indent=2)

    manifest["sources"]["reactome"] = {
        "endpoint": REACTOME,
        "database_version": version,
        "n_top_pathways": len(pathways),
        "n_pathways_traversed": min(n_pathways, len(pathways)),
        "n_reactions_retrieved": len(reactions),
        "cache_files": ["cache/reactome_pathways.json",
                        "cache/reactome_reactions.json"],
    }

    out = os.path.join(RESULTS_DIR, "v6_fetch_manifest.json")
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[V6] manifest -> {out}")
    return manifest


if __name__ == "__main__":
    main()
