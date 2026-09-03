"""
Offline fixtures mirroring the SHAPE of the four source kinds the questions
touch.  These are not copies of the public resources; they are small,
hand-checkable stand-ins with the same record structure, so that every number
in the paper is reproducible without a network.

Source kinds:
  RXN   reaction knowledge base   (SPARQL; Rhea-shaped)
  PROT  protein resource          (SPARQL + REST; UniProt-shaped)
  PATH  pathway resource          (flat-file REST)
  ELN   electronic lab notebook   (LARA-shaped; device/operator/buffer records)
"""

# --------------------------------------------------------------------------
# RXN -- reactions.  Master reaction + two directional children, as Rhea does.
# --------------------------------------------------------------------------
REACTIONS = {
    "RXN:19453": {
        "equation": "L-alanine + 2-oxoglutarate = pyruvate + L-glutamate",
        "ec": "2.6.1.2",
        "status": "approved",
        "substrates": ["CHEBI:57972", "CHEBI:16810"],   # L-Ala, 2-OG
        "products": ["CHEBI:15361", "CHEBI:29985"],     # pyruvate, L-Glu
        "directional": ["RXN:19454", "RXN:19455"],
        "bidirectional": True,
    },
    "RXN:20504": {
        "equation": "benzylethylamine + pyruvate = benzylethylketone + L-alanine",
        "ec": "2.6.1.-",
        "status": "approved",
        "substrates": ["CHEBI:90000", "CHEBI:15361"],
        "products": ["CHEBI:90001", "CHEBI:57972"],
        "directional": ["RXN:20505", "RXN:20506"],
        "bidirectional": True,
    },
    "RXN:31000": {
        "equation": "cyclohexanone + NADPH + O2 = epsilon-caprolactone + NADP+ + H2O",
        "ec": "1.14.13.22",
        "status": "approved",
        "substrates": ["CHEBI:17854", "CHEBI:57783", "CHEBI:15379"],
        "products": ["CHEBI:35604", "CHEBI:58349", "CHEBI:15377"],
        "directional": ["RXN:31001", "RXN:31002"],
        "bidirectional": True,
    },
}

# --------------------------------------------------------------------------
# PROT -- proteins.  Sequences are real-shaped but synthetic.
# --------------------------------------------------------------------------
PROTEINS = {
    "PR:AlaA_ECOLI": {
        "name": "alanine transaminase AlaA",
        "organism": "Escherichia coli",
        "lineage": ["Bacteria", "Pseudomonadota", "Gammaproteobacteria"],
        "domain": "Bacteria",
        "ec": "2.6.1.2",
        "catalyses": ["RXN:19453"],
        "sequence": "MADTRPERLSAFGSSFLDAMRLKAQGHDVLNFSAGEPDF",   # no C
    },
    "PR:ALT1_HUMAN": {
        "name": "alanine aminotransferase 1",
        "organism": "Homo sapiens",
        "lineage": ["Eukaryota", "Metazoa", "Chordata"],
        "domain": "Eukaryota",
        "ec": "2.6.1.2",
        "catalyses": ["RXN:19453"],
        "sequence": "MASSTGDRSQAVRHGLRAKVLTLDGMNPRVRRVEYAVRGPIC",  # has C
    },
    "PR:TAM_BACIL": {
        "name": "omega-transaminase",
        "organism": "Bacillus megaterium",
        "lineage": ["Bacteria", "Bacillota", "Bacilli"],
        "domain": "Bacteria",
        "ec": "2.6.1.-",
        "catalyses": ["RXN:20504"],
        "sequence": "MSFNAEQLNQIDAAHHLHPFTDMKSLNQAGARVMTRGEGVYLWD",  # no C
    },
    "PR:TAM_PSEUD": {
        "name": "omega-transaminase",
        "organism": "Pseudomonas fluorescens",
        "lineage": ["Bacteria", "Pseudomonadota", "Gammaproteobacteria"],
        "domain": "Bacteria",
        "ec": "2.6.1.-",
        "catalyses": ["RXN:20504"],
        "sequence": "MTQPLNVAECRALDAAHHLHPFTSLKALNEQGACVITKAEGAYIYD",  # has C
    },
    "PR:TAM_ARATH": {
        "name": "transaminase",
        "organism": "Arabidopsis thaliana",
        "lineage": ["Eukaryota", "Viridiplantae", "Streptophyta"],
        "domain": "Eukaryota",
        "ec": "2.6.1.-",
        "catalyses": ["RXN:20504"],
        "sequence": "MSLNTEQLNAIDAAHHLHPFTDMKSLNEKGSRVITRAEGVYLWD",  # no C
    },
    "PR:BVMO_ACINE": {
        "name": "cyclohexanone monooxygenase",
        "organism": "Acinetobacter calcoaceticus",
        "lineage": ["Bacteria", "Pseudomonadota", "Gammaproteobacteria"],
        "domain": "Bacteria",
        "ec": "1.14.13.22",
        "catalyses": ["RXN:31000"],
        "sequence": "MSQKMDFDAIVIGGGFGGLYAVKKLRDELELKVQAFDKATDVGGTWYWNRYPGA",
    },
}

# --------------------------------------------------------------------------
# PATH -- pathways (flat-file REST; no SPARQL)
# --------------------------------------------------------------------------
PATHWAYS = {
    "PW:00250": {
        "name": "Alanine, aspartate and glutamate metabolism",
        "reactions": ["RXN:19453"],
    },
    "PW:00930": {
        "name": "Caprolactam degradation",
        "reactions": ["RXN:31000"],
    },
}

# --------------------------------------------------------------------------
# ELN -- laboratory records (LARA-shaped).  This is where conditions live.
# --------------------------------------------------------------------------
EXPERIMENTS = {
    "BT3": {
        "title": "biocatalytic transformation BT3",
        "operator": "Y. Dikova",
        "date": "2026-03-23",
        "reaction": "RXN:20504",
        "biocatalyst": "PR:TAM_BACIL",
        "buffer": {"name": "HEPES", "conc_mM": 50, "pH": 7.5},
        "temperature_C": 30,
        "device": {
            "id": "DEV:UV-1900i",
            "kind": "UV-vis spectrophotometer",
            "vendor": "Shimadzu",
            "settings": {"wavelength_nm": 245, "bandwidth_nm": 1.0},
        },
        "datasets": ["DS:0031"],
    },
    "MT7": {
        "title": "methyl transfer with mt-X",
        "operator": "M. Doerr",
        "date": "2026-02-11",
        "reaction": None,
        "biocatalyst": "PR:MTX_BACSU",
        "buffer": {"name": "Tris-HCl", "conc_mM": 100, "pH": 8.0},
        "temperature_C": 37,
        "device": {
            "id": "DEV:AVANCE-400",
            "kind": "NMR spectrometer",
            "vendor": "Bruker",
            "settings": {"field_MHz": 400, "nucleus": "1H"},
        },
        "datasets": ["DS:0017"],
    },
    "KR9": {
        "title": "kinetic resolution with PFE",
        "operator": "Y. Dikova",
        "date": "2026-04-02",
        "reaction": None,
        "biocatalyst": "PR:PFE_PSEFL",
        "buffer": {"name": "HEPES", "conc_mM": 50, "pH": 9.0},
        "temperature_C": 25,
        "device": {
            "id": "DEV:UV-1900i",
            "kind": "UV-vis spectrophotometer",
            "vendor": "Shimadzu",
            "settings": {"wavelength_nm": 410, "bandwidth_nm": 2.0},
        },
        "datasets": ["DS:0044"],
    },
    "BV2": {
        "title": "BVMO-Y substrate scope screen",
        "operator": "D. Linke",
        "date": "2026-05-14",
        "reaction": "RXN:31000",
        "biocatalyst": "PR:BVMO_ACINE",
        "buffer": {"name": "phosphate", "conc_mM": 50, "pH": 7.0},
        "temperature_C": 30,
        "device": {
            "id": "DEV:AVANCE-400",
            "kind": "NMR spectrometer",
            "vendor": "Bruker",
            "settings": {"field_MHz": 400, "nucleus": "13C"},
        },
        "datasets": ["DS:0052", "DS:0053"],
    },
    # Same compound, different instrument kind: separates "about compound C"
    # from "measured on a UV-vis instrument".
    "BV3": {
        "title": "BVMO-Y conversion time course",
        "operator": "D. Linke",
        "date": "2026-05-15",
        "reaction": "RXN:31000",
        "biocatalyst": "PR:BVMO_ACINE",
        "buffer": {"name": "phosphate", "conc_mM": 50, "pH": 7.0},
        "temperature_C": 30,
        "device": {
            "id": "DEV:UV-1900i",
            "kind": "UV-vis spectrophotometer",
            "vendor": "Shimadzu",
            "settings": {"wavelength_nm": 340, "bandwidth_nm": 1.0},
        },
        "datasets": ["DS:0061"],
    },
    # Same compound, same vendor, DIFFERENT setting: separates "on a Bruker"
    # from "on a Bruker set to X".
    "BV4": {
        "title": "BVMO-Y proton check",
        "operator": "D. Linke",
        "date": "2026-05-16",
        "reaction": "RXN:31000",
        "biocatalyst": "PR:BVMO_ACINE",
        "buffer": {"name": "phosphate", "conc_mM": 50, "pH": 7.0},
        "temperature_C": 30,
        "device": {
            "id": "DEV:AVANCE-400",
            "kind": "NMR spectrometer",
            "vendor": "Bruker",
            "settings": {"field_MHz": 400, "nucleus": "1H"},
        },
        "datasets": ["DS:0062"],
    },
}

DATASETS = {
    "DS:0017": {"experiment": "MT7", "type": "NMR", "compounds": ["CHEBI:17790"]},
    "DS:0031": {
        "experiment": "BT3",
        "type": "UV-vis",
        "compounds": ["CHEBI:90000", "CHEBI:15361"],
    },
    "DS:0044": {
        "experiment": "KR9",
        "type": "UV-vis",
        "compounds": ["CHEBI:33308"],
    },
    "DS:0052": {"experiment": "BV2", "type": "NMR", "compounds": ["CHEBI:17854"]},
    "DS:0053": {"experiment": "BV2", "type": "NMR", "compounds": ["CHEBI:35604"]},
    # Cyclohexanone also measured on UV-vis in a separate run, and on a
    # 1H (not 13C) Bruker experiment.  These make the Chem-DCAT-AP filters
    # discriminating: each added constraint must visibly narrow the result,
    # otherwise a query returning one row proves nothing about the filter.
    "DS:0061": {"experiment": "BV3", "type": "UV-vis", "compounds": ["CHEBI:17854"]},
    "DS:0062": {"experiment": "BV4", "type": "NMR", "compounds": ["CHEBI:17854"]},
}

# --------------------------------------------------------------------------
# Chemistry: contact graphs for the medium/direction results.
# Ambient occupancies are illustrative, as the paper states.
# --------------------------------------------------------------------------
AMBIENT = {
    # a cytosol in which 2-oxoglutarate is ambient and L-glutamate is held low
    "cytosol_gln_depleted": {
        "CHEBI:57972": 2.0e-3,     # L-alanine
        "CHEBI:16810": 1.0e-4,     # 2-oxoglutarate  (ambient)
        "CHEBI:15361": 5.0e-4,     # pyruvate
        "CHEBI:29985": 1.0e-6,     # L-glutamate     (depleted)
    },
    # an organism running the same enzyme the other way
    "cytosol_og_depleted": {
        "CHEBI:57972": 2.0e-3,
        "CHEBI:16810": 1.0e-6,     # 2-oxoglutarate  (depleted)
        "CHEBI:15361": 5.0e-4,
        "CHEBI:29985": 1.0e-4,     # L-glutamate     (ambient)
    },
    # a balanced medium: no direction
    "balanced": {
        "CHEBI:57972": 1.0e-4,
        "CHEBI:16810": 1.0e-4,
        "CHEBI:15361": 1.0e-4,
        "CHEBI:29985": 1.0e-4,
    },
}

CHEBI_NAMES = {
    "CHEBI:57972": "L-alanine",
    "CHEBI:16810": "2-oxoglutarate",
    "CHEBI:15361": "pyruvate",
    "CHEBI:29985": "L-glutamate",
    "CHEBI:90000": "benzylethylamine",
    "CHEBI:90001": "benzylethylketone",
    "CHEBI:17854": "cyclohexanone",
    "CHEBI:35604": "epsilon-caprolactone",
    "CHEBI:17790": "methanol",
    "CHEBI:33308": "carboxylic ester",
    "CHEBI:15377": "water",
    "CHEBI:15379": "dioxygen",
}
