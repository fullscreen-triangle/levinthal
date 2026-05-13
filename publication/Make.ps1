#Requires -Version 5.1
<#
.SYNOPSIS
    PowerShell equivalent of the Levinthal Monograph Makefile.

.DESCRIPTION
    Run from the publication\ directory.

.PARAMETER Target
    all      - compile all papers then the master book  (default)
    papers   - compile all papers in dependency order
    book     - compile the master book (papers must exist as PDFs)
    clean    - remove auxiliary files (keeps PDFs)
    clobber  - remove auxiliary files AND all PDFs
    help     - print this message
    bpl, intro, p1 .. p15  - compile a single paper

.EXAMPLE
    .\Make.ps1 all
    .\Make.ps1 papers
    .\Make.ps1 book
    .\Make.ps1 p9
    .\Make.ps1 clean
#>
param(
    [Parameter(Position = 0)]
    [string]$Target = "all"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
$LMK_FLAGS = "-pdf", "-bibtex", "-interaction=nonstopmode", "-halt-on-error"
$P = "..\cytochrome\publications"   # relative to publication\

# ---------------------------------------------------------------------------
# Helper: run latexmk in a given directory, then return
# ---------------------------------------------------------------------------
function Invoke-Lmk {
    param(
        [string]$Dir,
        [string]$File
    )
    $abs = Resolve-Path $Dir -ErrorAction Stop
    Write-Host "`n==> latexmk $File  (in $abs)" -ForegroundColor Cyan
    Push-Location $abs
    try {
        & latexmk @LMK_FLAGS $File
        if ($LASTEXITCODE -ne 0) {
            throw "latexmk exited with code $LASTEXITCODE for $File"
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-LmkClean {
    param(
        [string]$Dir,
        [string]$File,
        [switch]$Full   # -Full => -C (remove PDFs too); otherwise -c
    )
    $abs = Resolve-Path $Dir -ErrorAction SilentlyContinue
    if (-not $abs) { return }
    $flag = if ($Full) { "-C" } else { "-c" }
    Push-Location $abs
    try { & latexmk $flag $File } finally { Pop-Location }
}

# ---------------------------------------------------------------------------
# Individual paper build functions
# ---------------------------------------------------------------------------
function Build-BPL   { Invoke-Lmk "biological-partition-landscape"                                     "biological-partition-landscape.tex" }
function Build-Intro { Invoke-Lmk "$P\introduction"                                                    "monograph-introduction.tex" }
function Build-P1    { Invoke-Lmk "$P\foundations\expression-algebra-for-biomolecules"                 "expression-algebra-proteins.tex" }
function Build-P2    { Invoke-Lmk "$P\manifold\p450-address-manifold-cyp3a4-fold"                      "p450-manifold-cyp3a4-fold.tex" }
function Build-P25   { Invoke-Lmk "$P\foundations\glb-structural-input"                                "glb-structural-input.tex" }
function Build-P3    { Invoke-Lmk "$P\equilibrium-states\cyp3a4-resting-substrate-bound"               "cyp3a4-resting-substrate-bound.tex" }
function Build-P4    { Invoke-Lmk "$P\catalytic-cycle\multi-hop-et-chain"                              "multi-hop-et-chain.tex" }
function Build-P5    { Invoke-Lmk "$P\catalytic-cycle\compound-i-formation"                            "compound-i-formation.tex" }
function Build-P6    { Invoke-Lmk "$P\catalytic-cycle\ch-activation-rebound"                           "ch-activation-rebound.tex" }
function Build-P7    { Invoke-Lmk "$P\reactions\heteroatom-dealkylation"                               "heteroatom-dealkylation.tex" }
function Build-P8    { Invoke-Lmk "$P\reactions\atypical-reactions-atlas"                              "atypical-reactions-atlas.tex" }
function Build-P9    { Invoke-Lmk "$P\diversity\57-human-isoforms"                                     "57-human-isoforms.tex" }
function Build-P10   { Invoke-Lmk "$P\pharmacology\pharmacogenomics-atlas"                             "pharmacogenomics-atlas.tex" }
function Build-P11   { Invoke-Lmk "$P\construction\membrane-cofactor-cpr"                              "membrane-cpr.tex" }
function Build-P12   { Invoke-Lmk "$P\synthesis\seven-state-closed-orbit"                              "closed-orbit.tex" }
function Build-P13   { Invoke-Lmk "$P\spectroscopy\spectroscopic-atlas"                                "spectroscopic-atlas.tex" }
function Build-P14   { Invoke-Lmk "$P\informatics\database-recovery"                                   "database-recovery.tex" }
function Build-P15   { Invoke-Lmk "$P\diversity\polymorphisms-ddi-inhibitors"                          "polymorphisms-ddi-inhibitors.tex" }

function Build-Book {
    Invoke-Lmk "." "cytochrome-p450-monograph.tex"
}

function Build-Papers {
    Build-BPL; Build-Intro
    Build-P1;  Build-P2;  Build-P25; Build-P3;  Build-P4
    Build-P5;  Build-P6;  Build-P7;  Build-P8;  Build-P9
    Build-P10; Build-P11; Build-P12; Build-P13; Build-P14; Build-P15
}

# ---------------------------------------------------------------------------
# Clean helpers
# ---------------------------------------------------------------------------
function Clean-All {
    param([switch]$Full)
    $flag = $Full
    Invoke-LmkClean "."                                                           "cytochrome-p450-monograph.tex"          -Full:$flag
    Invoke-LmkClean "biological-partition-landscape"                              "biological-partition-landscape.tex"     -Full:$flag
    Invoke-LmkClean "$P\introduction"                                             "monograph-introduction.tex"             -Full:$flag
    Invoke-LmkClean "$P\foundations\expression-algebra-for-biomolecules"         "expression-algebra-proteins.tex"        -Full:$flag
    Invoke-LmkClean "$P\manifold\p450-address-manifold-cyp3a4-fold"              "p450-manifold-cyp3a4-fold.tex"          -Full:$flag
    Invoke-LmkClean "$P\foundations\glb-structural-input"                        "glb-structural-input.tex"               -Full:$flag
    Invoke-LmkClean "$P\equilibrium-states\cyp3a4-resting-substrate-bound"       "cyp3a4-resting-substrate-bound.tex"     -Full:$flag
    Invoke-LmkClean "$P\catalytic-cycle\multi-hop-et-chain"                      "multi-hop-et-chain.tex"                 -Full:$flag
    Invoke-LmkClean "$P\catalytic-cycle\compound-i-formation"                    "compound-i-formation.tex"               -Full:$flag
    Invoke-LmkClean "$P\catalytic-cycle\ch-activation-rebound"                   "ch-activation-rebound.tex"              -Full:$flag
    Invoke-LmkClean "$P\reactions\heteroatom-dealkylation"                       "heteroatom-dealkylation.tex"            -Full:$flag
    Invoke-LmkClean "$P\reactions\atypical-reactions-atlas"                      "atypical-reactions-atlas.tex"           -Full:$flag
    Invoke-LmkClean "$P\diversity\57-human-isoforms"                             "57-human-isoforms.tex"                  -Full:$flag
    Invoke-LmkClean "$P\pharmacology\pharmacogenomics-atlas"                     "pharmacogenomics-atlas.tex"             -Full:$flag
    Invoke-LmkClean "$P\construction\membrane-cofactor-cpr"                      "membrane-cpr.tex"                       -Full:$flag
    Invoke-LmkClean "$P\synthesis\seven-state-closed-orbit"                      "closed-orbit.tex"                       -Full:$flag
    Invoke-LmkClean "$P\spectroscopy\spectroscopic-atlas"                        "spectroscopic-atlas.tex"                -Full:$flag
    Invoke-LmkClean "$P\informatics\database-recovery"                           "database-recovery.tex"                  -Full:$flag
    Invoke-LmkClean "$P\diversity\polymorphisms-ddi-inhibitors"                  "polymorphisms-ddi-inhibitors.tex"       -Full:$flag
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch ($Target.ToLower()) {
    "all"     { Build-Papers; Build-Book }
    "papers"  { Build-Papers }
    "book"    { Build-Book }
    "bpl"     { Build-BPL }
    "intro"   { Build-Intro }
    "p1"      { Build-P1 }
    "p2"      { Build-P2 }
    "p25"     { Build-P25 }
    "p3"      { Build-P3 }
    "p4"      { Build-P4 }
    "p5"      { Build-P5 }
    "p6"      { Build-P6 }
    "p7"      { Build-P7 }
    "p8"      { Build-P8 }
    "p9"      { Build-P9 }
    "p10"     { Build-P10 }
    "p11"     { Build-P11 }
    "p12"     { Build-P12 }
    "p13"     { Build-P13 }
    "p14"     { Build-P14 }
    "p15"     { Build-P15 }
    "clean"   { Clean-All }
    "clobber" { Clean-All -Full }
    "help"    {
        Write-Host @"
Levinthal Monograph — PowerShell build script
  .\Make.ps1 all      compile all papers then the book
  .\Make.ps1 papers   compile all papers in order
  .\Make.ps1 book     compile the book (papers must already be PDFs)
  .\Make.ps1 clean    remove auxiliary files (keeps PDFs)
  .\Make.ps1 clobber  remove auxiliary files AND all PDFs
  .\Make.ps1 p1 .. p15  compile a single paper
  .\Make.ps1 bpl        compile the BPL foundational paper
  .\Make.ps1 intro      compile the prefatory introduction
"@
    }
    default {
        Write-Error "Unknown target '$Target'. Run .\Make.ps1 help for usage."
    }
}
