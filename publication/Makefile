# =============================================================================
# Makefile — Levinthal Monograph
# =============================================================================
# Run from the publication/ directory.
#
# Targets:
#   make all        compile all papers then the master book
#   make papers     compile all papers only (in dependency order)
#   make book       compile the master book (papers must exist as PDFs)
#   make clean      remove all auxiliary files (keeps PDFs)
#   make clobber    remove auxiliary files AND all PDFs
#   make help       print this message
#
# Requires: latexmk, pdflatex, bibtex (TeX Live or MiKTeX)
# Shell:    bash (POSIX)
# =============================================================================

SHELL  := /bin/bash
LMK    := latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error

# Cytochrome publications root (relative to this Makefile)
P := ../cytochrome/publications

.PHONY: all papers book clean clobber help \
	bpl intro p1 p2 p25 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15

# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------

all: papers book

papers: bpl intro p1 p2 p25 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15

book:
	$(LMK) cytochrome-p450-monograph.tex

help:
	@echo "Levinthal Monograph — Makefile targets"
	@echo "  all      compile all papers then the book"
	@echo "  papers   compile all papers in order"
	@echo "  book     compile the book (papers must already be PDFs)"
	@echo "  clean    remove auxiliary files (keeps PDFs)"
	@echo "  clobber  remove auxiliary files AND all PDFs"
	@echo ""
	@echo "Individual paper targets: bpl intro p1 p2 p25 p3-p15"

# -----------------------------------------------------------------------------
# Foundational paper
# -----------------------------------------------------------------------------

bpl:
	cd biological-partition-landscape && $(LMK) biological-partition-landscape.tex

# -----------------------------------------------------------------------------
# Prefatory paper (introduction)
# -----------------------------------------------------------------------------

intro:
	cd $(P)/introduction && $(LMK) monograph-introduction.tex

# -----------------------------------------------------------------------------
# Papers 1–15
# -----------------------------------------------------------------------------

p1:
	cd $(P)/foundations/expression-algebra-for-biomolecules \
	  && $(LMK) expression-algebra-proteins.tex

p2:
	cd $(P)/manifold/p450-address-manifold-cyp3a4-fold \
	  && $(LMK) p450-manifold-cyp3a4-fold.tex

p25:
	cd $(P)/foundations/glb-structural-input \
	  && $(LMK) glb-structural-input.tex

p3:
	cd $(P)/equilibrium-states/cyp3a4-resting-substrate-bound \
	  && $(LMK) cyp3a4-resting-substrate-bound.tex

p4:
	cd $(P)/catalytic-cycle/multi-hop-et-chain \
	  && $(LMK) multi-hop-et-chain.tex

p5:
	cd $(P)/catalytic-cycle/compound-i-formation \
	  && $(LMK) compound-i-formation.tex

p6:
	cd $(P)/catalytic-cycle/ch-activation-rebound \
	  && $(LMK) ch-activation-rebound.tex

p7:
	cd $(P)/reactions/heteroatom-dealkylation \
	  && $(LMK) heteroatom-dealkylation.tex

p8:
	cd $(P)/reactions/atypical-reactions-atlas \
	  && $(LMK) atypical-reactions-atlas.tex

p9:
	cd $(P)/diversity/57-human-isoforms \
	  && $(LMK) 57-human-isoforms.tex

p10:
	cd $(P)/pharmacology/pharmacogenomics-atlas \
	  && $(LMK) pharmacogenomics-atlas.tex

p11:
	cd $(P)/construction/membrane-cofactor-cpr \
	  && $(LMK) membrane-cpr.tex

p12:
	cd $(P)/synthesis/seven-state-closed-orbit \
	  && $(LMK) closed-orbit.tex

p13:
	cd $(P)/spectroscopy/spectroscopic-atlas \
	  && $(LMK) spectroscopic-atlas.tex

p14:
	cd $(P)/informatics/database-recovery \
	  && $(LMK) database-recovery.tex

p15:
	cd $(P)/diversity/polymorphisms-ddi-inhibitors \
	  && $(LMK) polymorphisms-ddi-inhibitors.tex

# -----------------------------------------------------------------------------
# Housekeeping
# -----------------------------------------------------------------------------

clean:
	latexmk -c cytochrome-p450-monograph.tex
	cd biological-partition-landscape            && latexmk -c biological-partition-landscape.tex
	cd $(P)/introduction                         && latexmk -c monograph-introduction.tex
	cd $(P)/foundations/expression-algebra-for-biomolecules \
	                                             && latexmk -c expression-algebra-proteins.tex
	cd $(P)/manifold/p450-address-manifold-cyp3a4-fold \
	                                             && latexmk -c p450-manifold-cyp3a4-fold.tex
	cd $(P)/foundations/glb-structural-input     && latexmk -c glb-structural-input.tex
	cd $(P)/equilibrium-states/cyp3a4-resting-substrate-bound \
	                                             && latexmk -c cyp3a4-resting-substrate-bound.tex
	cd $(P)/catalytic-cycle/multi-hop-et-chain   && latexmk -c multi-hop-et-chain.tex
	cd $(P)/catalytic-cycle/compound-i-formation && latexmk -c compound-i-formation.tex
	cd $(P)/catalytic-cycle/ch-activation-rebound \
	                                             && latexmk -c ch-activation-rebound.tex
	cd $(P)/reactions/heteroatom-dealkylation    && latexmk -c heteroatom-dealkylation.tex
	cd $(P)/reactions/atypical-reactions-atlas   && latexmk -c atypical-reactions-atlas.tex
	cd $(P)/diversity/57-human-isoforms          && latexmk -c 57-human-isoforms.tex
	cd $(P)/pharmacology/pharmacogenomics-atlas  && latexmk -c pharmacogenomics-atlas.tex
	cd $(P)/construction/membrane-cofactor-cpr   && latexmk -c membrane-cpr.tex
	cd $(P)/synthesis/seven-state-closed-orbit   && latexmk -c closed-orbit.tex
	cd $(P)/spectroscopy/spectroscopic-atlas     && latexmk -c spectroscopic-atlas.tex
	cd $(P)/informatics/database-recovery        && latexmk -c database-recovery.tex
	cd $(P)/diversity/polymorphisms-ddi-inhibitors \
	                                             && latexmk -c polymorphisms-ddi-inhibitors.tex

clobber:
	latexmk -C cytochrome-p450-monograph.tex
	cd biological-partition-landscape            && latexmk -C biological-partition-landscape.tex
	cd $(P)/introduction                         && latexmk -C monograph-introduction.tex
	cd $(P)/foundations/expression-algebra-for-biomolecules \
	                                             && latexmk -C expression-algebra-proteins.tex
	cd $(P)/manifold/p450-address-manifold-cyp3a4-fold \
	                                             && latexmk -C p450-manifold-cyp3a4-fold.tex
	cd $(P)/foundations/glb-structural-input     && latexmk -C glb-structural-input.tex
	cd $(P)/equilibrium-states/cyp3a4-resting-substrate-bound \
	                                             && latexmk -C cyp3a4-resting-substrate-bound.tex
	cd $(P)/catalytic-cycle/multi-hop-et-chain   && latexmk -C multi-hop-et-chain.tex
	cd $(P)/catalytic-cycle/compound-i-formation && latexmk -C compound-i-formation.tex
	cd $(P)/catalytic-cycle/ch-activation-rebound \
	                                             && latexmk -C ch-activation-rebound.tex
	cd $(P)/reactions/heteroatom-dealkylation    && latexmk -C heteroatom-dealkylation.tex
	cd $(P)/reactions/atypical-reactions-atlas   && latexmk -C atypical-reactions-atlas.tex
	cd $(P)/diversity/57-human-isoforms          && latexmk -C 57-human-isoforms.tex
	cd $(P)/pharmacology/pharmacogenomics-atlas  && latexmk -C pharmacogenomics-atlas.tex
	cd $(P)/construction/membrane-cofactor-cpr   && latexmk -C membrane-cpr.tex
	cd $(P)/synthesis/seven-state-closed-orbit   && latexmk -C closed-orbit.tex
	cd $(P)/spectroscopy/spectroscopic-atlas     && latexmk -C spectroscopic-atlas.tex
	cd $(P)/informatics/database-recovery        && latexmk -C database-recovery.tex
	cd $(P)/diversity/polymorphisms-ddi-inhibitors \
	                                             && latexmk -C polymorphisms-ddi-inhibitors.tex
