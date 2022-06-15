# WARNING: BIBTEX TURNED OFF FOR NOW

LATEX       = pdflatex -interaction nonstopmode -halt-on-error -file-line-error
CHECK_RERUN = grep "Rerun to get"
CRUFT_SUFFS = pdf aux bbl blg log dvi ps eps out brf fls fdb_latexmk synctex.gz bcf run.xml
NAME        = yellowCard_draft

all: ${NAME}.pdf

${NAME}.pdf: ${NAME}.tex refs.bib
	${LATEX} ${NAME}.tex
	bibtex ${NAME}
	${LATEX} ${NAME}.tex
	${LATEX} ${NAME}.tex
	( ${CHECK_RERUN} ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."
	( ${CHECK_RERUN} ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."

clean:
	${RM} $(foreach suff, ${CRUFT_SUFFS}, ${NAME}.${suff})
	${RM} *Notes.bib
