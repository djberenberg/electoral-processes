HOMEWORK_1: $(wildcard *.tex)
	mkdir -p out
	pdflatex --output-directory=out report_showerter-dberenberg-revtex4.tex
	cp report_showerter-dberenberg.bib out
	cd out ; bibtex report_showerter-dberenberg-revtex4
	cd ../
	pdflatex --output-directory=out report_showerter-dberenberg-revtex4.tex
	pdflatex --output-directory=out report_showerter-dberenberg-revtex4.tex
	cp out/report_showerter-dberenberg-revtex4.pdf ../report_showerter-dberenberg.pdf
	open out/report_showerter-dberenberg-revtex4.pdf

.PHONY: clean
clean:
	rm -f out/report_showerter-dberenberg-revtex4.pdf
	rm -rf out/

