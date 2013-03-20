all: voxel_likelihood.pdf

voxels.png: voxels.py
	python voxels.py

voxel_likelihood.pdf: voxels.png

%.pdf: %.tex
	pdflatex $<
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
