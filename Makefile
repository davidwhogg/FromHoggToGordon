all: voxel_likelihood.pdf

voxels.png: voxels.py
	python voxels.py

voxel_likelihood.pdf: voxels.eps

%.pdf: %.tex
	latex $<
	bash -c " ( grep Rerun $*.log && latex $< ) || echo noRerun "
	bash -c " ( grep Rerun $*.log && latex $< ) || echo noRerun "
	dvipdf $*