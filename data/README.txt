Hindi Visual Genome 1.0 Release Description
-------------------------------------------
http://hdl.handle.net/11234/1-2997
Shantipriya Parida, Ondrej Bojar
Charles University, Faculty of Mathematics and Physics,
Institute of Formal and Applied Linguistics (UFAL)
2019

Data
----
Hindi Visual Genome 1.0, a multimodal dataset consisting of text and images suitable for English-to-Hindi multimodal machine translation task and multimodal research. We have selected short English segments (captions) from Visual Genome along with associated images and automatically translated them to Hindi with manual post-editing, taking the associated images into account. The training set contains 29K segments. Further 1K and 1.6K segments are provided in a development and test sets, respectively, which follow the same (random) sampling from the original Hindi Visual Genome.

Additionally, a challenge test set of 1400 segments will be released for the WAT2019 multi-modal task. This challenge test set was created by searching for (particularly) ambiguous English words based on the embedding similarity and manually selecting those where the image helps to resolve the ambiguity.

Dataset Formats
--------------
The multimodal dataset contains both text and images.

The text parts of the dataset (train and test sets) are in simple tab-delimited plain text files.

All the text files have seven columns as follows:

Column1	- image_id
Column2	- X
Column3 - Y
Column4 - Width
Column5 - Height
Column6 - English Text
Column7 - Hindi Text

The image part contains the full images with the corresponding image_id as the file name. The X, Y, Width and Height columns indicate the rectangular region in the image described by the caption.

Data Statistics
----------------
The statistics of the current release is given below.

Parallel Corpus Statistics
---------------------------

Dataset       	Segments 	English Words   	Hindi Words
-------       	---------	----------------	-------------
Train         	    28932	          143178	       136722
Dev           	      998	            4922	         4695
Test          	     1595	            7852	         7535
Challenge Test	     1400	            8185	         8665	(Released separately)
-------       	---------	----------------	-------------
Total         	    32925	          164137	       157617

The word counts are approximate, prior to tokenization.

Citation
--------

If you use this corpus, please cite the following paper:

@article{hindi-visual-genome:2019,
  title={{Hindi Visual Genome: A Dataset for Multimodal English-to-Hindi Machine Translation}},
  author={Parida, Shantipriya and Bojar, Ond{\v{r}}ej and Dash, Satya Ranjan},
  journal={Computaci{\'o}n y Sistemas},
  note={In print. Presented at CICLing 2019, La Rochelle, France},
  year={2019},
}

