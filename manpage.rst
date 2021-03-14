=========
ai-models
=========

-----------------------------------
train and test ai models for UX-SAD
-----------------------------------

:Manual section: 1
:Manual group: UX-SAD
:Author: Andrea Esposito <a.esposito39@studenti.uniba.it>
:Version: 1.0.0

Synopsis
========

**ai-models**
[**-r** *RANDOM*]
[**-o** *OUT*]
[**-V**]
[**-P**]
[**-j** *JOBS*]
[**-m** *MODEL*]
[**-s** *STRATEGY*]...
*DATASET*
*EMOTION*
*WIDTH*
*HALF*

**ai-models** [**-h**\|\ **-v**]

Description
===========

Train and test various AI models using multiple algorithms. This can be used
either to train a specific model using a specific algorithm on a specific
dataset or to compare the performances of multiple models.

The training uses the dataset provided in the folder *DATASET* (see the section
`The Dataset`_ for more information on the dataset format) and will train a
model to predict the emotion *EMOTION* using the half *HALF* of windows of
*WIDTH* milliseconds.

The accepted value for the positional arguments are the following:

*DATASET*
	Any path to a folder. The content of the folder must follow the
	specification provided in the section `The Dataset`_.
*EMOTION*
	The available emotions depend on the provided dataset. The standard
	ones are the "universal" emotions identified by P. Ekman plus two
	additional values, ie. *EMOTION* must be one in {**anger**\|\
	**contempt**\|\ **disgust**\|\ **engagement**\|\ **fear**\|\ **joy**\|\
	**sadness**\|\ **surprise**\|\ **users**\|\ **valence**\|\
	**websites**}
*WIDTH*
	The width of the window frames to be considered. The available values
	depend on the actual dataset.
*HALF*
	The window half to be considered. Must be one of {**before**\|\
	**after**\|\ **full**}

The available options are the following. Mandatory arguments to long options
are mandatory for short options too.

**-h**, **--help** 
        Show this message and exit.
**-v**, **--version**
        Show the version and exit.
**-r**, **--random** *RANDOM*  
        Set the random seed to *RANDOM*
**-o**, **--out** *OUT*  
	Set the path where the model will be saved to *OUT*.  A new file with
	the same name as the model's emotion will be created inside this
	directory.
**-V**, **--verbose**
        If specified, increase the tool verbosity (messages will be printed to stderr).
**-P**, **--progress**
        If specified, show the progress of the tool using a loading bar.
**-j**, **--jobs** *JOBS*
        Set the number of parallel jobs.
**-a**, **--algorithm** {**forest**\|\ **svm**\|\ **adaboost**\|\ **tree**\|\ **perceptron**}
        Set the algorithm on which the model will be based.
**-s**, **--strategy** [**sfs**\|\ **pca**\|\ **efs**]
        The strategy to apply in order to reduce the number of final features.

The Dataset
-----------

The dataset specified by the *DATASET* argument must follow a specific
structure.  This is automatically done by the provided tools (see
``dataset.py``), but if manual processing is done be sure to adhere the this
assumptions before running the tool (otherwise, the outcome is unpredicted).

The folder *DATASET* must contain multiple CSV files: one for each emotion. The
names of this files must be exactly "*emotion*\ **.csv**", where *emotion* is
the name of the emotion in lower case. Each CSV file must contain an header
row. Each CSV must contain the keys **middle.emotions.**\ *emotion*, as well as
each of the other emotions (that will be discarded by the tool once the dataset
has been loaded).

Copyright
=========

| Copyright (C) 2020 Andrea Esposito.
| License GPLv3+: GNU GPL version 3 or Later <https://gnu.org/licenses/gpl.html>.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
