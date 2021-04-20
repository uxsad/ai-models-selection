========
classify
========

----------------------------------------------------
convert continous emotion values to discrete classes
----------------------------------------------------

:Manual section: 1
:Manual group: UX-SAD
:Author: Andrea Esposito <a.esposito39@studenti.uniba.it>
:Version: 1.0.0

Synopsis
========

**classify** <\ *INPUT_FILE* >\ *OUTPUT_FILE*

Description
===========

This tool classifies reads a UX-SAD dataset from standard input and, for each line, converts the emotion values into discrete classes.

The tool manages 3 classes, converted to numeric values to ease the training process and to show their natural ordering relation: LOW (0), MEDIUM (1) and HIGH (2).

A new dataset, having as emotion values the number associated with the class, is printed to standard output.

Class Selection
---------------

The conversion is done using a simple math condition:

- If the emotion takes values in the range [0, 100]

  - The class LOW is returned if the emotion value is in the range [0, 33.33)
  - The class MEDIUM is returned if the emotion is in the range [33.33, 66.66)
  - The class HIGH is returned if the emotion is in the range [66.66, 100].

- If the emotion takes values in the range [-100, 100]

  - The class LOW is returned if the emotion value is in the range [-100, -33.33)
  - The class MEDIUM is returned if the emotion is in the range [-33.33, 33.33)
  - The class HIGH is returned if the emotion is in the range [33.33, 100].

Known Limitations
=================

- The tool detects the columns to convert by their position (only from the 29th to the 38th column are considered). The columns should, instead, be detected by their name.
- The tool only supports the conversion using three classes. An option to select the number of classes should be provided.
- The tool can only read from standard input. An optional argument to read from a file should be provided (to allow usage on systems without I/O redirection).
- The tool can only write to standard output. An optional argument to write to a file should be provided (to allow usage on systems without I/O redirection).

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
