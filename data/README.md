## Data
- `gold` directory contains gold-standard taxonomies (used for evaluation) for each terminology in the `terms` directory.

- Files:
  - equipment: terms from a manual taxonomy rooted on "equipment"
  - environment: terms from the EuroVoc taxonomy rooted on "environment"
  - food: terms from a manual taxonomy rooted on "food"
  - science: terms from a manual taxonomy rooted on "science"
  - science_ev: terms from the EuroVoc taxonomy rooted on "science"
  - science_wn: terms from the WordNet taxonomy rooted on "science"

- File format:
  - term_id <TAB> term
  - *term_id* is a term identifier
  - *term* is a domain term.

- Sources:
  - [TExEval-1](https://alt.qcri.org/semeval2015/task17/index.php?id=data-and-tools)
  - [TExEval-2](https://alt.qcri.org/semeval2016/task13/index.php?id=data-and-tools)