# GWAS Inventory Summary

## Overview

- Files indexed: 159
- Total size: 113.88 GB
- Readable files: 159/159

## File extensions

| Extension | Count |
|---|---:|
| `.DHN.gz` | 78 |
| `.TBL` | 39 |
| `.TBL.gz` | 39 |
| `.txt` | 3 |

## Detected GWAS columns

| Semantic column | Files detected |
|---|---:|
| `col_variant_id` | 78 |
| `col_chromosome` | 117 |
| `col_position` | 117 |
| `col_p_value` | 78 |
| `col_beta` | 156 |
| `col_standard_error` | 156 |
| `col_sample_size` | 0 |
| `col_frequency` | 0 |

## Notes

- `.DHN.gz` files appear to contain more standardized GWAS-style columns such as `CHR`, `POS`, `P`, `BETA`, and `SE`.
- `.TBL` and `.TBL.gz` files contain related GWAS result columns such as `MarkerName`, `Effect`, and `StdErr`, but their p-value and genomic position columns may require additional column-name normalization.
- The files are large, with many containing tens of millions of rows. Therefore, the first tabular component focuses on metadata extraction and inventory search rather than loading full tables into memory.
