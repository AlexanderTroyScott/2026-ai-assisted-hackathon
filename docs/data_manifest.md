# Data Manifest

Parquet files as received from Kaggle (`public/` prefix in the competition).
The parquets themselves are **not committed** — `.gitignore` excludes `developed/`.
This manifest lets the dataset be re-verified against the Kaggle download
(`kaggle competitions files -c iaa-ai-assisted-hackathon-2026-part-1`).

| file | rows | bytes | sha256 |
|---|---|---|---|
| addresses.parquet | 575,983 | 8,209,304 | `2f1a9a157649a73e` |
| claims.parquet | 163,150 | 6,393,494 | `323adf2e0f6c5975` |
| customers.parquet | 545,240 | 9,840,123 | `065458a0b3aa9d34` |
| policies_buildings.parquet | 707,726 | 28,411,001 | `eb84372c23c16ed0` |
| policies_contents.parquet | 721,617 | 28,203,870 | `8c067cc80baed583` |
| policies_motor.parquet | 979,968 | 44,218,528 | `4d2d7ae03971d5d6` |
| sample_submission.parquet | 626,639 | 3,271,617 | `9529f3b6a679d341` |
| test_buildings.parquet | 184,155 | 7,917,305 | `7b2b96507de29241` |
| test_contents.parquet | 187,498 | 7,795,903 | `c2212f62f3b7a214` |
| test_motor.parquet | 254,986 | 12,035,693 | `8d28fb9fd091caf2` |
| weather.parquet | 2,879,915 | 41,863,008 | `b8407f0f3c7279a4` |
