[README.md](https://github.com/user-attachments/files/24154732/README.md)
# Crash Data Standardizer

Transform messy police crash report data into clean, standardized categories for Power BI, Tableau & ArcGIS.

## Features

- **MMUCC Standards** - 15 categories based on NHTSA Model Minimum Uniform Crash Criteria
- **Smart Matching** - Regex → Fuzzy with input preprocessing (handles typos, punctuation, suffixes)
- **Power BI Ready** - Exports fact tables + dimension tables
- **Confidence Scores** - Flag uncertain matches for human review

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

## Usage

1. **Upload** - CSV or Excel crash data
2. **Map Columns** - Select which MMUCC category each column should match
3. **Review** - Check flagged items, edit as needed
4. **Export** - Download Excel (with dimension tables) or CSV

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `mmucc_loader.py` | Dictionary loader |
| `mmucc_dictionaries.json` | MMUCC categories, codes, and synonyms |
| `matching_engine.py` | Regex/fuzzy matching with preprocessing |

## MMUCC Categories

- Manner of Collision
- Injury Severity (KABCO)
- Weather Condition
- Light Condition
- Road Surface Condition
- First Harmful Event
- Contributing Factors (Driver)
- Contributing Factors (Environment/Road)
- Distracted By
- Condition at Time
- Junction Type
- Vehicle Body Type
- Trafficway Type
- Traffic Control Device
- Pre-Crash Maneuver

## Adding Synonyms

Edit `mmucc_dictionaries.json` to add new synonyms:

```json
"manner_of_collision": {
  "synonyms": {
    "3": ["rear end", "rear-end", "YOUR NEW SYNONYM HERE"],
    ...
  }
}
```

## Output Format

### Excel Export
- **Crash_Data** sheet: Original data + standardized columns
- **Dim_*** sheets: Lookup tables for each category (for Power BI relationships)
- **Needs_Review** sheet: Flagged items requiring human review

### Column Ordering
Standardized columns appear immediately after their source column for easy review:

`Weather | Weather_Code | Weather_Standardized | Weather_Confidence | Light | Light_Code | ...`

### Column Naming
- `{Column}_Code` - Numeric MMUCC code
- `{Column}_Standardized` - Standard label
- `{Column}_Confidence` - Match confidence (0-100)

## Requirements

- Python 3.10+
- ~50MB disk space

## License

Free to use and adapt for internal business, personal, or educational use. Please don’t sell it or turn it into a paid product.

Licensed under Creative Commons Attribution–NonCommercial 4.0 (CC BY-NC 4.0)

## Feedback

Found a bug or have a suggestion? Send feedback to contact@alexengineered.com
