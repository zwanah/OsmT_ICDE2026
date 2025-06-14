# OsmT_ICDE2026

## Demo Instructions

This repository contains demonstration code for the OsmT, providing two core functionalities:
1. Obtaining geographic data from OpenStreetMap
2. Visualizing geographic data on interactive maps

## File Structure

```
├── geojson_get.ipynb              # Get OSM data
├── folium_plot.ipynb              # Visualize maps
├── text2ovq_case_study.geojson    # Sample data file
├── text2ovq_case_study.html       # Demo visualization (ready to view)
└── README.md                      # Usage instructions
```

## Quick Demo (No Setup Required)

**For reviewers who want to see the results immediately:**

Simply open `text2ovq_case_study.html` in any web browser to view the interactive map demonstration. This file contains a pre-generated visualization of 17 geographic features (public squares) and requires no installation or setup.

## Installation Requirements

```bash
pip install folium requests geopandas json
```

## Usage Instructions

### 1. Data Acquisition (`geojson_get.ipynb`)

This notebook queries geographic data from OpenStreetMap and saves it as GeoJSON format.

**Quick Start:**
1. Open `geojson_get.ipynb`
2. Run all cells to load functions
3. In the last cell, run `custom_overpass_query()`
4. Choose to use default query (public squares) or input custom query
5. Specify output filename
6. Wait for data download to complete

**Predefined Query Templates:**
- `squares`: Public squares
- `restaurants`: Restaurants
- `parks`: Parks  
- `schools`: Schools

**Usage Examples:**
```python
# Use predefined template
query_overpass_api(query_templates['squares'], 'squares.geojson')

# Interactive query
custom_overpass_query()
```

### 2. Map Visualization (`folium_plot.ipynb`)

This notebook visualizes GeoJSON data as interactive maps.

**Steps:**
1. Open `folium_plot.ipynb`
2. Modify filename in the first cell if needed:
   ```python
   with open('your_file.geojson', 'r', encoding='utf-8') as f:
   ```
3. Run all cells
4. View the generated interactive map
5. Map will be automatically saved as HTML file

**Features:**
- Auto-calculate optimal map view bounds
- Blue markers for all geographic features
- Support for both Point and Polygon geometries
- Generate shareable HTML files

## Complete Workflow

1. **Data Acquisition**: Run `geojson_get.ipynb` → Generate `.geojson` file
2. **Data Visualization**: Load the generated file in `folium_plot.ipynb` → Generate `.html` map
3. **View Results**: Open HTML file in browser to view interactive map

## Sample Data

The repository includes `text2ovq_case_study.geojson` as demonstration data, containing 17 geographic features. You can directly visualize this file using `folium_plot.ipynb`, or simply open the pre-generated `text2ovq_case_study.html` for immediate viewing.

## Output Formats

- **GeoJSON Files**: Standard geographic data format containing coordinates and attribute information
- **HTML Maps**: Interactive maps that can be opened in browsers, supporting zoom and click interactions

## Notes

- Overpass API queries may take considerable time, please be patient
- Large-scale queries may return extensive data, consider limiting query scope appropriately
- Generated HTML files can be directly shared or used for demonstrations
- The demo HTML file (`text2ovq_case_study.html`) can be opened directly without any setup
