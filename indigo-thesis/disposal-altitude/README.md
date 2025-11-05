# Disposal Altitude Analysis

This folder contains analysis of satellite disposal altitudes for S and Su satellite types based on varying disposal lifetimes.

## Files

### Data Files
- `disposal_lookup_S.mat` - MATLAB data file containing disposal lifetime lookup tables for S satellites
- `disposal_lookup_Su.mat` - MATLAB data file containing disposal lifetime lookup tables for Su satellites

### Analysis Scripts
- `detailed_contour_plots.py` - Creates detailed contour plots showing perigee vs apogee relationships
- `fixed_disposal_analysis.py` - Fixed analysis using interpolation instead of log-quadratic fitting
- `debug_get_disposal_orbits.py` - Debug script that identified the issue with the original function
- `debug_disposal_data.py` - Initial debug script for exploring the data structure

### Generated Plots
- `detailed_contour_plots.png` - Side-by-side contour plots for S and Su satellites
- `S_satellites_contour_detailed.png` - Individual detailed contour plot for S satellites
- `Su_satellites_contour_detailed.png` - Individual detailed contour plot for Su satellites
- `satellites_comparison_heatmap.png` - Heatmap comparison between S and Su satellites
- `lifetime_vs_altitude_analysis.png` - Analysis showing lifetime vs altitude relationships
- `disposal_altitude_analysis_fixed.png` - Fixed line plots showing perigee height vs lifetime
- `disposal_altitude_heatmap_fixed.png` - Fixed heatmap showing perigee heights for different combinations

## Key Findings

### Issue with Original Function
The original `get_disposal_orbits` function in `PostMissionDisposal.py` was returning NaN values because:

1. The function expected `coef_logquad` and `R2_log` arrays to contain valid log-quadratic fit coefficients
2. However, these arrays in the .mat files contained all NaN values, indicating the log-quadratic fitting step was not completed in the MATLAB script
3. The function relied on these coefficients to calculate perigee heights for given lifetimes

### Solution
Created a fixed version that:
1. Uses direct interpolation on the `lifetimes_years` data instead of log-quadratic fitting
2. Interpolates between available perigee altitudes to find the required perigee for a given disposal lifetime
3. Handles edge cases where the target lifetime is outside the available data range

### Results
The fixed function now correctly calculates perigee heights for disposal lifetimes:

**S Satellites:**
- For 1000km apogee: 1yr → 461.6km perigee, 5yr → 533.0km perigee, 25yr → 600.0km perigee
- For 1500km apogee: 1yr → 425.5km perigee, 5yr → 513.6km perigee, 25yr → 600.0km perigee

**Su Satellites:**
- For 1000km apogee: 1yr → 372.9km perigee, 5yr → 433.7km perigee, 25yr → 535.7km perigee
- For 1500km apogee: 1yr → 344.0km perigee, 5yr → 419.4km perigee, 25yr → 511.7km perigee

## Usage

To run the analysis:

```bash
cd /Users/indigobrownhall/Code/OPUS/indigo-thesis/disposal-altitude
python fixed_disposal_analysis.py
```

To create detailed contour plots:

```bash
python detailed_contour_plots.py
```

## Data Structure

The .mat files contain:
- `years`: Array of years (2024-2050)
- `apogee_alts_km`: Array of apogee altitudes (400-1500 km)
- `perigee_alts_km`: Array of perigee altitudes (300-600 km)
- `lifetimes_years`: 3D array (years × apogee × perigee) containing disposal lifetimes
- `coef_logquad`: 3D array for log-quadratic coefficients (all NaN in current data)
- `R2_log`: 2D array for R² values (all NaN in current data)
- `circ_lifetime_years`: 2D array for circular orbit lifetimes
- `decay_alt_km`: Decay altitude threshold (150 km)

## Notes

- The analysis uses year 2024 data
- Perigee heights are calculated for disposal lifetimes from 1 to 25 years
- The contour plots show the relationship between apogee height, perigee height, and disposal lifetime
- Su satellites generally require lower perigee heights for the same disposal lifetime compared to S satellites

