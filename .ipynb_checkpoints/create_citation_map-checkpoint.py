import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import numpy as np
import os

try:
    from adjustText import adjust_text
except ImportError:
    print("Info: 'adjustText' library not found. "
          "To use 'adjust_labels=True', please install it: pip install adjusttext")
    def adjust_text(texts, **kwargs):
        print("Warning: 'adjustText' not installed. Skipping label adjustment.")
        pass

def create_citation_map(
    csv_filepath: str,
    output_filename: str = 'citation_map.png',
    # --- Data Scaling ---
    scale: str = 'linear', # 'linear', 'log', 'rank', 'log_rank'
    
    # --- Country Fill Style ---
    fill_mode: str = 'heatmap', # 'heatmap', 'alpha', 'simple'
    fill_color: str = '#E63946', # Base color for 'simple' & 'alpha'
    fill_alpha: float = 1.0, # Alpha for 'simple' mode. Default is 1.0
    fill_cmap: str = 'YlOrRd', # Colormap for 'heatmap'
    
    # --- Pin Style ---
    show_pins: bool = False,
    pin_color: str = '#E63946', # Base color if pin_scale_color is False
    pin_cmap: str = 'viridis', # Colormap if pin_scale_color is True
    pin_scale_color: bool = False, # Vary pin color with value?
    pin_scale_size: bool = True,  # Vary pin size with value?
    pin_scale_alpha: bool = True, # Vary pin alpha with value?
    pin_size_range: tuple = (20, 200), # (min, max) for scaled pins
    pin_size_static: int = 50,  # Size for static pins
    
    # --- Other Options ---
    show_labels: bool = False,
    show_counts: bool = False,
    adjust_labels: bool = False,
    label_top_n: int = None,
    show_legend: bool = False, # Show simple categorical legend
    base_color: str = '#EEEEEE',
    border_color: str = '#FFFFFF'
):
    """
    Generates a static map of citing countries based on a modular design.
    """
    
    # --- 0. Input Validation ---
    if fill_mode not in ['heatmap', 'alpha', 'simple']:
        print(f"Warning: Invalid fill_mode '{fill_mode}'. Defaulting to 'heatmap'.")
        fill_mode = 'heatmap'
    if scale not in ['linear', 'log', 'rank', 'log_rank']:
        print(f"Warning: Invalid scale '{scale}'. Defaulting to 'linear'.")
        scale = 'linear'
    
    file_extension = os.path.splitext(output_filename)[1].lower()
    if file_extension not in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
        print(f"Warning: Output file '{output_filename}' is not a recognized image format.")
        print("Defaulting to 'citation_map.png'")
        output_filename = 'citation_map.png'
    
    # --- 1. Load Citation Data ---
    try:
        df = pd.read_csv(csv_filepath)
        if 'cited_by_country' not in df.columns:
            print(f"Error: CSV file must contain 'cited_by_country' column.")
            return
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_filepath}'")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- 2. Load World Map ---
    try:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = geopandas.read_file(url)
    except Exception as e:
        print(f"Error loading world map dataset: {e}")
        return

    # Robustness: Convert column names to lowercase
    world.columns = world.columns.str.lower()
    world = world[world.name != "Antarctica"] # Filter out Antarctica

    # --- 3. Process Data and Merge ---
    citation_counts = df['cited_by_country'].value_counts()
    world = world.merge(
        citation_counts.rename('count'), 
        left_on='iso_a2', 
        right_index=True, 
        how='left'
    )
    world['count'] = world['count'].fillna(0).astype(int)
    
    # --- 4. Global Scaling (The "Master" Value) ---
    if scale == 'linear':
        world['scaled_value'] = world['count']
    elif scale == 'log':
        world['scaled_value'] = np.log1p(world['count'])
    elif scale == 'rank':
        ranks = world['count'].rank(method='dense')
        min_real_rank = ranks[world['count'] > 0].min() if ranks[world['count'] > 0].any() else 0
        world['scaled_value'] = ranks - min_real_rank + 1
        world.loc[world['count'] == 0, 'scaled_value'] = 0
    elif scale == 'log_rank':
        ranks = world['count'].rank(method='dense')
        min_real_rank = ranks[world['count'] > 0].min() if ranks[world['count'] > 0].any() else 0
        world['scaled_value'] = ranks - min_real_rank + 1
        world.loc[world['count'] == 0, 'scaled_value'] = 0
        world['scaled_value'] = np.log1p(world['scaled_value'])

    # --- 5. Normalization (0-1) for scaling Alpha, Size, Color ---
    world['normalized_value'] = 0.0
    cited_geometries_df = world[world['count'] > 0].copy() 
    
    if not cited_geometries_df.empty:
        min_val = cited_geometries_df['scaled_value'].min()
        max_val = cited_geometries_df['scaled_value'].max()
        range_val = max_val - min_val
        
        if range_val == 0:
            world.loc[world['count'] > 0, 'normalized_value'] = 1.0 # All have same count
        else:
            world.loc[world['count'] > 0, 'normalized_value'] = (world['scaled_value'] - min_val) / range_val

    # Refresh cited_geometries with new columns
    cited_geometries = world[world['count'] > 0]

    # --- 6. Plotting ---
    print(f"Generating citation map ({output_filename})...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    # 6a. Plot base map
    world.plot(
        ax=ax, 
        color=base_color, 
        edgecolor=border_color, 
        linewidth=0.5
    )

    # 6b. Plot data based on fill_mode
    if not cited_geometries.empty:
        if fill_mode == 'simple':
            cited_geometries.plot(
                ax=ax,
                color=fill_color,
                edgecolor=border_color,
                linewidth=0.5,
                alpha=fill_alpha # Use configurable alpha
            )
        
        elif fill_mode == 'alpha':
            for _, row in cited_geometries.iterrows():
                actual_alpha = 0.1 + row['normalized_value'] * 0.8 # Scale 0.1 to 0.9
                geopandas.GeoSeries([row.geometry]).plot(
                    ax=ax,
                    color=fill_color,
                    edgecolor=border_color,
                    linewidth=0.5,
                    alpha=actual_alpha
                )
    
        elif fill_mode == 'heatmap':
            cited_geometries.plot(
                ax=ax,
                column='scaled_value',
                cmap=fill_cmap,
                edgecolor=border_color,
                linewidth=0.5,
                legend=False # No numeric legend, as requested
            )

    # 6c. Add title and (optional) legend
    ax.set_axis_off()
    ax.set_title(
        'Global Distribution of Citations',
        fontdict={'fontsize': '20', 'fontweight': 'bold'}
    )

    # Add categorical legend for simple mode
    if show_legend and fill_mode == 'simple':
        cited_patch = mpatches.Patch(color=fill_color, alpha=fill_alpha, label='Citing Country')
        base_patch = mpatches.Patch(color=base_color, label='Not a Citing Country')
        ax.legend(
            handles=[cited_patch, base_patch],
            loc='lower left',
            bbox_to_anchor=(0.0, 0.0), # Position at bottom-left
            frameon=False, # No border
            # fontsize='small'
        )

    # 6d. Add labels and pins
    if show_labels or show_pins or show_counts:
        if not cited_geometries.empty:
            # Get colormap for pins if needed
            texts_to_adjust = [] 
            if pin_scale_color:
                pin_cmap_obj = plt.get_cmap(pin_cmap)
                
            # Determine which geometries to label (top N filtering)
            if label_top_n is not None and label_top_n < len(cited_geometries):
                min_count_for_top_n = cited_geometries['count'].nlargest(label_top_n).min()
                geometries_to_label = cited_geometries[cited_geometries['count'] >= min_count_for_top_n]
            else:
                geometries_to_label = cited_geometries

            # Sort by normalized_value descending so largest pins are drawn first
            sorted_geometries = cited_geometries.sort_values(by='normalized_value', ascending=False)
            
            for _, row in sorted_geometries.iterrows():
                centroid = row.geometry.centroid
                if centroid.is_empty:
                    continue
                
                # Plot pins first, so labels are on top
                if show_pins:
                    normalized_val = row['normalized_value']

                    # --- Determine Pin Properties based on flags ---
                    current_pin_color = pin_cmap_obj(normalized_val) if pin_scale_color else pin_color
                    
                    if pin_scale_size:
                        min_size, max_size = pin_size_range
                        size_delta = max_size - min_size
                        pin_size = min_size + (normalized_val * size_delta)
                    else:
                        pin_size = pin_size_static # Use static size

                    pin_alpha = (0.3 + normalized_val * 0.5) if pin_scale_alpha else 0.7
                    
                    ax.scatter(
                        x=centroid.x, 
                        y=centroid.y,
                        s=pin_size,
                        alpha=pin_alpha,
                        color=current_pin_color,
                        edgecolors='black',
                        linewidth=0.5,
                        zorder=10 # Draw pins above map but below labels
                    )
            
            # label top N 
            for _, row in geometries_to_label.iterrows():
                centroid = row.geometry.centroid
                if centroid.is_empty:
                    continue
                # --- labels/counts ---
                if show_labels or show_counts:
                    # Determine text to display
                    label_text = ""
                    if show_labels:
                        label_text = row['name']
                    if show_counts:
                        # Add a newline if both are shown
                        if label_text: 
                            label_text += f"\n{row['count']}"
                        else:
                            label_text = str(row['count']) # Use raw count

                    if label_text: # Ensure we have something to plot
                        text_obj = ax.annotate(
                                text=label_text,
                            xy=(centroid.x, centroid.y),
                                ha='center', 
                                va='center', # Center-align multi-line text
                            fontsize=8,
                            color='black',
                            path_effects=[
                                PathEffects.withStroke(linewidth=2, foreground="white")
                            ],
                                zorder=11 # Draw labels on top of pins
                        )
                        texts_to_adjust.append(text_obj)

            if adjust_labels and texts_to_adjust:
                print("Adjusting labels to avoid overlap...")
                adjust_text(
                    texts_to_adjust,
                    ax=ax,
                    # force_text=(0.2, 0.1),
                    # force_pull=(0.5, 0.5),
                    arrowprops=dict(
                        arrowstyle='-', 
                        color='gray', 
                        lw=0.5,
                        alpha=0.7
                    ),
                    lims=(*ax.get_xlim(), *ax.get_ylim()),
                )


    # 6e. Save plot
    plt.tight_layout()
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Success! Citation map saved to: {output_filename}\n")
    except Exception as e:
        print(f"Error saving citation map: {e}\n")
    plt.close(fig) # Close the figure to free up memory

def plot_citations_by_country_bar(
    csv_filepath: str,
    output_filename: str = "citations_by_country_bar.png",
    country_col: str = "cited_by_country",
    top_n: int | None = 30,
    include_unknown: bool = False,
    unknown_values: tuple[str, ...] = ("N/A", "", "None", "nan"),
    figsize: tuple[int, int] = (14, 6),
    dpi: int = 300,
    title: str = "Citations by Country",
    xlabel: str = "Country",
    ylabel: str = "Number of citing papers",
    rotate_xticks: int = 45,
) -> pd.DataFrame:
    """
    Create a bar plot: x = country, y = count of citing papers, sorted high -> low.

    Parameters
    ----------
    csv_filepath : str
        Path to citation_info.csv created by citation_fetcher.py
    output_filename : str
        Output image filename (.png/.jpg/.pdf/.svg supported by matplotlib)
    country_col : str
        Column name for country codes (default: 'cited_by_country')
        Typically ISO-2 codes from OpenAlex, e.g., 'US', 'CN', 'GB'.
    top_n : int | None
        If provided, plot only the top N countries by count.
        If None, plot all countries (can be crowded).
    include_unknown : bool
        Whether to include unknown/blank entries (e.g., 'N/A') as a category.
    unknown_values : tuple[str, ...]
        Values treated as unknown / missing.
    figsize : tuple[int, int]
        Figure size in inches.
    dpi : int
        Save resolution.
    title, xlabel, ylabel : str
        Plot labels.
    rotate_xticks : int
        Rotation angle for country labels.

    Returns
    -------
    counts_df : pandas.DataFrame
        A dataframe with columns: [country_col, 'count'] sorted descending.
        Useful if you want to export or inspect numbers.
    """

    # -----------------------------
    # 1) Load and validate
    # -----------------------------
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"CSV not found: {csv_filepath}")

    df = pd.read_csv(csv_filepath)

    if country_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{country_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    # -----------------------------
    # 2) Clean country codes
    # -----------------------------
    # Convert to string, strip whitespace, standardize case
    countries = (
        df[country_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Convert typical "unknown" tokens to NA
    unknown_set = {v.upper() for v in unknown_values}
    countries = countries.where(~countries.isin(unknown_set), other=pd.NA)

    # Optionally drop unknowns
    if not include_unknown:
        countries = countries.dropna()

    # If everything is missing, stop early
    if countries.empty:
        raise ValueError(
            "No valid country entries found to plot. "
            "Try include_unknown=True or check cited_by_country generation."
        )

    # -----------------------------
    # 3) Count and sort
    # -----------------------------
    counts = countries.value_counts(dropna=False)

    # If you chose to include unknown, label it clearly
    if include_unknown:
        counts.index = counts.index.fillna("UNKNOWN")

    counts_df = (
        counts
        .rename("count")
        .reset_index()
        .rename(columns={"index": country_col})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    # Apply top_n selection
    if top_n is not None:
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive int or None.")
        counts_df = counts_df.head(top_n)

    # -----------------------------
    # 4) Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(counts_df[country_col].astype(str), counts_df["count"].astype(int))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Improve readability for many categories
    plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    # -----------------------------
    # 5) Save
    # -----------------------------
    fig.savefig(output_filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved bar plot to: {output_filename}")
    return counts_df

