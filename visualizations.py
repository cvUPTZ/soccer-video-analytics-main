import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting (important for scripts)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

def generate_player_heatmap(
    positions_history: List[Dict[str, Any]],
    output_path: str,
    filter_team_name: Optional[str] = None,
    filter_player_id: Optional[int] = None,
    img_size: Optional[Tuple[int, int]] = None, # e.g., (1280, 720) video dimensions
    pitch_background_path: Optional[str] = None, # Optional path to a pitch image
    cmap: str = "viridis", # Colormap for the heatmap
    alpha: float = 0.7, # Transparency of the heatmap layer
    levels: int = 10 # Number of contour levels for kdeplot
):
    """
    Generates a player position heatmap and saves it to a file.

    Args:
        positions_history: List of position data dictionaries.
                           Each dict expects 'position_abs_center': (x,y),
                           'team_name': str, 'player_id': int.
        output_path: Path to save the generated heatmap image.
        filter_team_name: Optional team name to filter positions for.
        filter_player_id: Optional player ID to filter positions for.
        img_size: Optional tuple (width, height) to define heatmap canvas size.
                  If None, determined by data range.
        pitch_background_path: Optional path to an image to use as background.
        cmap: Matplotlib colormap name.
        alpha: Transparency for the heatmap overlay.
        levels: Number of contour levels for seaborn's kdeplot.
    """
    if not positions_history:
        print("Warning: No position history provided to generate heatmap.")
        return

    # Filter positions
    filtered_positions = []
    for entry in positions_history:
        if filter_team_name and entry.get('team_name') != filter_team_name:
            continue
        if filter_player_id and entry.get('player_id') != filter_player_id:
            continue
        if entry.get('position_abs_center'):
            filtered_positions.append(entry['position_abs_center'])

    if not filtered_positions:
        print(f"Warning: No positions found for filter (Team: {filter_team_name}, PlayerID: {filter_player_id}). Heatmap not generated.")
        return

    x_coords = [pos[0] for pos in filtered_positions]
    y_coords = [pos[1] for pos in filtered_positions]

    if not x_coords or not y_coords or len(x_coords) < 2: # KDE plot needs at least 2 points
        print("Warning: Not enough data points (need at least 2) after filtering to create a heatmap.")
        return

    plt.figure(figsize=(12.8, 7.2) if img_size is None else (img_size[0]/100.0, img_size[1]/100.0)) # Adjust figsize

    ax = plt.gca()
    # Do not set aspect equal here if using imshow with extent that might not be equal, handle aspect with imshow
    # ax.set_aspect('equal')

    # Attempt to load and display pitch background if provided
    if pitch_background_path:
        try:
            pitch_img = plt.imread(pitch_background_path)
            # If img_size is provided, use it for extent, otherwise, data range.
            if img_size:
                plt.imshow(pitch_img, extent=[0, img_size[0], img_size[1], 0], aspect='auto') # Invert y-axis for image
            else:
                # Auto-scale based on data - this might not align well if pitch_background has fixed aspect
                # Determine extent from data, but this is tricky if pitch_img aspect is fixed.
                # For now, let's assume if no img_size, imshow will auto-adjust.
                plt.imshow(pitch_img, aspect='auto')
        except FileNotFoundError:
            print(f"Warning: Pitch background image not found at {pitch_background_path}. Drawing on blank canvas.")
        except Exception as e:
            print(f"Warning: Could not load pitch background {pitch_background_path}: {e}. Drawing on blank canvas.")


    # Create heatmap using seaborn's kdeplot for a smooth density map
    sns.kdeplot(
        x=x_coords, y=y_coords,
        cmap=cmap,
        fill=True,
        alpha=alpha,
        levels=levels, # More levels for smoother gradient
        thresh=0.05, # Threshold for KDE, adjust as needed
        # bw_adjust=0.5 # Adjust bandwidth for smoothing
        ax=ax
    )

    if img_size:
        plt.xlim(0, img_size[0])
        plt.ylim(img_size[1], 0) # Invert y-axis to match typical image coordinates (0,0 at top-left)
    else:
        # Auto-scale, but ensure y-axis is inverted for consistency if desired
        plt.gca().invert_yaxis()


    title_str = "Player Heatmap"
    if filter_team_name:
        title_str += f" (Team: {filter_team_name}"
        if filter_player_id:
            title_str += f", Player ID: {filter_player_id}"
        title_str += ")"
    elif filter_player_id:
        title_str += f" (Player ID: {filter_player_id})"
    else:
        title_str += " (All Players)"

    plt.title(title_str)
    # plt.xlabel("X Coordinate (Absolute)") # Turned off by axis('off')
    # plt.ylabel("Y Coordinate (Absolute)") # Turned off by axis('off')
    plt.grid(False) # Turn off grid for cleaner heatmap image
    plt.axis('off') # Turn off axis numbers and ticks

    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Heatmap saved to {output_path}")
    except Exception as e:
        print(f"Error saving heatmap to {output_path}: {e}")
    finally:
        plt.close() # Close the figure to free memory

if __name__ == '__main__':
    # Example Usage (for testing visualizations.py directly)
    print("Running example heatmap generation...")
    example_history = [
        {'frame': 1, 'player_id': 1, 'team_name': "Chelsea", 'position_abs_center': (100, 200)},
        {'frame': 2, 'player_id': 1, 'team_name': "Chelsea", 'position_abs_center': (110, 210)},
        {'frame': 1, 'player_id': 2, 'team_name': "Man City", 'position_abs_center': (500, 300)},
        {'frame': 2, 'player_id': 2, 'team_name': "Man City", 'position_abs_center': (510, 310)},
        # Add more data points for a meaningful heatmap
        {'frame': 3, 'player_id': 1, 'team_name': "Chelsea", 'position_abs_center': (105, 205)},
        {'frame': 4, 'player_id': 1, 'team_name': "Chelsea", 'position_abs_center': (115, 215)},
        {'frame': 5, 'player_id': 1, 'team_name': "Chelsea", 'position_abs_center': (120, 220)},
        {'frame': 3, 'player_id': 2, 'team_name': "Man City", 'position_abs_center': (505, 305)},
        {'frame': 4, 'player_id': 2, 'team_name': "Man City", 'position_abs_center': (515, 315)},
        {'frame': 5, 'player_id': 2, 'team_name': "Man City", 'position_abs_center': (520, 320)},
    ]
    # Simulate video dimensions for canvas scaling
    video_width = 1920
    video_height = 1080

    # Add more points for player 1 to make a denser area
    for i in range(100):
        example_history.append({'frame': i+6, 'player_id': 1, 'team_name': "Chelsea",
                                'position_abs_center': (np.random.normal(400, 100), np.random.normal(500, 100))}) # Adjusted mean for visibility
    for i in range(80):
        example_history.append({'frame': i+6, 'player_id': 2, 'team_name': "Man City",
                                'position_abs_center': (np.random.normal(1200, 120), np.random.normal(600, 80))}) # Adjusted mean for visibility


    generate_player_heatmap(example_history, "heatmap_all_players.png", img_size=(video_width, video_height))
    generate_player_heatmap(example_history, "heatmap_chelsea.png", filter_team_name="Chelsea", img_size=(video_width, video_height))
    generate_player_heatmap(example_history, "heatmap_player1.png", filter_player_id=1, img_size=(video_width, video_height))

    # Test case with no matching player ID
    generate_player_heatmap(example_history, "heatmap_player_none.png", filter_player_id=999, img_size=(video_width, video_height))
    # Test case with empty history
    generate_player_heatmap([], "heatmap_empty_history.png", img_size=(video_width, video_height))

    print("Example heatmap generation complete. Check for .png files.")
