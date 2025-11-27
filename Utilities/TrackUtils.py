import fastf1
import fastf1.plotting
import fastf1.utils
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import json
import os

try:
    fastf1.plotting.setup_mpl()
except Exception:
    pass

class TrackAnalyzer:
    """
    Spatial analysis of the track (Dominance maps, Delta maps, Telemetry maps).
    
    Usage:
        Track = TrackAnalyzer(session)
        Track.gain.map(ref_driver='VER', comp_driver='NOR')
    """
    def __init__(self, session, team_colors=None):
        self.session = session
        
        if team_colors is None:
            self.team_colors = self._load_default_colors()
        else:
            self.team_colors = team_colors
            
        self.laps = session.laps.pick_quicklaps().pick_wo_box()
        
        # Initialize sub-modules
        self.gain = GainPhase(self)

    def _load_default_colors(self):
        try:
            current_dir = os.path.dirname(__file__)
            path = os.path.join(current_dir, 'team_colors.json')
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load team_colors.json. {e}")
            return {}

    def _get_save_name(self, suffix):
        event_name = self.session.event.EventName.replace(" ", "")
        year = self.session.event.year
        session_type_map = {
            'Practice 1': 'FP1', 'Practice 2': 'FP2', 'Practice 3': 'FP3',
            'Qualifying': 'Q', 'Sprint Qualifying': 'SQ', 'Sprint Shootout': 'SQ',
            'Sprint': 'S', 'Race': 'R'
        }
        session_id = session_type_map.get(self.session.name, self.session.name.replace(" ", ""))
        return f"{event_name}{year}_{session_id}_{suffix}.png"

class GainPhase:
    def __init__(self, parent):
        self.parent = parent

    def map(self, ref_driver, comp_driver):
        """
        Plots the track map colored by time delta.
        Purple/Green = Reference driver is FASTER (Gaining).
        Yellow/Red   = Reference driver is SLOWER (Losing).
        """
        print(f"Generating Gain Map: {ref_driver} vs {comp_driver}...")
        
        # 1. Get Fastest Laps
        lap_ref = self.parent.laps.pick_drivers(ref_driver).pick_fastest()
        lap_comp = self.parent.laps.pick_drivers(comp_driver).pick_fastest()
        
        if lap_ref is None or lap_comp is None:
            print(f"Could not find laps for {ref_driver} or {comp_driver}")
            return

        # 2. Calculate Delta
        # delta > 0: Ref is FASTER (Comp is behind)
        # delta < 0: Ref is SLOWER (Comp is ahead)
        delta_time, ref_tel, comp_tel = fastf1.utils.delta_time(lap_ref, lap_comp)
        
        # 3. Prepare Data for Plotting
        # We need X, Y coordinates from the reference telemetry
        x = ref_tel['X'].values
        y = ref_tel['Y'].values
        
        # Create segments (lines connecting point i to i+1)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 4. Setup Color Map
        # We want a diverging map. 
        # Positive (Green/Purple) = Good for Ref
        # Negative (Red/Yellow) = Bad for Ref
        # Let's use 'PRGn' (Purple-Green) or 'coolwarm'
        # Or custom: Red (Loss) -> White (Neutral) -> Green (Gain)
        cmap = plt.get_cmap('RdYlGn') 
        
        # Normalize the delta to center around 0
        # We clamp extreme outliers to keeping the color scale useful
        max_delta = max(abs(delta_time.min()), abs(delta_time.max()))
        norm = plt.Normalize(-max_delta, max_delta)

        # 5. Plot
        plt.figure(figsize=(12, 12))
        
        # Create the LineCollection
        lc = LineCollection(segments, cmap=cmap, norm=norm, linestyle='-', linewidth=5)
        lc.set_array(delta_time)
        
        ax = plt.gca()
        ax.add_collection(lc)
        
        # Auto-scale axes
        ax.set_xlim(x.min() - 200, x.max() + 200)
        ax.set_ylim(y.min() - 200, y.max() + 200)
        ax.set_aspect('equal')
        plt.axis('off') # Hide axes for clean map look
        
        # Add Colorbar
        cbar = plt.colorbar(lc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_label(f"Time Delta (seconds)\nGreen = {ref_driver} Faster | Red = {ref_driver} Slower")
        
        # Add Title
        plt.suptitle(f"{self.parent.session.event.year} {self.parent.session.event.EventName} - Gain Map", fontsize=16, y=0.95)
        plt.title(f"Reference: {ref_driver} | Comparison: {comp_driver}", fontsize=12)
        
        # Save
        filename = self.parent._get_save_name(f"Track_GainMap_{ref_driver}_vs_{comp_driver}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.show()