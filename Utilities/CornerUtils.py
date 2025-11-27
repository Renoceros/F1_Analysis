import fastf1
import fastf1.plotting
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

# Ensure FastF1 default styling is applied when this module is imported
try:
    fastf1.plotting.setup_mpl()
except Exception:
    pass

class CircuitAnalyzer:
    """
    Main controller for analyzing corner performance.
    
    Usage:
        # Automatically loads colors from team_colors.json in this folder
        Corner = CircuitAnalyzer(session)
        Corner.entry.braking_dist(7)       # Specific corner
        Corner.all.velo_dist()             # Average across ALL corners
    """
    def __init__(self, session, team_colors=None):
        self.session = session
        
        # Load colors if not provided
        if team_colors is None:
            self.team_colors = self._load_default_colors()
        else:
            self.team_colors = team_colors
        
        print("Initializing CircuitAnalyzer: Pre-filtering laps...")
        self.laps = session.laps.pick_quicklaps().pick_wo_box()
        self.circuit_info = session.get_circuit_info()
        
        # Initialize sub-modules
        self.entry = EntryPhase(self)
        self.exit = ExitPhase(self)
        self.all = AllPhase(self)

    def _load_default_colors(self):
        """Loads team_colors.json from the same directory as this script."""
        try:
            current_dir = os.path.dirname(__file__)
            path = os.path.join(current_dir, 'team_colors.json')
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load team_colors.json. {e}")
            return {}

    def _get_corner_distance(self, corner_number):
        try:
            val = self.circuit_info.corners.loc[self.circuit_info.corners['Number'] == corner_number, 'Distance']
            if val.empty:
                raise ValueError(f"Corner {corner_number} not found.")
            return val.values[0]
        except Exception as e:
            raise ValueError(f"Error finding Corner {corner_number}: {e}")

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

    def _plot_distribution(self, df, x_col, y_col, title, ylabel, filename_suffix, higher_is_better=False):
        if df.empty:
            print(f"No data available to plot for {title}")
            return

        medians = df.groupby(x_col)[y_col].median()
        if higher_is_better:
            order = medians.sort_values(ascending=False).index
        else:
            order = medians.sort_values(ascending=True).index
        
        palette = {driver: self.team_colors.get(team, '#CCCCCC') 
                   for driver, team in zip(df[x_col], df['Team'])}

        plt.figure(figsize=(16, 8))
        sns.boxplot(
            data=df, x=x_col, y=y_col, order=order, palette=palette,
            whis=1.5, fliersize=3, linewidth=1.2
        )
        
        arrow_text = "← Better (Higher)" if higher_is_better else "← Better (Lower)"
        plt.annotate(arrow_text, xy=(0, 1.01), xycoords='axes fraction', fontsize=10, 
                     color='gray', fontstyle='italic')

        plt.title(f"{title}\n{self.session.event.year} {self.session.event.EventName}", fontsize=16)
        plt.xlabel("Driver", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
        sns.despine(offset=10, trim=True)
        
        filename = self._get_save_name(filename_suffix)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
        plt.show()

class EntryPhase:
    def __init__(self, parent):
        self.parent = parent

    def braking_dist(self, corner_number):
        print(f"Analyzing Braking Distance for Turn {corner_number}...")
        center_dist = self.parent._get_corner_distance(corner_number)
        start_w, end_w = center_dist - 250, center_dist + 50
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            
            driver_info = self.parent.session.get_driver(drv)
            driver_code = driver_info['Abbreviation']
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    car = lap.get_car_data().add_distance()
                    mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                    zone = car.loc[mask]
                    braking = zone[zone['Brake'] >= 1]
                    if not braking.empty:
                        b_dist = braking['Distance'].max() - braking['Distance'].min()
                        if 10 < b_dist < 250:
                            data.append({'Driver': driver_code, 'Team': team_name, 'Value': b_dist})
                except Exception: continue

        df = pd.DataFrame(data)
        self.parent._plot_distribution(
            df, 'Driver', 'Value', f"Turn {corner_number} Braking Distance", 
            "Braking Distance (m)", f"T{corner_number}_Braking", higher_is_better=False
        )

    def velo_dist(self, corner_number):
        print(f"Analyzing Entry/Apex Velocity for Turn {corner_number}...")
        center_dist = self.parent._get_corner_distance(corner_number)
        start_w, end_w = center_dist - 20, center_dist + 20
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    car = lap.get_car_data().add_distance()
                    mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                    zone = car.loc[mask]
                    if not zone.empty:
                        min_speed = zone['Speed'].min()
                        data.append({'Driver': driver_code['Abbreviation'], 'Team': team_name, 'Value': min_speed})
                except Exception: continue

        df = pd.DataFrame(data)
        self.parent._plot_distribution(
            df, 'Driver', 'Value', f"Turn {corner_number} Apex (Min) Speed", 
            "Speed (km/h)", f"T{corner_number}_ApexSpeed", higher_is_better=True
        )

class ExitPhase:
    def __init__(self, parent):
        self.parent = parent

    def velo_dist(self, corner_number, distance_after=100):
        print(f"Analyzing Exit Velocity for Turn {corner_number} (+{distance_after}m)...")
        center_dist = self.parent._get_corner_distance(corner_number)
        target_dist = center_dist + distance_after
        start_w, end_w = target_dist - 10, target_dist + 10
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    car = lap.get_car_data().add_distance()
                    mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                    zone = car.loc[mask]
                    if not zone.empty:
                        speed = zone['Speed'].mean()
                        data.append({'Driver': driver_code['Abbreviation'], 'Team': team_name, 'Value': speed})
                except Exception: continue

        df = pd.DataFrame(data)
        self.parent._plot_distribution(
            df, 'Driver', 'Value', f"Turn {corner_number} Exit Speed (+{distance_after}m)", 
            "Speed (km/h)", f"T{corner_number}_ExitSpeed", higher_is_better=True
        )

    def throttle_commit(self, corner_number):
        print(f"Analyzing Full Throttle Commitment for Turn {corner_number}...")
        center_dist = self.parent._get_corner_distance(corner_number)
        start_w, end_w = center_dist, center_dist + 300
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    car = lap.get_car_data().add_distance()
                    mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                    zone = car.loc[mask]
                    full_throttle = zone[zone['Throttle'] >= 99]
                    if not full_throttle.empty:
                        dist_to_full = full_throttle['Distance'].min() - center_dist
                        data.append({'Driver': driver_code['Abbreviation'], 'Team': team_name, 'Value': dist_to_full})
                except Exception: continue

        df = pd.DataFrame(data)
        self.parent._plot_distribution(
            df, 'Driver', 'Value', f"Turn {corner_number} Distance to Full Throttle", 
            "Meters after Apex", f"T{corner_number}_ThrottleCommit", higher_is_better=False
        )

class AllPhase:
    def __init__(self, parent):
        self.parent = parent

    def velo_dist(self):
        """
        Calculates the Average Minimum (Apex) Speed across ALL corners for each lap.
        """
        print("Analyzing Average Apex Speed across ALL corners...")
        
        # Get all corner distances
        corners = self.parent.circuit_info.corners
        corner_distances = corners['Distance'].tolist()
        
        data = []
        
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)['Abbreviation']
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    # Get telemetry once per lap to optimize speed
                    car = lap.get_car_data().add_distance()
                    
                    apex_speeds = []
                    
                    for dist in corner_distances:
                        # Window: +/- 20m around apex
                        start_w, end_w = dist - 20, dist + 20
                        
                        mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                        zone = car.loc[mask]
                        
                        if not zone.empty:
                            apex_speeds.append(zone['Speed'].min())
                    
                    # Calculate average apex speed for this lap
                    if apex_speeds:
                        avg_apex_speed = np.mean(apex_speeds)
                        data.append({'Driver': driver_code, 'Team': team_name, 'Value': avg_apex_speed})
                        
                except Exception:
                    continue

        df = pd.DataFrame(data)
        self.parent._plot_distribution(
            df, 'Driver', 'Value', 
            "Average Apex Speed (All Corners)", 
            "Average Speed (km/h)", 
            "AllCorners_ApexSpeed",
            higher_is_better=True
        )

    def braking_dist(self):
        """
        Calculates the Average Braking Distance across ALL braking zones for each lap.
        """
        print("Analyzing Average Braking Distance across ALL corners...")
        
        corners = self.parent.circuit_info.corners
        corner_distances = corners['Distance'].tolist()
        
        data = []
        
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)['Abbreviation']
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    car = lap.get_car_data().add_distance()
                    braking_dists = []
                    
                    for dist in corner_distances:
                        # Look for braking 250m before to 50m after corner
                        start_w, end_w = dist - 250, dist + 50
                        
                        mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                        zone = car.loc[mask]
                        
                        braking = zone[zone['Brake'] >= 1]
                        if not braking.empty:
                            b_dist = braking['Distance'].max() - braking['Distance'].min()
                            # Only include valid braking zones (e.g. not lifting for 5m)
                            if 10 < b_dist < 250:
                                braking_dists.append(b_dist)
                    
                    # Calculate average braking distance for this lap
                    if braking_dists:
                        avg_brake_dist = np.mean(braking_dists)
                        data.append({'Driver': driver_code, 'Team': team_name, 'Value': avg_brake_dist})
                        
                except Exception:
                    continue

        df = pd.DataFrame(data)
        self.parent._plot_distribution(
            df, 'Driver', 'Value', 
            "Average Braking Distance (All Corners)", 
            "Avg Distance (m)", 
            "AllCorners_BrakingDist",
            higher_is_better=False
        )