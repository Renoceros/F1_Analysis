import fastf1
import fastf1.plotting
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

# Apply FastF1 styling
try:
    fastf1.plotting.setup_mpl()
except Exception:
    pass

class StraightAnalyzer:
    """
    Analysis of performance on Straights (V-Max, Drag, Acceleration).
    
    Usage:
        Straight = StraightAnalyzer(session)
        Straight.speed.vmax_dist(start_turn=12, end_turn=14)
        Straight.accel.time_to_speed(100, 200, after_turn=1)
    """
    def __init__(self, session, team_colors=None):
        self.session = session
        
        # Load colors
        if team_colors is None:
            self.team_colors = self._load_default_colors()
        else:
            self.team_colors = team_colors
        
        print("Initializing StraightAnalyzer: Pre-filtering laps...")
        self.laps = session.laps.pick_quicklaps().pick_wo_box()
        self.circuit_info = session.get_circuit_info()
        
        # Initialize Sub-modules
        self.speed = SpeedPhase(self)
        self.accel = AccelPhase(self)

    def _load_default_colors(self):
        """Loads team_colors.json from the same directory."""
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

    def _plot_distribution(self, df, x_col, y_col, title, ylabel, filename_suffix, higher_is_better=True):
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

class SpeedPhase:
    def __init__(self, parent):
        self.parent = parent

    def vmax_dist(self, start_turn, end_turn):
        """
        Calculates the Maximum Speed reached between two corners.
        Example: Vegas Strip is usually Turn 12 to Turn 14.
        """
        print(f"Analyzing Max Speed between Turn {start_turn} and Turn {end_turn}...")
        
        try:
            start_dist = self.parent._get_corner_distance(start_turn)
            end_dist = self.parent._get_corner_distance(end_turn)
        except ValueError as e:
            print(e)
            return

        # Handle wrap-around (crossing finish line)
        crosses_finish = False
        if start_dist > end_dist:
            crosses_finish = True
            print("Note: Segment crosses the finish line.")
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)['Abbreviation']
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    # Get telemetry
                    car = lap.get_car_data().add_distance()
                    
                    # Slice the straight
                    if crosses_finish:
                        mask = (car['Distance'] > start_dist) | (car['Distance'] < end_dist)
                    else:
                        mask = (car['Distance'] > start_dist) & (car['Distance'] < end_dist)
                        
                    zone = car.loc[mask]
                    
                    if not zone.empty:
                        # Max speed in this zone
                        v_max = zone['Speed'].max()
                        data.append({'Driver': driver_code, 'Team': team_name, 'Value': v_max})
                except Exception:
                    continue

        df = pd.DataFrame(data)
        self.parent._plot_distribution(
            df, 'Driver', 'Value', 
            f"V-Max Distribution (Turn {start_turn} -> {end_turn})", 
            "Speed (km/h)", 
            f"Straight_VMax_T{start_turn}_T{end_turn}",
            higher_is_better=True
        )

class AccelPhase:
    def __init__(self, parent):
        self.parent = parent

    def time_to_speed(self, start_speed, end_speed, after_turn):
        """
        Calculates the time taken to accelerate from start_speed to end_speed.
        Starts searching immediately after the specific corner.
        """
        print(f"Analyzing Acceleration ({start_speed}->{end_speed} kph) after Turn {after_turn}...")
        
        try:
            corner_dist = self.parent._get_corner_distance(after_turn)
        except ValueError as e:
            print(e)
            return
            
        # Window: Look at the 1000m AFTER the corner to find the acceleration event
        # This prevents picking up acceleration from a completely different part of the track
        search_start = corner_dist
        search_end = corner_dist + 1000
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_drivers(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)['Abbreviation']
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    car = lap.get_car_data().add_distance()
                    
                    # 1. Slice to the area after the corner
                    mask = (car['Distance'] > search_start) & (car['Distance'] < search_end)
                    zone = car.loc[mask]
                    
                    if zone.empty: continue
                    
                    # 2. Find timestamp where we pass Start Speed
                    start_points = zone[zone['Speed'] >= start_speed]
                    if start_points.empty: continue
                    t_start = start_points.iloc[0]['Time']
                    
                    # 3. Find timestamp where we pass End Speed
                    # We only look at points AFTER the start point
                    end_points = zone[(zone['Speed'] >= end_speed) & (zone['Time'] > t_start)]
                    if end_points.empty: continue
                    t_end = end_points.iloc[0]['Time']
                    
                    # 4. Calculate Delta
                    delta_seconds = (t_end - t_start).total_seconds()
                    
                    # Sanity check (e.g., if it took 10 seconds to go 100-200, they probably lifted)
                    if 0.5 < delta_seconds < 8.0:
                        data.append({'Driver': driver_code, 'Team': team_name, 'Value': delta_seconds})

                except Exception:
                    continue

        df = pd.DataFrame(data)
        self.parent._plot_distribution(
            df, 'Driver', 'Value', 
            f"Acceleration Time ({start_speed}-{end_speed} kph) after Turn {after_turn}", 
            "Time (seconds)", 
            f"Straight_Accel_T{after_turn}",
            higher_is_better=False # Lower time is better
        )