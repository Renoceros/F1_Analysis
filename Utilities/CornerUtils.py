import fastf1
import fastf1.plotting
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure FastF1 default styling is applied when this module is imported
try:
    fastf1.plotting.setup_mpl()
except Exception:
    pass

class CircuitAnalyzer:
    """
    Main controller for analyzing corner performance.
    
    Usage:
        Corner = CircuitAnalyzer(session, team_colors)
        Corner.entry.braking_dist(7)
        Corner.entry.velo_dist(7)
        Corner.exit.throttle_commit(7)
    """
    def __init__(self, session, team_colors):
        self.session = session
        self.team_colors = team_colors
        
        # Pre-filter laps to save processing time (Quick laps + No Pit In/Out)
        print("Initializing CircuitAnalyzer: Pre-filtering laps...")
        self.laps = session.laps.pick_quicklaps().pick_wo_box()
        self.circuit_info = session.get_circuit_info()
        
        # Initialize sub-modules
        self.entry = EntryPhase(self)
        self.exit = ExitPhase(self)

    def _get_corner_distance(self, corner_number):
        """Helper to get the track distance (m) of a specific corner."""
        try:
            val = self.circuit_info.corners.loc[self.circuit_info.corners['Number'] == corner_number, 'Distance']
            if val.empty:
                raise ValueError(f"Corner {corner_number} not found.")
            return val.values[0]
        except Exception as e:
            raise ValueError(f"Error finding Corner {corner_number}: {e}")

    def _plot_distribution(self, df, x_col, y_col, title, ylabel, filename_suffix, higher_is_better=False):
        """
        Standardized plotting function for all metrics.
        
        Args:
            higher_is_better (bool): If True, sorts highest median to the left (e.g., Speed).
                                     If False, sorts lowest median to the left (e.g., Distance).
        """
        if df.empty:
            print(f"No data available to plot for {title}")
            return

        # Calculate median for sorting
        medians = df.groupby(x_col)[y_col].median()
        
        # Determine sort order based on metric type
        if higher_is_better:
            order = medians.sort_values(ascending=False).index # Best (Highest) on left
        else:
            order = medians.sort_values(ascending=True).index  # Best (Lowest) on left
        
        # Color mapping
        palette = {driver: self.team_colors.get(team, '#CCCCCC') 
                   for driver, team in zip(df[x_col], df['Team'])}

        plt.figure(figsize=(16, 8))
        sns.boxplot(
            data=df, x=x_col, y=y_col, order=order, palette=palette,
            whis=1.5, fliersize=3, linewidth=1.2
        )
        
        # Add a visual indicator for "Better" direction
        arrow_text = "← Better (Higher)" if higher_is_better else "← Better (Lower)"
        plt.annotate(arrow_text, xy=(0, 1.01), xycoords='axes fraction', fontsize=10, 
                     color='gray', fontstyle='italic')

        plt.title(f"{title}\n{self.session.event.year} {self.session.event.EventName}", fontsize=16)
        plt.xlabel("Driver", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
        sns.despine(offset=10, trim=True)
        
        plt.tight_layout()
        filename = f"Analysis_{filename_suffix}.png"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")
        plt.show()

class EntryPhase:
    def __init__(self, parent):
        self.parent = parent

    def braking_dist(self, corner_number):
        """
        Calculates the braking distance leading into the specific corner.
        Logic: Distance traveled while Brake >= 1 within a 300m window.
        """
        print(f"Analyzing Braking Distance for Turn {corner_number}...")
        
        center_dist = self.parent._get_corner_distance(corner_number)
        start_w, end_w = center_dist - 250, center_dist + 50
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_driver(drv)
            if drv_laps.empty: continue
            
            driver_info = self.parent.session.get_driver(drv)
            driver_code = driver_info['Abbreviation']
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    # Get telemetry slice
                    car = lap.get_car_data().add_distance()
                    mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                    zone = car.loc[mask]
                    
                    # Filter for braking
                    braking = zone[zone['Brake'] >= 1]
                    if not braking.empty:
                        b_dist = braking['Distance'].max() - braking['Distance'].min()
                        # Sanity Check
                        if 10 < b_dist < 250:
                            data.append({'Driver': driver_code, 'Team': team_name, 'Value': b_dist})
                except Exception:
                    continue

        df = pd.DataFrame(data)
        # Lower distance is usually "shorter/more aggressive braking" -> ascending=True
        self.parent._plot_distribution(
            df, 'Driver', 'Value', 
            f"Turn {corner_number} Braking Distance", 
            "Braking Distance (m)", 
            f"T{corner_number}_Braking",
            higher_is_better=False
        )

    def velo_dist(self, corner_number):
        """
        Calculates Minimum Speed (Apex Speed) at the corner.
        """
        print(f"Analyzing Entry/Apex Velocity for Turn {corner_number}...")
        
        center_dist = self.parent._get_corner_distance(corner_number)
        start_w, end_w = center_dist - 20, center_dist + 20
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_driver(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)['Abbreviation']
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    car = lap.get_car_data().add_distance()
                    mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                    zone = car.loc[mask]
                    
                    if not zone.empty:
                        min_speed = zone['Speed'].min()
                        data.append({'Driver': driver_code, 'Team': team_name, 'Value': min_speed})
                except Exception:
                    continue

        df = pd.DataFrame(data)
        # Higher speed is better -> ascending=False
        self.parent._plot_distribution(
            df, 'Driver', 'Value', 
            f"Turn {corner_number} Apex (Min) Speed", 
            "Speed (km/h)", 
            f"T{corner_number}_ApexSpeed",
            higher_is_better=True
        )

class ExitPhase:
    def __init__(self, parent):
        self.parent = parent

    def velo_dist(self, corner_number, distance_after=100):
        """
        Calculates speed at a fixed distance AFTER the corner (Traction/Exit).
        Default checks 100m after the corner apex.
        """
        print(f"Analyzing Exit Velocity for Turn {corner_number} (+{distance_after}m)...")
        
        center_dist = self.parent._get_corner_distance(corner_number)
        target_dist = center_dist + distance_after
        start_w, end_w = target_dist - 10, target_dist + 10
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_driver(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)['Abbreviation']
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    car = lap.get_car_data().add_distance()
                    mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                    zone = car.loc[mask]
                    
                    if not zone.empty:
                        speed = zone['Speed'].mean()
                        data.append({'Driver': driver_code, 'Team': team_name, 'Value': speed})
                except Exception:
                    continue

        df = pd.DataFrame(data)
        # Higher speed is better -> ascending=False
        self.parent._plot_distribution(
            df, 'Driver', 'Value', 
            f"Turn {corner_number} Exit Speed (+{distance_after}m)", 
            "Speed (km/h)", 
            f"T{corner_number}_ExitSpeed",
            higher_is_better=True
        )

    def throttle_commit(self, corner_number):
        """
        Calculates the distance AFTER the apex where the driver hits 99%+ throttle.
        Smaller number = Earlier on the gas (Better traction/Confidence).
        """
        print(f"Analyzing Full Throttle Commitment for Turn {corner_number}...")
        
        center_dist = self.parent._get_corner_distance(corner_number)
        # Look from Apex (0m) to 300m after
        start_w, end_w = center_dist, center_dist + 300
        
        data = []
        for drv in self.parent.session.drivers:
            drv_laps = self.parent.laps.pick_driver(drv)
            if drv_laps.empty: continue
            
            driver_code = self.parent.session.get_driver(drv)['Abbreviation']
            team_name = drv_laps.iloc[0]['Team']

            for _, lap in drv_laps.iterlaps():
                try:
                    car = lap.get_car_data().add_distance()
                    mask = (car['Distance'] > start_w) & (car['Distance'] < end_w)
                    zone = car.loc[mask]
                    
                    # Find first point where Throttle >= 99
                    full_throttle = zone[zone['Throttle'] >= 99]
                    
                    if not full_throttle.empty:
                        # Distance from Apex to that point
                        dist_to_full = full_throttle['Distance'].min() - center_dist
                        data.append({'Driver': driver_code, 'Team': team_name, 'Value': dist_to_full})
                except Exception:
                    continue

        df = pd.DataFrame(data)
        # Lower distance is better -> ascending=True
        self.parent._plot_distribution(
            df, 'Driver', 'Value', 
            f"Turn {corner_number} Distance to Full Throttle", 
            "Meters after Apex", 
            f"T{corner_number}_ThrottleCommit",
            higher_is_better=False
        )