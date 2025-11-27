import fastf1
import fastf1.plotting
import fastf1.utils
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Apply FastF1 default styling
try:
    fastf1.plotting.setup_mpl()
except Exception:
    pass

class TelemetryAnalyzer:
    """
    Micro-level analysis of Car Physics and Line Traces.
    
    Usage:
        Tel = TelemetryAnalyzer(session)
        Tel.export_to_csv()
        Tel.speed_comparison(drivers=['VER', 'NOR'])
        Tel.delta_to_driver(ref_driver='VER', comp_driver='NOR')
    """
    def __init__(self, session, team_colors=None):
        self.session = session
        
        # Load colors if not provided
        if team_colors is None:
            self.team_colors = self._load_default_colors()
        else:
            self.team_colors = team_colors
            
        # We generally want the fastest laps for comparisons, 
        # but the export function handles all laps independently.
        self.laps = session.laps.pick_quicklaps().pick_wo_box()

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

    def _get_save_name(self, suffix):
        """Generates consistent filenames."""
        event_name = self.session.event.EventName.replace(" ", "")
        year = self.session.event.year
        session_type_map = {
            'Practice 1': 'FP1', 'Practice 2': 'FP2', 'Practice 3': 'FP3',
            'Qualifying': 'Q', 'Sprint Qualifying': 'SQ', 'Sprint Shootout': 'SQ',
            'Sprint': 'S', 'Race': 'R'
        }
        session_id = session_type_map.get(self.session.name, self.session.name.replace(" ", ""))
        return f"{event_name}{year}_{session_id}_{suffix}"

    def export_to_csv(self, output_folder="telemetry_data"):
        """
        Exports detailed telemetry for every driver to CSV files.
        """
        print(f"Starting Telemetry Export to '{output_folder}'...")
        
        # Ensure output directory exists
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a specific subfolder for this session to keep things organized
        session_subfolder = self._get_save_name("Exports")
        final_dir = out_dir / session_subfolder
        final_dir.mkdir(parents=True, exist_ok=True)

        for drv in self.session.drivers:
            driver_info = self.session.get_driver(drv)
            name = driver_info["Abbreviation"]
            
            # Select all laps for this driver (including slow ones for completeness)
            driver_laps = self.session.laps.pick_drivers(drv)

            if driver_laps.empty:
                continue

            print(f"Processing {name}...")
            
            # Initialize empty DataFrame
            full_tel = pd.DataFrame()
            
            # Iterate laps and append telemetry
            for _, lap in driver_laps.iterlaps():
                try:
                    # Get telemetry and add lap number
                    tel = lap.get_telemetry()
                    tel["LapNumber"] = lap["LapNumber"]
                    tel["Driver"] = name
                    full_tel = pd.concat([full_tel, tel], ignore_index=True)
                except Exception:
                    continue

            # Save CSV
            if not full_tel.empty:
                file_name = f"{name}_Telemetry.csv"
                full_tel.to_csv(final_dir / file_name, index=False)
                
        print(f"All files saved to: {final_dir}")

    def speed_comparison(self, drivers=None):
        """
        Plots Speed vs Distance for the fastest lap of selected drivers.
        """
        if drivers is None:
            print("Please provide a list of drivers (e.g. ['VER', 'NOR'])")
            return

        print(f"Generating Speed Trace Comparison: {drivers}...")
        
        plt.figure(figsize=(16, 8))
        
        for drv in drivers:
            # Get fastest lap
            fastest_lap = self.laps.pick_drivers(drv).pick_fastest()
            if fastest_lap is None: continue
            
            team = fastest_lap['Team']
            color = self.team_colors.get(team, '#CCCCCC')
            
            # Get Car Data
            car_data = fastest_lap.get_car_data().add_distance()
            
            plt.plot(car_data['Distance'], car_data['Speed'], 
                     color=color, label=f"{drv} ({fastest_lap['LapTime'].total_seconds():.3f}s)",
                     linewidth=2)

        plt.title(f"Top Speed Comparison - Fastest Lap", fontsize=16)
        plt.xlabel("Distance (m)", fontsize=12)
        plt.ylabel("Speed (km/h)", fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.3)
        sns.despine(offset=10, trim=True)
        
        filename = self._get_save_name("Telemetry_SpeedTrace.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.show()

    def delta_to_driver(self, ref_driver, comp_driver):
        """
        Plots the time delta (gap) between two drivers over a single lap.
        Positive Delta = Comparison driver is SLOWER.
        """
        print(f"Calculating Delta: {ref_driver} (Ref) vs {comp_driver}...")
        
        # Get fastest laps
        lap_ref = self.laps.pick_drivers(ref_driver).pick_fastest()
        lap_comp = self.laps.pick_drivers(comp_driver).pick_fastest()
        
        if lap_ref is None or lap_comp is None:
            print("Could not find laps for one or both drivers.")
            return

        # Calculate Delta
        delta_time, ref_tel, comp_tel = fastf1.utils.delta_time(lap_ref, lap_comp)
        
        # Get colors
        ref_color = self.team_colors.get(lap_ref['Team'], '#CCCCCC')
        comp_color = self.team_colors.get(lap_comp['Team'], '#CCCCCC')

        plt.figure(figsize=(16, 8))
        
        # Plot Delta Line
        plt.plot(ref_tel['Distance'], delta_time, color='white', linewidth=1.5)
        
        # Fill area to show who is ahead
        # If delta < 0 (Ref is slower/behind), fill with Ref color
        # If delta > 0 (Ref is faster/ahead), fill with Comp color (Comp is losing time)
        
        # Note: fastf1 delta_time is calculated as (comp - ref).
        # So if result is POSITIVE, Comp is BEHIND (Ref is faster).
        
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        plt.fill_between(ref_tel['Distance'], delta_time, 0, 
                         where=delta_time > 0, 
                         facecolor=comp_color, alpha=0.5, label=f"{comp_driver} losing time")
        
        plt.fill_between(ref_tel['Distance'], delta_time, 0, 
                         where=delta_time < 0, 
                         facecolor=ref_color, alpha=0.5, label=f"{ref_driver} losing time")

        plt.title(f"Time Delta: {ref_driver} (Reference) vs {comp_driver}", fontsize=16)
        plt.xlabel("Distance (m)", fontsize=12)
        plt.ylabel(f"Gap (s)\n(Above line = {ref_driver} is faster)", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.3)
        sns.despine(offset=10, trim=True)
        
        filename = self._get_save_name(f"Telemetry_Delta_{ref_driver}_vs_{comp_driver}.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.show()

    def throttle_comparison(self, drivers=None):
        """
        Plots Throttle % vs Distance.
        """
        if drivers is None:
            print("Please provide a list of drivers.")
            return

        print(f"Generating Throttle Trace Comparison: {drivers}...")
        
        plt.figure(figsize=(16, 5))
        
        for drv in drivers:
            fastest_lap = self.laps.pick_drivers(drv).pick_fastest()
            if fastest_lap is None: continue
            
            team = fastest_lap['Team']
            color = self.team_colors.get(team, '#CCCCCC')
            
            car_data = fastest_lap.get_car_data().add_distance()
            
            plt.plot(car_data['Distance'], car_data['Throttle'], 
                     color=color, label=drv, linewidth=1.5)

        plt.title(f"Throttle Application Comparison", fontsize=16)
        plt.xlabel("Distance (m)", fontsize=12)
        plt.ylabel("Throttle %", fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.3)
        sns.despine(offset=10, trim=True)
        
        filename = self._get_save_name("Telemetry_ThrottleTrace.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.show()