import fastf1
import fastf1.plotting
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

try:
    fastf1.plotting.setup_mpl()
except Exception:
    pass

class TimingAnalyzer:
    """
    Macro-level analysis of Race Pace, Tyre Degradation, and Stints.
    
    Usage:
        # Automatically loads colors from team_colors.json in this folder
        Timing = TimingAnalyzer(session)
        Timing.pace_distribution()
    """
    def __init__(self, session, team_colors=None):
        self.session = session
        
        if team_colors is None:
            self.team_colors = self._load_default_colors()
        else:
            self.team_colors = team_colors

        self.laps = session.laps.pick_quicklaps().pick_wo_box()
        self.all_laps = session.laps 

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
        event_name = self.session.event.EventName.replace(" ", "")
        year = self.session.event.year
        session_type_map = {
            'Practice 1': 'FP1', 'Practice 2': 'FP2', 'Practice 3': 'FP3',
            'Qualifying': 'Q', 'Sprint Qualifying': 'SQ', 'Sprint Shootout': 'SQ',
            'Sprint': 'S', 'Race': 'R'
        }
        session_id = session_type_map.get(self.session.name, self.session.name.replace(" ", ""))
        return f"{event_name}{year}_{session_id}_{suffix}.png"

    def pace_distribution(self):
        print("Generating Pace Distribution Boxplot...")
        df = self.laps.copy()
        df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
        
        order = df.groupby('Driver')['LapTimeSec'].median().sort_values().index
        palette = {driver: self.team_colors.get(team, '#CCCCCC') 
                   for driver, team in zip(df['Driver'], df['Team'])}

        plt.figure(figsize=(16, 8))
        sns.boxplot(
            data=df, x='Driver', y='LapTimeSec', 
            order=order, palette=palette, 
            whis=1.5, fliersize=3, linewidth=1.2
        )
        
        plt.title(f"Race Pace Distribution - {self.session.event.year} {self.session.event.EventName}", fontsize=16)
        plt.xlabel("Driver", fontsize=12)
        plt.ylabel("Lap Time (s)", fontsize=12)
        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
        sns.despine(offset=10, trim=True)
        
        filename = self._get_save_name("Timing_PaceDistribution")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.show()

    def tyre_degradation(self, drivers=None, compound=None):
        print("Analyzing Tyre Degradation & Consistency...")
        df = self.laps.copy()
        df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
        
        if drivers is None:
            drivers = list(df['Driver'].unique())[:5]
            
        df = df[df['Driver'].isin(drivers)]
        if compound:
            df = df[df['Compound'] == compound]

        plt.figure(figsize=(12, 8))
        for drv in drivers:
            drv_data = df[df['Driver'] == drv]
            if drv_data.empty: continue
            
            color = self.team_colors.get(drv_data.iloc[0]['Team'], '#CCCCCC')
            sns.regplot(
                data=drv_data, x='LapNumber', y='LapTimeSec',
                label=drv, color=color, scatter_kws={'s': 20, 'alpha': 0.6},
                line_kws={'linewidth': 2}
            )

        plt.title(f"Tyre Degradation Analysis (Pace Evolution)", fontsize=16)
        plt.xlabel("Lap Number", fontsize=12)
        plt.ylabel("Lap Time (s)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        sns.despine(offset=10, trim=True)
        
        filename = self._get_save_name("Timing_TyreDegradation")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.show()

    def delta_to_best(self):
        print("Calculating Delta to Session Best Lap...")
        fastest_lap = self.laps.pick_fastest()
        best_time = fastest_lap['LapTime'].total_seconds()
        print(f"Session Best: {fastest_lap['Driver']} - {best_time:.3f}s")
        
        df = self.laps.copy()
        df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
        df['DeltaToBest'] = df['LapTimeSec'] - best_time
        
        order = df.groupby('Driver')['DeltaToBest'].median().sort_values().index
        palette = {driver: self.team_colors.get(team, '#CCCCCC') 
                   for driver, team in zip(df['Driver'], df['Team'])}

        plt.figure(figsize=(16, 8))
        sns.stripplot(
            data=df, x='Driver', y='DeltaToBest',
            order=order, palette=palette,
            size=4, alpha=0.7, jitter=0.25
        )
        sns.pointplot(
            data=df, x='Driver', y='DeltaToBest',
            order=order, color='white', 
            errorbar=None, markers="D", scale=0.6, join=False
        )

        plt.title(f"Pace Deficit to Best Lap (+{best_time:.3f}s)", fontsize=16)
        plt.xlabel("Driver (Sorted by Median Deficit)", fontsize=12)
        plt.ylabel("Seconds Slower than Best Lap", fontsize=12)
        plt.ylim(-0.5, 5.0) 
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        sns.despine(offset=10, trim=True)
        
        filename = self._get_save_name("Timing_DeltaToBest")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.show()