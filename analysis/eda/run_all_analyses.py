#!/usr/bin/env python3
"""
Master Script to Run All Statistical Analyses
"""

import os
import sys
import subprocess
from datetime import datetime

def run_script(script_name):
    """Run a Python script and capture output"""
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd='.')
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    """Run all statistical analysis scripts"""
    
    print("="*80)
    print("DBPs PROJECT - COMPREHENSIVE STATISTICAL ANALYSIS SUITE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List of scripts to run in order
    scripts = [
        'run_analysis.py',
        'advanced_analysis.py', 
        'visualization_suite.py'
    ]
    
    success_count = 0
    
    for script in scripts:
        if os.path.exists(script):
            success = run_script(script)
            if success:
                success_count += 1
                print(f"✓ {script} completed successfully")
            else:
                print(f"✗ {script} failed")
        else:
            print(f"✗ {script} not found")
    
    print(f"\n{'='*80}")
    print("ANALYSIS SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scripts run successfully: {success_count}/{len(scripts)}")
    
    if success_count == len(scripts):
        print("🎉 ALL ANALYSES COMPLETED SUCCESSFULLY!")
        
        # List generated files
        print(f"\n{'='*50}")
        print("GENERATED FILES:")
        print(f"{'='*50}")
        
        files = [f for f in os.listdir('.') if f.endswith(('.csv', '.png', '.md'))]
        files.sort()
        
        csv_files = [f for f in files if f.endswith('.csv')]
        png_files = [f for f in files if f.endswith('.png')]
        md_files = [f for f in files if f.endswith('.md')]
        
        print(f"\n📊 CSV Files ({len(csv_files)}):")
        for f in csv_files:
            print(f"  - {f}")
            
        print(f"\n📈 Visualization Files ({len(png_files)}):")
        for f in png_files:
            print(f"  - {f}")
            
        print(f"\n📝 Report Files ({len(md_files)}):")
        for f in md_files:
            print(f"  - {f}")
        
        print(f"\nTotal files generated: {len(files)}")
    else:
        print("⚠️  Some analyses failed. Check error messages above.")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
