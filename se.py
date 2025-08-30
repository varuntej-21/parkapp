"""
Parking Slot Editor - Enhanced Version
Run this script to launch the parking slot selection GUI
"""

from parking import ParkingPtsSelection
import sys
import os

def main():
    """Launch the parking slot selection tool"""
    try:
        print("=" * 50)
        print("Parking Slot Editor - Enhanced Version")
        print("=" * 50)
        print("Instructions:")
        print("1. Click 'Upload Image' to load your parking lot image")
        print("2. Select slot type (car/bike) using radio buttons")
        print("3. Click 4 points to create each parking slot polygon")
        print("4. Right-click to cancel current slot creation")
        print("5. Use 'Remove Last Box' to undo last slot")
        print("6. Use 'Clear All' to remove all slots")
        print("7. Click 'Save JSON' to save your parking layout")
        print("8. Click 'Load JSON' to load existing parking layout")
        print("=" * 50)
        print("Starting GUI...")

        # Launch the parking slot selection tool
        ParkingPtsSelection()

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error launching parking slot editor: {str(e)}")
        print("Make sure you have all required dependencies installed:")
        print("- tkinter (usually comes with Python)")
        print("- PIL/Pillow (pip install Pillow)")
        print("- opencv-python (pip install opencv-python)")
        sys.exit(1)

if __name__ == "__main__":
    main()