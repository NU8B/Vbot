import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Now we can import from tha4
from tha4.app.autonomous_animation import AutonomousAnimationFrame
import wx
import torch

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    app = wx.App()
    frame = AutonomousAnimationFrame(device)
    frame.Show()
    app.MainLoop()

if __name__ == "__main__":
    main() 