import matplotlib
matplotlib.use('TkAgg') # Use a reliable interactive backend
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- All initial parameters are defined up top ---

# Analysis Parameters
IMAGE_PATH = 'images/chargefem1_gs.jpg'
TEMP_RANGE_C = [20.2, 38.2]
# IMPORTANT: These are the real-world dimensions of the object you will trace in Step 1.
BOARD_DIMS_MM = [363.47, 280.0] # [height, width]
PLOT_TITLE = 'Interactive Power Integration (2-Stage)'

# Model Parameters
EMISSIVITY = 0.95
H_CONVECTION = 10.0
INCLUDE_CONVECTION = True
T_AMBIENT_C = 20.0

# --- Main calculation and plotting logic ---

# 1. Load Image and Create Temperature Map in Kelvin
img = Image.open(IMAGE_PATH).convert('L')
temp_map_k = TEMP_RANGE_C[0] + (np.array(img)/255.0 * (TEMP_RANGE_C[1] - TEMP_RANGE_C[0])) + 273.15

# 2. Define Physical Constants
T_ambient_k = T_AMBIENT_C + 273.15
sigma_sb = 5.67e-8

# 3. Calculate Power Density Map (W/m^2)
power_density_rad_w_m2 = 2 * EMISSIVITY * sigma_sb * (temp_map_k**4 - T_ambient_k**4)
total_power_density_w_m2 = power_density_rad_w_m2
if INCLUDE_CONVECTION:
    power_density_conv_w_m2 = 2 * H_CONVECTION * (temp_map_k - T_ambient_k)
    total_power_density_w_m2 += power_density_conv_w_m2

# 4. Set up the plot that will be used for interaction
fig, ax = plt.subplots(figsize=(12, 9))
plt.title(PLOT_TITLE, fontsize=16)
ax.set_xlim(0, img.size[0] - 100) # Crop the x-axis as requested

# Convert power density to W/mm^2 for plotting
power_density_plot_w_mm2 = total_power_density_w_m2 / (1000.0**2)
power_plot = ax.imshow(power_density_plot_w_mm2, cmap='inferno')

# Add and format the color bar
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(power_plot, cax=cax)
cbar.set_label('Power Density (W/mm²)', rotation=270, labelpad=20, fontsize=12)

# Add a text box for instructions and results
results_text = ax.text(0.02, 0.98, "Initializing...",
                         transform=ax.transAxes, fontsize=12, color='white', va='top',
                         bbox=dict(boxstyle='round,pad=0.5', fc='black', alpha=0.7))


# 5. Define the class that will manage the two-stage interaction
class PowerIntegrator:
    def __init__(self):
        self.pixel_area_m2 = None
        self.avg_scale_mm_per_px = None
        
        # Update instructions for the first stage
        results_text.set_text(
            f"STEP 1: Calibrate Scale\n"
            f"Draw a rectangle over the board area.\n"
            f"(Known board size: {BOARD_DIMS_MM[1]:.1f} x {BOARD_DIMS_MM[0]:.1f} mm)"
        )
        
        # Create and activate the first selector for calibration
        self.selector_cal = RectangleSelector(ax, self.on_calibrate_select, useblit=False, button=[1],
                                              minspanx=5, minspany=5, spancoords='pixels', interactive=True,
                                              props=dict(edgecolor='yellow', facecolor='none', linestyle='--', lw=1.5))

    def on_calibrate_select(self, eclick, erelease):
        """Callback for the first (calibration) selection."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Calculate pixel dimensions of the user's box
        px_width = abs(x2 - x1)
        px_height = abs(y2 - y1)
        
        # Calculate scale based on this selection
        scale_x = BOARD_DIMS_MM[1] / px_width  # mm/px
        scale_y = BOARD_DIMS_MM[0] / px_height # mm/px
        self.avg_scale_mm_per_px = (scale_x + scale_y) / 2.0
        self.pixel_area_m2 = (self.avg_scale_mm_per_px**2) / (1000.0**2)
        
        # Update instructions for the second stage
        results_text.set_text(
            f"Scale Set: {self.avg_scale_mm_per_px:.3f} mm/px\n\n"
            f"STEP 2: Integrate Power\n"
            f"Draw a new rectangle to measure its power."
        )

        # Disconnect the calibration selector and activate the integration one
        self.selector_cal.set_active(False)
        self.selector_int = RectangleSelector(ax, self.on_integrate_select, useblit=False, button=[1],
                                              minspanx=5, minspany=5, spancoords='pixels', interactive=True,
                                              props=dict(edgecolor='cyan', facecolor='none', linestyle='-', lw=1.5))
                                              
    def on_integrate_select(self, eclick, erelease):
        """Callback for all subsequent (integration) selections."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Slice the main power density map with the new coordinates
        roi_w_m2 = total_power_density_w_m2[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
        
        # Integrate power for the selection
        integrated_power = np.sum(roi_w_m2) * self.pixel_area_m2
        
        # Get dimensions of the new box in mm
        selected_dims_mm = (abs(x2-x1) * self.avg_scale_mm_per_px, abs(y2-y1) * self.avg_scale_mm_per_px)
        
        # Update text with results
        results_text.set_text(
            f"Scale: {self.avg_scale_mm_per_px:.3f} mm/px\n\n"
            f"Region: {selected_dims_mm[0]:.1f} x {selected_dims_mm[1]:.1f} mm\n"
            f"Integrated Power: {integrated_power:.4f} W"
        )

# Create an instance of our interactive tool and show the plot
integrator = PowerIntegrator()
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()
