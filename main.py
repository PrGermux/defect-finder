import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, QWidget, QSizePolicy, QHeaderView
)
from PyQt5.QtGui import QIcon, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import math

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class DefectFinderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Finder")
        self.setGeometry(100, 100, 1400, 900)  # Increased width for better table display
        self.setWindowIcon(QIcon(resource_path('icon.png')))
        self.initUI()

    def initUI(self):
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        font = QFont("Arial", 10)

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("File: ")
        self.file_label.setFont(font)
        self.file_path = QLineEdit()
        self.file_path.setFont(font)
        self.file_path.setReadOnly(True)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setFont(font)
        self.browse_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(self.browse_button)

        # Start and End position
        range_layout = QHBoxLayout()
        self.start_label = QLabel("Start Position:")
        self.start_label.setFont(font)
        self.start_input = QLineEdit()
        self.start_input.setFont(font)
        self.end_label = QLabel("End Position:")
        self.end_label.setFont(font)
        self.end_input = QLineEdit()
        self.end_input.setFont(font)
        self.find_button = QPushButton("Find")
        self.find_button.setFont(font)
        self.find_button.clicked.connect(self.perform_analysis)
        range_layout.addWidget(self.start_label)
        range_layout.addWidget(self.start_input)
        range_layout.addWidget(self.end_label)
        range_layout.addWidget(self.end_input)
        range_layout.addWidget(self.find_button)

        # Plot area
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # Table and statistics layout
        results_layout = QHBoxLayout()

        # Results table
        self.table = QTableWidget()
        self.table.setFont(font)
        self.table.horizontalHeader().setFont(font)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Type", "Position (m)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Updated line
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout.addWidget(self.table)

        # Statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setFont(font)
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels([
            "Mechanical Defects",
            "Non-Mechanical Defects",
            "Mechanical Periodicities (mm)",
            "Non-Mechanical Periodicities (mm)"
        ])
        self.stats_table.horizontalHeader().setFont(font)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Updated line
        results_layout.addWidget(self.stats_table)

        # Add widgets to the main layout
        layout.addLayout(file_layout)
        layout.addLayout(range_layout)
        layout.addWidget(self.canvas)
        layout.addLayout(results_layout)

        central_widget.setLayout(layout)

        # Data placeholders
        self.data = None

        # Figures for histograms
        self.mechanical_fig = None
        self.non_mechanical_fig = None

    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "DAT Files (*.dat);;All Files (*)", options=options)
        if file_path:
            self.file_path.setText(file_path)
            self.data = pd.read_csv(file_path, sep='\t', skiprows=2, header=None, names=["Position", "Ic"])
            self.plot_full_data()

    def plot_full_data(self):
        if self.data is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.data["Position"], self.data["Ic"], label="All Data", color="blue", linewidth=0.8)
            ax.set_xlabel("Position (m)", fontsize=12)
            ax.set_ylabel("Ic [A]", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize=10, loc='upper right')

            # Tighten layout
            self.figure.tight_layout()
            self.canvas.draw()

    def perform_analysis(self):
        if self.data is None:
            self.file_label.setText("Please load a valid file!")
            return

        try:
            start = float(self.start_input.text())
            end = float(self.end_input.text())
        except ValueError:
            self.file_label.setText("Invalid start or end position!")
            return

        filtered_data = self.data.query(f"{start} <= Position <= {end}")
        if filtered_data.empty:
            self.file_label.setText("No data in the specified range!")
            return

        x = filtered_data["Position"].values
        y = filtered_data["Ic"].values

        # Detect abrupt changes using a refined threshold
        delta_y = np.diff(y)
        threshold = np.mean(np.abs(delta_y)) + np.std(np.abs(delta_y))
        significant_changes = np.where(np.abs(delta_y) > threshold)[0]

        # Group points into defects
        defects = []
        if len(significant_changes) > 0:
            current_defect = [significant_changes[0]]
            for i in range(1, len(significant_changes)):
                if significant_changes[i] <= significant_changes[i - 1] + 2:
                    current_defect.append(significant_changes[i])
                else:
                    defects.append(current_defect)
                    current_defect = [significant_changes[i]]
            if current_defect:
                defects.append(current_defect)

        # Categorize defects based on width
        mechanical_defects = []
        non_mechanical_defects = []
        for defect in defects:
            start_idx, end_idx = defect[0], defect[-1]
            defect_points = list(zip(x[start_idx:end_idx + 2], y[start_idx:end_idx + 2]))
            min_point = min(defect_points, key=lambda p: p[1])
            if 5 <= len(defect_points) <= 7:
                mechanical_defects.append(min_point)
            elif len(defect_points) > 7:
                non_mechanical_defects.append(min_point)

        # Extract positions
        mechanical_positions = [pos[0] for pos in mechanical_defects]
        non_mechanical_positions = [pos[0] for pos in non_mechanical_defects]

        # Close existing histogram figures
        if self.mechanical_fig:
            plt.close(self.mechanical_fig)
        if self.non_mechanical_fig:
            plt.close(self.non_mechanical_fig)

        # Detect and visualize for mechanical defects
        mechanical_periodicities = self.detect_and_visualize_periodicities(
            mechanical_positions, title="Mechanical Defect", figure_title="Mechanical Defect Distances Histogram"
        )

        # Detect and visualize for non-mechanical defects
        non_mechanical_periodicities = self.detect_and_visualize_periodicities(
            non_mechanical_positions, title="Non-Mechanical Defect", figure_title="Non-Mechanical Defect Distances Histogram"
        )

        # Update results table
        self.table.setRowCount(len(mechanical_defects) + len(non_mechanical_defects))
        row = 0
        for pos in mechanical_defects:
            self.table.setItem(row, 0, QTableWidgetItem("Mechanical"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{pos[0]:.3f}"))  # Three decimal places for position in m
            row += 1
        for pos in non_mechanical_defects:
            self.table.setItem(row, 0, QTableWidgetItem("Non-Mechanical"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{pos[0]:.3f}"))
            row += 1

        # Update statistics table
        counts = [
            str(len(mechanical_defects)),
            str(len(non_mechanical_defects))
        ]

        # Combine periodicities into columns, ensuring they align properly
        max_periodicities = max(len(mechanical_periodicities), len(non_mechanical_periodicities))
        self.stats_table.setRowCount(max_periodicities + 1)  # +1 for counts

        # Populate counts
        self.stats_table.setItem(0, 0, QTableWidgetItem(counts[0]))
        self.stats_table.setItem(0, 1, QTableWidgetItem(counts[1]))

        # Populate Mechanical Periodicities
        for i in range(len(mechanical_periodicities)):
            formatted_value = f"{mechanical_periodicities[i]:.3f}"
            self.stats_table.setItem(i + 1, 2, QTableWidgetItem(formatted_value))

        # Populate Non-Mechanical Periodicities
        for i in range(len(non_mechanical_periodicities)):
            formatted_value = f"{non_mechanical_periodicities[i]:.3f}"
            self.stats_table.setItem(i + 1, 3, QTableWidgetItem(formatted_value))

        # Adjust table to use full width
        self.stats_table.resizeColumnsToContents()
        self.stats_table.horizontalHeader().setStretchLastSection(True)

        # Plot original data and defects
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y, label="Original Data", color="blue")
        if mechanical_defects:
            ax.scatter(
                [point[0] for point in mechanical_defects],
                [point[1] for point in mechanical_defects],
                label="Mechanical Defect",
                color="red",
            )
        if non_mechanical_defects:
            ax.scatter(
                [point[0] for point in non_mechanical_defects],
                [point[1] for point in non_mechanical_defects],
                label="Non-Mechanical Defect",
                color="orange",
            )
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Ic [A]")
        ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()

    def detect_and_visualize_periodicities(self, positions, title, figure_title):
        if len(positions) < 2:
            print(f"No sufficient data for {title}.")
            return []

        # Calculate all pairwise differences to the right
        positions = np.array(positions)
        distances = []
        for i, pos in enumerate(positions[:-1]):
            distances.extend(positions[i + 1:] - pos)
        distances = np.array(distances) * 1000  # Convert to mm

        # Create histogram
        bins = np.arange(0, max(distances) + 10, 10)  # Bins of width 10 mm
        hist, bin_edges = np.histogram(distances, bins=bins)

        # Calculate mean frequency of histogram
        mean_frequency = np.mean(hist)

        # Identify distances corresponding to histogram bins with frequency > mean and appearing at least 10 times
        periodicities = []
        for i, freq in enumerate(hist):
            if freq > mean_frequency and freq >= 10:
                periodicity = (bin_edges[i] + bin_edges[i + 1]) / 2
                periodicities.append(round(periodicity, 3))  # Three decimal places

        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(bin_edges[:-1], hist, width=10, color="blue", alpha=0.7, edgecolor="black", label="All Distances")

        # Highlight periodic distances
        periodic_indices = [i for i, freq in enumerate(hist) if freq > mean_frequency and freq >= 10]
        periodic_bins = bin_edges[periodic_indices]
        periodic_freqs = hist[periodic_indices]
        ax.bar(
            periodic_bins, periodic_freqs,
            width=10, color="yellow", alpha=0.7, edgecolor="black", label="Periodic Distances",
        )

        ax.axhline(mean_frequency, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean Frequency: {mean_frequency:.1f}")
        ax.set_title(figure_title, fontsize=14)
        ax.set_xlabel("Distance (mm)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.grid(axis="y", alpha=0.75)

        # Add secondary x-axis for periodicities at multiples of 60π and 100π
        secax = ax.secondary_xaxis("top")
        secax.set_xlabel("Circle Periodicity (mm/π)", fontsize=12)

        def update_ticks(event=None):
            """Update top x-axis ticks dynamically to limit to max 13 ticks."""
            # Get current x-axis range
            xmin, xmax = ax.get_xlim()

            # Generate multiples of 60π and 100π
            multiple_60_pi = 60 * math.pi  # Approximately 188.495559 mm
            multiple_100_pi = 100 * math.pi  # Approximately 314.159265 mm

            tick_positions = []
            labels = []

            # Generate multiples for the current x-range
            for base, label in [(multiple_60_pi, '60'), (multiple_100_pi, '100')]:
                n = 1
                while base * n <= xmax:
                    tick_pos = base * n
                    if tick_pos >= xmin:
                        tick_positions.append(tick_pos)
                        labels.append(label)
                    n += 1

            # Combine and sort ticks
            combined_ticks = sorted(zip(tick_positions, labels), key=lambda x: x[0])

            # Limit the number of ticks to a maximum of 13
            max_ticks = 13
            if len(combined_ticks) > max_ticks:
                step = max(1, len(combined_ticks) // max_ticks)
                combined_ticks = combined_ticks[::step][:max_ticks]

            # Apply filtered ticks
            filtered_positions = [pos for pos, label in combined_ticks]
            filtered_labels = [label for pos, label in combined_ticks]

            secax.set_xticks(filtered_positions)
            secax.set_xticklabels(filtered_labels, fontsize=10)

        # Initial tick update
        update_ticks()

        # Connect the event to dynamically update ticks on zoom/pan
        fig.canvas.mpl_connect('draw_event', update_ticks)

        ax.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        plt.show()

        # Store the figure so we can close it later
        if title == "Mechanical Defect":
            self.mechanical_fig = fig
        elif title == "Non-Mechanical Defect":
            self.non_mechanical_fig = fig

        return periodicities

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DefectFinderApp()
    window.showMaximized()
    sys.exit(app.exec_())
