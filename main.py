import sys
import os
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem, QWidget
)
from PyQt5.QtGui import QIcon, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class DefectFinderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Finder")
        self.setGeometry(100, 100, 1200, 800)
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
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Table and statistics layout
        results_layout = QHBoxLayout()

        # Results table
        self.table = QTableWidget()
        self.table.setFont(font)
        self.table.horizontalHeader().setFont(font)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Type", "Position"])
        results_layout.addWidget(self.table)

        # Statistics block
        stats_layout = QVBoxLayout()
        self.mechanical_count_label = QLabel("Mechanical Defects: 0")
        self.mechanical_count_label.setFont(font)
        self.non_mechanical_count_label = QLabel("Non-Mechanical Defects: 0")
        self.non_mechanical_count_label.setFont(font)
        self.mechanical_periodicity_label = QLabel("Mechanical Periodicity: ~None")
        self.mechanical_periodicity_label.setFont(font)
        self.non_mechanical_periodicity_label = QLabel("Non-Mechanical Periodicity: ~None")
        self.non_mechanical_periodicity_label.setFont(font)
        stats_layout.addWidget(self.mechanical_count_label)
        stats_layout.addWidget(self.non_mechanical_count_label)
        stats_layout.addWidget(self.mechanical_periodicity_label)
        stats_layout.addWidget(self.non_mechanical_periodicity_label)
        results_layout.addLayout(stats_layout)

        # Add widgets to the main layout
        layout.addLayout(file_layout)
        layout.addLayout(range_layout)
        layout.addWidget(self.canvas)
        layout.addLayout(results_layout)

        central_widget.setLayout(layout)

        # Data placeholders
        self.data = None

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
        current_defect = [significant_changes[0]] if len(significant_changes) > 0 else []
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

        # Function to detect multiple periodicities in mm
        def detect_multiple_periodicities(positions):
            if len(positions) < 2:
                return []
            
            # Calculate distances between sorted positions
            distances = np.diff(sorted(positions)) * 1000  # Convert to mm
            
            # Use histogram to detect recurring distances
            hist, bin_edges = np.histogram(distances, bins='auto')
            significant_bins = np.where(hist > np.mean(hist))[0]
            
            # Extract periodicities from bin edges
            periodicities = []
            for bin_idx in significant_bins:
                bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                periodicities.append(round(bin_center, 1))  # Retain one decimal place for mm
            
            return periodicities

        # Detect multiple periodicities
        mechanical_periodicities = detect_multiple_periodicities(mechanical_positions)
        non_mechanical_periodicities = detect_multiple_periodicities(non_mechanical_positions)

        # Update table
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

        # Update statistics block
        self.mechanical_count_label.setText(f"Mechanical Defects: {len(mechanical_defects)}")
        self.non_mechanical_count_label.setText(f"Non-Mechanical Defects: {len(non_mechanical_defects)}")
        self.mechanical_periodicity_label.setText(
            f"Mechanical Periodicities (mm): {', '.join(map(str, mechanical_periodicities))}")
        self.non_mechanical_periodicity_label.setText(
            f"Non-Mechanical Periodicities (mm): {', '.join(map(str, non_mechanical_periodicities))}")

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DefectFinderApp()
    window.showMaximized()
    sys.exit(app.exec_())
