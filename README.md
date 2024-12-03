### Defect Finder

**Defect Finder** is a Python-based GUI application designed for detecting and analyzing mechanical and non-mechanical defects in data. It leverages PyQt5 for an intuitive and interactive user interface, enabling users to visualize data, identify abrupt changes, and analyze periodicities between detected defects. The tool is designed for both research and industrial applications, providing a streamlined workflow for defect identification and statistical analysis.

#### Key Features:
- **Defect Detection**: Automatically identifies mechanical and non-mechanical defects based on abrupt changes and defect width criteria.
- **Periodicities Analysis**: Computes all possible periodicities between defects and filters those with significant occurrences (e.g., appearing at least 3 times).
- **Interactive Visualization**: Allows users to dynamically explore data trends and visualize detected defects on a plotted chart.
- **Data Import**: Supports importing data from `.dat` files for easy integration with various data sources.
- **Customizable Analysis Range**: Users can define start and end positions for focused analysis on specific data segments.
- **Real-Time Results**: Displays defect positions and periodicity statistics in an easy-to-read format, directly within the application.

#### Usage
This tool is ideal for:
- Engineers identifying defects in manufacturing or quality control processes.
- Researchers analyzing trends in experimental or real-world data.
- Professionals requiring a quick and efficient defect detection solution.

#### Python Branch and Complexity
- **Python Branch**: Built using PyQt5 for GUI, `matplotlib` for data visualization, and `pandas` for data handling.
- **Complexity**: Incorporates advanced data analysis, pairwise distance computations for periodicity, and interactive GUI elements, making it moderately complex.

#### Code Structure
- **Main Interface**: Visualizes data trends with interactive defect markers.
- **Defect Detection**: Identifies and categorizes mechanical and non-mechanical defects.
- **Periodicities Analysis**: Finds significant periodicities by analyzing all pairwise distances between defects.
- **Statistics**: Displays counts of defects and significant periodicities in a dedicated statistics panel.

#### Screenshots:

![grafik](https://github.com/user-attachments/assets/f3000a47-d68d-4c67-8389-f82a026d2329)
![grafik](https://github.com/user-attachments/assets/28460fee-92e2-42cf-accb-3a3459d4f532)

#### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/defect-finder.git
   cd defect-finder
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

#### Usage
Run the main application:
```sh
python main.py
```

#### Freezing
To package the application as a standalone executable:
```sh
pyinstaller --onefile --windowed --icon=icon.png --add-data "icon.png;." --name "Defect Finder" main.py
```

#### Dependencies
- Python 3.x
- PyQt5
- matplotlib
- pandas

#### Future Enhancements
- **Enhanced Detection Criteria**: Allow users to define custom defect detection thresholds and criteria.
- **Data Export**: Add functionality to export analysis results and plots for further processing.
- **Multi-Format Data Import**: Support additional data formats like Excel or CSV.
- **Real-Time Monitoring**: Enable continuous defect detection for real-time data streams.
- **Advanced Statistics**: Provide more detailed statistical summaries and visualizations for defects and periodicities.

#### License
This project is licensed under the MIT License for non-commercial use.
