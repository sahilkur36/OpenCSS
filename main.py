## ############################################################### ##
## OpenCSS ()                                                      ##
##                                                                 ##
## Developed by:                                                   ## 
##       Jesus D. Caballero (caballerojd@uninorte.edu.co)          ##
##        Cesar A. Pájaro (cesar.pajaromiranda@canterbury.ac.nz)   ##
##       Carlos A. Arteta (carteta@uninorte.edu.co)                ##
## ############################################################### ##

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QButtonGroup, QRadioButton, QTabWidget,
    QMessageBox, QWidget, QSplitter, QComboBox, QGroupBox, QListWidget, QDoubleSpinBox, QInputDialog, QSizePolicy
    , QGridLayout, QCheckBox
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import numpy as np
from scipy import interpolate
import re
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import openquake.hazardlib.gsim as gsim
from openquake.hazardlib.gsim.base import GMPE
import pkgutil
import inspect
import os, shutil
from io import StringIO
import importlib
from correlations import Jayaram_Baker, Macedo_Liu20
from openquake.hazardlib.imt import SA, PGA
from openquake.hazardlib.contexts import RuptureContext, SitesContext, DistancesContext
import subprocess
import time

# Lista para almacenar las GMPEs encontradas
gmpe_classes = []

# Iterar sobre los módulos dentro de openquake.hazardlib.gsim
for _, modname, _ in pkgutil.iter_modules(gsim.__path__):
    try:
        module = __import__(f"openquake.hazardlib.gsim.{modname}", fromlist=[""])
        # Filtrar solo las clases que heredan de GMPE
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, GMPE) and obj is not GMPE:
                gmpe_classes.append(f"{modname}.{name}")
    except (ImportError, AttributeError):
        continue
gmpe_classes_label = [gmpe.split('.')[1] for gmpe in gmpe_classes]
gmpe_map = dict(zip(gmpe_classes_label, gmpe_classes))

class HazardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Aquí guardamos la info de hazard para graficar UHS
        self.HzCurvess_Dict = {}
        self.gmpe_weights = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Open CSS")
        self.setGeometry(100, 100, 1200, 800)

        # Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                height: 30px;
                width: 80px;                
                font-family: Arial;         
                font-size: 14px;
                font-weight: bold;         
                font-style: italic;        
                color: black; }
        """)

        # -------------- Pestana Hazard --------------
        self.hazard_tab = QWidget()
        self.init_hazard_tab()
        self.tabs.addTab(self.hazard_tab, "Hazard")

        # -------------- Pestana CMS --------------
        self.cms_tab = QWidget()
        self.init_cms_tab()
        self.tabs.addTab(self.cms_tab, "CMS")
        
        # -------------- Pestaña CSS --------------
        self.css_tab = QWidget()
        self.init_css_tab()
        self.tabs.addTab(self.css_tab, "CSS")        

    def init_hazard_tab(self):
        main_layout = QHBoxLayout()

        # --------------- Panel Izquierdo ---------------
        left_panel = QVBoxLayout()
        
        # --------------- Grupo de Base de Datos ---------------
        database_group = QGroupBox("Database Selection")
        database_group.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        database_group.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)  # Se ajusta automáticamente
        database_layout = QVBoxLayout()
        
        # ----------------- Opciones de Base de Datos -----------------
        self.database_group = QButtonGroup(self)
        self.radio_usgs = QRadioButton("USGS")
        self.radio_sgc = QRadioButton("SGC")
        self.radio_others = QRadioButton("Others")
        self.radio_usgs.setChecked(True)
        
        self.database_group.addButton(self.radio_usgs)
        self.database_group.addButton(self.radio_sgc)
        self.database_group.addButton(self.radio_others)
        
        database_buttons_layout = QVBoxLayout()
        database_buttons_layout.addWidget(self.radio_usgs)
        database_buttons_layout.addWidget(self.radio_sgc)
        database_buttons_layout.addWidget(self.radio_others)
        
        # ----------------- Selección de Ciudad -----------------
        ciudad_label = QLabel("Select City (SGC):")
        self.ciudad_combo = QComboBox()
        ciudades_colombia = [
            "Arauca", "Armenia", "Barranquilla", "Bogota", "Bucaramanga", "Cali",
            "Cartagena", "Cucuta", "Florencia", "Ibague", "Inirida", "Leticia",
            "Manizales", "Medellin", "Mitu", "Mocoa", "Monteria", "Neiva",
            "Pasto", "Pereira", "Popayan", "Puerto Carreno", "Quibdo", "Riohacha",
            "San Andres", "San Jose del Guaviare", "Santa Marta", "Sincelejo",
            "Tunja", "Valledupar", "Villavicencio", "Yopal", "Other"
        ]
        self.ciudad_combo.addItems(ciudades_colombia)
        
        ciudad_layout = QVBoxLayout()
        ciudad_layout.addWidget(ciudad_label)
        ciudad_layout.addWidget(self.ciudad_combo)
        
        # ----------------- Selección de Archivo -----------------
        self.file_label = QLabel("Selected File:")
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        self.file_button = QPushButton("Select File")
        self.file_button.setFixedSize(130, 30)
        self.file_button.clicked.connect(self.select_file)
        
        file_layout = QVBoxLayout()
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.file_button)
        
        # Agregar elementos al layout del grupo de Base de Datos
        database_layout.addLayout(database_buttons_layout)
        database_layout.addLayout(ciudad_layout)
        database_layout.addLayout(file_layout)
        database_group.setLayout(database_layout)
        
        # --------------- Agregar el Grupo al Panel Izquierdo ---------------
        left_panel.addWidget(database_group)
        
        # --------------- Grupo de Periodos de Retorno (TRs) ---------------
        tr_group = QGroupBox("Return Periods")
        tr_group.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        tr_group.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        
        # Layout para los TRs
        tr_layout = QVBoxLayout()
        tr_label = QLabel("Enter TRs (comma-separated):")
        self.tr_input = QLineEdit()
        
        # Agregar al layout del grupo
        tr_layout.addWidget(tr_label)
        tr_layout.addWidget(self.tr_input)
        tr_group.setLayout(tr_layout)
        
        # Agregar el grupo al panel izquierdo
        left_panel.addWidget(tr_group)
        
        # --------------- Grupo de Periodos de Retorno (TRs) ---------------
        disaggregation_group_hazard = QGroupBox("Disaggregation")
        disaggregation_group_hazard.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        disaggregation_hazard_layout = QVBoxLayout()
        
        # Selector de Carpeta (Select Folder)
        self.folder_label_hazard = QLabel("Selected Folder (USGS):")
        self.folder_input_hazard = QLineEdit()
        self.folder_input_hazard.setReadOnly(True)  # Evitar que el usuario edite manualmente la ruta
        
        self.folder_button_hazard = QPushButton("Select Folder")
        self.folder_button_hazard.setFixedSize(130, 30)
        self.folder_button_hazard.clicked.connect(lambda: self.select_disaggregation_folder(self.folder_input_hazard))
        
        # Agregar elementos al layout del grupo
        disaggregation_hazard_layout.addWidget(self.folder_label_hazard)
        disaggregation_hazard_layout.addWidget(self.folder_input_hazard)
        disaggregation_hazard_layout.addWidget(self.folder_button_hazard)
        disaggregation_group_hazard.setLayout(disaggregation_hazard_layout)
        
        # Agregar el grupo al panel izquierdo
        left_panel.addWidget(disaggregation_group_hazard)

        
        # Botón para graficar curvas de amenaza
        self.plot_hazard_button = QPushButton("Calculate")
        self.plot_hazard_button.setStyleSheet("font-weight: bold;")
        self.plot_hazard_button.setFixedSize(130, 30)
        self.plot_hazard_button.clicked.connect(self.load_and_plot)
        self.plot_hazard_button.clicked.connect(self.plot_uhs)
        self.plot_hazard_button.clicked.connect(self.plot_disaggregation_results)     
        
        left_panel.addWidget(self.plot_hazard_button)

        
        # Esto evita espacios vacíos adicionales debajo del último botón
        left_panel.addStretch()

        # --------------- Panel Derecho (Gráficas) ---------------
        right_panel = QVBoxLayout()
        
        # Crear un layout de cuadrícula para las 4 gráficas
        grid_layout = QGridLayout()
        
        # Figuras para la pestaña Hazard con tamaño corregido
        self.hazard_figure1, self.hazard_ax1 = plt.subplots(figsize=(7, 6))
        self.hazard_canvas1 = FigureCanvas(self.hazard_figure1)
        
        self.hazard_figure2, self.hazard_ax2 = plt.subplots(figsize=(7, 6))
        self.hazard_canvas2 = FigureCanvas(self.hazard_figure2)
        
        self.Disaggregation_Tabs = QTabWidget()
        self.Disaggregation_Tabs.setStyleSheet("QTabWidget::pane { border: 0; }")
        
        # Agregar las 4 figuras a la cuadrícula
        grid_layout.addWidget(self.hazard_canvas1, 0, 0)  # Fila 0, Columna 0
        grid_layout.addWidget(self.hazard_canvas2, 0, 1)  # Fila 0, Columna 1
        grid_layout.addWidget(self.Disaggregation_Tabs, 1, 0, 1, 2)  # Fila 1, Columna 0
        
        # Agregar el layout de cuadrícula al panel derecho
        right_panel.addLayout(grid_layout)
        
        # --------------- QSplitter ---------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: transparent; }")
        
        # Panel izquierdo
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)  # Fijo en ancho
        
        # Panel derecho con las 4 gráficas
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        # Agregar los widgets al QSplitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Ajustar tamaños iniciales del splitter
        splitter.setSizes([200, 800])
        
        # Agregar el splitter al layout principal
        main_layout.addWidget(splitter)
        self.hazard_tab.setLayout(main_layout)

        
    def init_cms_tab(self):
        main_layout = QVBoxLayout()
        params_layout = QHBoxLayout()

        # Sección de Parámetros de Análisis (más compacto en Y)
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_group.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        analysis_layout = QVBoxLayout()
        analysis_group.setFixedSize(200, 200)

        source_label = QLabel("Source (Correlation):")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Crustal", "Interface", "Intraslab"])

        im_label = QLabel("IM Target:")
        self.im_target_input = QLineEdit("Sa(0p7s)")

        analysis_layout.addWidget(source_label)
        analysis_layout.addWidget(self.source_combo)
        analysis_layout.addWidget(im_label)
        analysis_layout.addWidget(self.im_target_input)
        analysis_group.setLayout(analysis_layout)

        # Grupo de Disaggregation
        disaggregation_group = QGroupBox("Disaggregation")
        disaggregation_group.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        disaggregation_layout2 = QVBoxLayout()
        disaggregation_group.setFixedSize(200, 130)  # Se ajusta para incluir el botón
        
        # Selector de Carpeta (Select Folder)
        self.folder_label = QLabel("Folder:")
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.setFixedSize(130, 30)
        self.folder_button.clicked.connect(self.select_disaggregation_folder2)


        disaggregation_layout2.addWidget(self.folder_label)
        disaggregation_layout2.addWidget(self.folder_input)
        disaggregation_layout2.addWidget(self.folder_button)
        disaggregation_group.setLayout(disaggregation_layout2)

        analysis_disaggregation_layout = QVBoxLayout()
        analysis_disaggregation_layout.addWidget(analysis_group)
        analysis_disaggregation_layout.addWidget(disaggregation_group)

        # Sección de Parámetros del Sitio
        site_group = QGroupBox("Site Parameters")
        site_group.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        site_layout = QVBoxLayout()

        vs30_label = QLabel("Vs30 (m/sec):")
        self.vs30_input = QLineEdit()

        z1_label = QLabel("Z1.0 (m):")
        self.z1_input = QLineEdit()

        z25_label = QLabel("Z2.5 (km):")
        self.z25_input = QLineEdit()
        
        vs30measured_label = QLabel("Vs30 measured:")
        self.vs30measured_combo = QComboBox()
        self.vs30measured_combo.addItems(["True", "False"])
        self.vs30measured_combo.setCurrentText("False")

        backarc_label = QLabel("Back-arc:")
        self.backarc_combo = QComboBox()
        self.backarc_combo.addItems(["True", "False"])
        self.backarc_combo.setCurrentText("False")

        site_layout.addWidget(vs30_label)
        site_layout.addWidget(self.vs30_input)
        site_layout.addWidget(z1_label)
        site_layout.addWidget(self.z1_input)
        site_layout.addWidget(z25_label)
        site_layout.addWidget(self.z25_input)
        site_layout.addWidget(vs30measured_label)
        site_layout.addWidget(self.vs30measured_combo)
        site_layout.addWidget(backarc_label)
        site_layout.addWidget(self.backarc_combo)
        site_group.setLayout(site_layout)

        # Sección de Parámetros del Terremoto
        earthquake_group = QGroupBox("Earthquake Parameters")
        earthquake_group.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        earthquake_layout = QVBoxLayout()
        
        self.earthquake_inputs = {}  # Diccionario para almacenar los inputs
        
        for label_text in ["Zhypo (km):", "Magnitude (Mw):", "Width (km):", "Rake (°):", "Dip (°):"]:
            label = QLabel(label_text)
            input_field = QLineEdit()
            earthquake_layout.addWidget(label)
            earthquake_layout.addWidget(input_field)
        
            # Guardamos la referencia con un nombre clave (sin espacios ni caracteres especiales)
            key = label_text.split(" (")[0].replace(" ", "_").lower()
            self.earthquake_inputs[key] = input_field
        
        earthquake_group.setLayout(earthquake_layout)

        # Sección de Parámetros de Distancia
        distance_group = QGroupBox("Distance Context")
        distance_group.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        distance_layout = QVBoxLayout()
        
        self.distance_inputs = {}
        
        for label_text in ["Rhypo (km):", "Rrup (km):", "Rjb (km):", "Repi (km):", "Rx (km):", "Rvolc (km):"]:
            label = QLabel(label_text)
            input_field = QLineEdit()
            distance_layout.addWidget(label)
            distance_layout.addWidget(input_field)
            key = label_text.split(" (")[0].replace(" ", "_").lower()  # Ej: "Rhypo (km):" → "rhypo"
            self.distance_inputs[key] = input_field
        
        distance_group.setLayout(distance_layout)

        # Sección de GMM
        gmpe_group = QGroupBox("GMPE Selection")
        gmpe_group.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        gmpe_layout = QVBoxLayout()
        
        self.gmpe_combo = QComboBox()
        self.gmpe_combo.addItems(gmpe_classes_label)
        
        self.weight_spinbox = QDoubleSpinBox()
        self.weight_spinbox.setRange(0.0, 1.0)
        self.weight_spinbox.setSingleStep(0.01)
        self.weight_spinbox.setValue(0.1)
        
        self.add_gmpe_button = QPushButton("Add GMPE")
        self.add_gmpe_button.clicked.connect(self.add_gmpe)
        
        self.gmpe_list = QListWidget()  # Se inicializa antes de agregar GMPEs predefinidas
        
        self.remove_gmpe_button = QPushButton("Remove Selected GMPE")
        self.remove_gmpe_button.clicked.connect(self.remove_gmpe)
        
        self.edit_weight_button = QPushButton("Edit GMPE Weight")
        self.edit_weight_button.clicked.connect(self.edit_gmpe_weight)
        
        gmpe_layout.addWidget(QLabel("Select GMPE:"))
        gmpe_layout.addWidget(self.gmpe_combo)
        gmpe_layout.addWidget(QLabel("Weight:"))
        gmpe_layout.addWidget(self.weight_spinbox)
        gmpe_layout.addWidget(self.add_gmpe_button)
        gmpe_layout.addWidget(QLabel("Added GMPEs (with weights):"))
        gmpe_layout.addWidget(self.gmpe_list)
        gmpe_layout.addWidget(self.remove_gmpe_button)
        gmpe_layout.addWidget(self.edit_weight_button)
        
        gmpe_group.setLayout(gmpe_layout)
        
        # 🔹 Llamar a la función que carga GMPEs predefinidas
        self.initialize_default_gmpes()

        # Botón Calculate (pequeño)
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.setStyleSheet("font-weight: bold;")
        self.calculate_button.setFixedSize(100, 30)
        
        # Conectar el botón a la función de cálculo del CMS
        self.calculate_button.clicked.connect(self.calculate_cms)
        
        # Centrar el botón
        centered_layout = QHBoxLayout()
        centered_layout.addStretch()
        centered_layout.addWidget(self.calculate_button)
        centered_layout.addStretch()

        params_layout.addLayout(analysis_disaggregation_layout)
        params_layout.addWidget(site_group)
        params_layout.addWidget(earthquake_group)
        params_layout.addWidget(distance_group)
        params_layout.addWidget(gmpe_group)

        main_layout.addLayout(params_layout, stretch=1)
        main_layout.addLayout(centered_layout, stretch=0)
        main_layout.addStretch()


        plots_layout = QHBoxLayout()
        self.cms_tabs = QTabWidget()
        self.cms_tabs.setStyleSheet("QTabWidget::pane { border: 0; }") 
        self.cms_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expande
        self.cms_tabs.setMinimumHeight(400)
        plots_layout.addWidget(self.cms_tabs, stretch=2)
        
        self.figure2, self.ax2 = plt.subplots(figsize=(9, 7))
        self.canvas2 = FigureCanvas(self.figure2)
        plots_layout.addWidget(self.canvas2)
        
        main_layout.addLayout(plots_layout)
        self.cms_tab.setLayout(main_layout)
        
    def init_css_tab(self):
        main_layout = QHBoxLayout()

        # --------------- Panel Izquierdo ---------------
        left_panel = QVBoxLayout()

        # -------------- Grupo de CSS Calculation --------------
        css_group = QGroupBox("CSS Calculation")
        css_group.setStyleSheet("QGroupBox { font-weight: bold; text-decoration: underline; }")
        css_group.setSizePolicy(230, 160)
        css_layout = QVBoxLayout()

        # Creando los checkboxes
        self.default_checkbox = QCheckBox("Default")
        self.personalized_checkbox = QCheckBox("Personalized")

        # Agrupar los checkboxes para que solo uno pueda seleccionarse a la vez
        self.css_check_group = QButtonGroup(self)
        self.css_check_group.addButton(self.default_checkbox)
        self.css_check_group.addButton(self.personalized_checkbox)
        
        self.default_checkbox.toggled.connect(self.toggle_input_fields)
        self.personalized_checkbox.toggled.connect(self.toggle_input_fields)

        # Agregar elementos al layout del grupo
        css_layout.addWidget(self.default_checkbox)
        css_layout.addWidget(self.personalized_checkbox)
        css_group.setLayout(css_layout)
        
        # -------------- Campos de entrada (Mmin, Mmax, Rmin, Rmax) --------------
        self.mmin_input = QLineEdit()
        self.mmax_input = QLineEdit()
        self.rmin_input = QLineEdit()
        self.rmax_input = QLineEdit()
        
        self.mmin_input.setFixedWidth(60)
        self.mmax_input.setFixedWidth(60)
        self.rmin_input.setFixedWidth(60)
        self.rmax_input.setFixedWidth(60)

        # Alinear los campos en un layout horizontal
        m_layout = QHBoxLayout()
        m_layout.addWidget(QLabel("Mmin:"))
        m_layout.addWidget(self.mmin_input)
        m_layout.addWidget(QLabel("Mmax:"))
        m_layout.addWidget(self.mmax_input)

        r_layout = QHBoxLayout()
        r_layout.addWidget(QLabel("Rmin:"))
        r_layout.addWidget(self.rmin_input)
        r_layout.addWidget(QLabel("Rmax:"))
        r_layout.addWidget(self.rmax_input)

        # Agregar los campos al layout del grupo
        css_layout.addLayout(m_layout)
        css_layout.addLayout(r_layout)

        css_group.setLayout(css_layout)

        # -------------- Botón para ejecutar CSS Calculation --------------
        self.css_button = QPushButton("Run CSS Calculation")
        self.css_button.setFixedSize(130, 30)
        self.css_button.clicked.connect(self.run_scenario_spectra) 
        self.css_button.clicked.connect(self.plot_css_calculation)
        
        # Agregar elementos al panel izquierdo
        left_panel.addWidget(css_group)
        left_panel.addWidget(self.css_button)
        left_panel.addStretch()

        # --------------- Panel Derecho (Gráficas) ---------------
        right_panel = QVBoxLayout()
        
        # Crear un layout de cuadrícula para las 4 gráficas
        grid_layout = QGridLayout()
        
        # Figuras para la pestaña CSS
        self.css_figure1, self.css_ax1 = plt.subplots(figsize=(7, 6))
        self.css_canvas1 = FigureCanvas(self.css_figure1)
        
        self.css_figure2, self.css_ax2 = plt.subplots(figsize=(7, 6))
        self.css_canvas2 = FigureCanvas(self.css_figure2)
        
        self.css_figure3, self.css_ax3 = plt.subplots(figsize=(7, 6))
        self.css_canvas3 = FigureCanvas(self.css_figure3)
        
        self.css_figure4, self.css_ax4 = plt.subplots(figsize=(7, 6))
        self.css_canvas4 = FigureCanvas(self.css_figure4)
        
        # Agregar las 4 figuras a la cuadrícula
        grid_layout.addWidget(self.css_canvas1, 0, 0)  # Fila 0, Columna 0
        grid_layout.addWidget(self.css_canvas2, 0, 1)  # Fila 0, Columna 1
        grid_layout.addWidget(self.css_canvas3, 1, 0)  # Fila 1, Columna 0
        grid_layout.addWidget(self.css_canvas4, 1, 1)  # Fila 1, Columna 1
        
        # Agregar el layout de cuadrícula al panel derecho
        right_panel.addLayout(grid_layout)

        # --------------- QSplitter ---------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: transparent; }")
        
        # Panel izquierdo
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)  # Fijo en ancho
        
        # Panel derecho con las 4 gráficas
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        # Agregar los widgets al QSplitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Ajustar tamaños iniciales del splitter
        splitter.setSizes([200, 800])
        
        # Agregar el splitter al layout principal
        main_layout.addWidget(splitter)
        self.css_tab.setLayout(main_layout)

    def run_scenario_spectra(self):
        """Ejecuta ScenarioSpectra_2017.exe dentro de 'Selection_Sa(1s)' y escribe 'Main_CSS.txt' seguido de Enter."""
        
        imt_tag = self.im_target_input.text()
        folder_sele = 'Selection_%s'%(imt_tag)
        
        try:
            # Ruta del ejecutable y el archivo Main_CSS.txt
            selection_folder = folder_sele
            exe_path = os.path.join(selection_folder, "ScenarioSpectra_2017.exe")
            input_file = os.path.join(selection_folder, "Main_CSS.txt")

            # Verificar si el ejecutable y el archivo existen
            if not os.path.exists(exe_path):
                QMessageBox.critical(self, "Error", f"No se encontró '{exe_path}'")
                return
            
            if not os.path.exists(input_file):
                QMessageBox.critical(self, "Error", f"No se encontró '{input_file}'")
                return

            # Ejecutar el programa dentro de la carpeta correcta
            process = subprocess.Popen(
                exe_path,
                cwd=selection_folder,  # 💡 Cambia el directorio de trabajo al de la carpeta
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1 
            )

            # Escribir en la entrada estándar
            process.stdin.write("Main_CSS.txt\n")
            process.stdin.write("\n")
            process.stdin.write("\n") # Segundo Enter
            process.stdin.flush()  

            # Leer la salida en tiempo real y detectar "monte carlo"
            monte_carlo_detected = False
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Mostrar en consola
    
                    # Si se detecta "monte carlo", enviar el último ENTER
                    if "monte carlo" in output.lower():
                        time.sleep(0.5)  # Pequeña espera antes de enviar ENTER
                        process.stdin.write("\n")
                        process.stdin.flush()
                        monte_carlo_detected = True
    
            # Leer posibles errores
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"❌ Error: {stderr_output}")
    
            if monte_carlo_detected:
                QMessageBox.information(self, "Success", "CSS Calculation Completed Successfully!")
            else:
                QMessageBox.warning(self, "Warning", "The 'monte carlo' step was not detected. Check the process output.")
    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al ejecutar el programa:\n{e}")
        
    def plot_css_calculation(self):
        
        tr_text = self.tr_input.text().strip()
        self.tr_list = [int(x.strip()) for x in tr_text.split(",") if x.strip()]
        
        imt_tag = self.im_target_input.text()
        folder_sele = 'Selection_%s'%(imt_tag)
        
        filename_CMS = 'CSS_%s'%(imt_tag).replace('(','_')
        filename_CMS = filename_CMS.replace(')','')
        
        file_HazH = filename_CMS+"_HazH.out3"
        file_MR = filename_CMS+"_CS.out1"
    
        try:
            # Leer el archivo
            with open(os.path.join(folder_sele, file_HazH), 'r') as file:
                lines = file.readlines()
            
            # Procesar los datos
            data = [line.split() for line in lines[2:]]
            df_HazH = pd.DataFrame(data[1:], columns=data[0])
    
            # Convertir columnas a float
            for col in ['Period', 'iLevel', 'Haz_Initial', 'Haz_Final', 'Haz_Target', 'Sa(g)']:
                df_HazH[col] = df_HazH[col].astype(float)
    
            # Limpiar las figuras antes de dibujar
            self.css_ax3.clear()  
    
            periodos = [0.2, 1.0, 1.5]
            colormap = plt.get_cmap('viridis')(np.linspace(0, 1, len(periodos)))
            
            for idx, periodo in enumerate(periodos):
                df_HazH_periodo = df_HazH[df_HazH['Period'] == periodo]
                color = colormap[idx]
                self.css_ax3.plot(df_HazH_periodo['Sa(g)'], df_HazH_periodo['Haz_Target'], color=color, lw=2.0, label=f'HZCurve: {periodo}s')
                self.css_ax3.plot(df_HazH_periodo['Sa(g)'], df_HazH_periodo['Haz_Final'], color=color, lw=1.5, ls='--')
            
            self.css_ax3.set_title("Hazard Recovery (Initial vs Final)", fontdict={'fontsize': 12})
            self.css_ax3.set_xlabel('Sa [g]', fontdict={'fontsize': 12})
            self.css_ax3.set_ylabel('Annual rate of being exceeded', fontdict={'fontsize': 12})
            self.css_ax3.set_xscale('log')
            self.css_ax3.set_yscale('log')
            self.css_ax3.set_ylim([1 / max(self.tr_list), 1 / min(self.tr_list)]) 
            self.css_ax3.grid(True, which="both", linestyle="--", linewidth=0.5)
            self.css_ax3.legend(scatterpoints=1, frameon=True, loc='upper right', handletextpad=0.2, columnspacing=0.20,
           ncol=1, fancybox=False, shadow=False, prop={'size': 10}, borderpad=0.2)
            self.css_canvas3.draw()
            
            with open(os.path.join(folder_sele, file_MR), 'r') as file:
                lines = file.readlines()
            data = [line.split() for line in lines[5:]]  
            df_rates = pd.DataFrame(data[1:], columns=data[0])
            df_rates['Mag'] = df_rates['Mag'].astype(float)
            df_rates['Rrup'] = df_rates['Rrup'].astype(float)
            df_rates['Rate'] = df_rates['Rate'].astype(float)
            df_rates['Rate_Initial'] = df_rates['Rate_Initial'].astype(float)
            
            self.css_ax2.scatter(df_rates.Rrup, df_rates.Mag, alpha=0.65, edgecolor = 'k', linewidths = 0.5, color = 'darkblue', s= 50)
            self.css_ax2.set_title("Magnirude vs Rupture Distance", fontdict={'fontsize': 12})
            self.css_ax2.set_xlabel('Rupture Distance [km]', fontdict={'fontsize': 12})
            self.css_ax2.set_ylabel('Magnitude, $M_w$', fontdict={'fontsize': 12})
            self.css_ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
            self.css_canvas2.draw()
            
            self.css_ax1.semilogy(df_rates.Rate, '.k', label = 'Final rate', markersize = 7.5)
            self.css_ax1.semilogy(df_rates.Rate_Initial, '.', color = 'darkred', label = 'Initial rate', markersize = 7.5)
            self.css_ax1.set_title("Initial Rate vs Final Rate", fontdict={'fontsize': 12})
            self.css_ax1.set_xlabel('Record Index', fontdict={'fontsize': 12})
            self.css_ax1.set_ylabel('Rate', fontdict={'fontsize': 12})
            self.css_ax1.legend(loc="best", bbox_to_anchor=(1,1),prop={'size': 10});
            self.css_ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
            self.css_canvas1.draw()
            
            
        except Exception as e:
            print(f"❌ Error while Ploting CSS: {e}")

            
    def toggle_input_fields(self):
        """Habilita o deshabilita los campos de Mmin, Mmax, Rmin, Rmax según el checkbox seleccionado."""
        enabled = self.personalized_checkbox.isChecked()

        # Bloquear/desbloquear los campos según la opción seleccionada
        for field in [self.mmin_input, self.mmax_input, self.rmin_input, self.rmax_input]:
            field.setReadOnly(not enabled)
            field.setStyleSheet("background-color: lightgray;" if not enabled else "")
            
    def initialize_default_gmpes(self):
        """Inicializa la lista con GMPEs predefinidas y sus pesos"""
        default_gmpes = {
            "AbrahamsonEtAl2014": 0.22,
            "Idriss2014": 0.39,
            "CauzziEtAl2014": 0.39
        }
    
        self.gmpe_list.clear()   
        for gmpe, weight in default_gmpes.items():
            self.gmpe_weights[gmpe] = weight
            self.gmpe_list.addItem(f"{gmpe} - Weight: {weight:.2f}")
    
    def add_gmpe(self):
        selected_gmpe = self.gmpe_combo.currentText()
        weight = self.weight_spinbox.value()
        if selected_gmpe in self.gmpe_weights:
            QMessageBox.warning(self, "Error", "This GMPE is already added. Edit its weight instead.")
            return
        self.gmpe_weights[selected_gmpe] = weight
        self.gmpe_list.addItem(f"{selected_gmpe} - Weight: {weight:.2f}")

    def remove_gmpe(self):
        selected_items = self.gmpe_list.selectedItems()
        for item in selected_items:
            gmpe_name = item.text().split(" - Weight: ")[0]  # Extrae el nombre de la GMPE
            if gmpe_name in self.gmpe_weights:
                del self.gmpe_weights[gmpe_name]
                self.gmpe_list.takeItem(self.gmpe_list.row(item))
            else:
                QMessageBox.warning(self, "Error", f"GMPE '{gmpe_name}' not found. Check if it was added correctly.")

    def show_gmpe_weights(self):
        """ Muestra en un mensaje emergente los pesos de las GMPEs seleccionadas """
        if not self.gmpe_weights:
            QMessageBox.information(self, "GMPE Weights", "No GMPEs have been added.") 
            
    def edit_gmpe_weight(self):
        """ Permite editar el peso de una GMPE seleccionada """
        selected_items = self.gmpe_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Edit GMPE", "Please select a GMPE to edit.")
            return
        
        selected_item = selected_items[0]  # Solo permite editar una GMPE a la vez
        gmpe_name = selected_item.text().split(" - Weight: ")[0]  # Extrae el nombre
        
        # Pedir nuevo peso al usuario
        new_weight, ok = QInputDialog.getDouble(self, "Edit Weight", 
                                                f"New weight for {gmpe_name}:",
                                                min=0.0, max=1.0, decimals=2)
        if ok:
            self.gmpe_weights[gmpe_name] = new_weight  # Actualizar diccionario
            selected_item.setText(f"{gmpe_name} - Weight: {new_weight:.2f}")

    def select_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file:
            self.file_input.setText(file)
            
    def select_disaggregation_folder(self, target_input):
        """Abre un cuadro de diálogo para seleccionar una carpeta y actualiza el input."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", "")
    
        if folder:  # Si el usuario selecciona una carpeta válida
            target_input.setText(folder)
        else:
            target_input.setText('')
            print("No folder selected.")

    def select_disaggregation_folder2(self):
        """Abre un cuadro de diálogo para seleccionar una carpeta y actualiza el input."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", "")
    
        if folder:  # Si el usuario selecciona una carpeta válida
            self.folder_input.setText(folder)
            self.imput_parameters_context(False)
        else:
            self.folder_input.setText('')
            self.imput_parameters_context(True) 

    def imput_parameters_context(self, enabled):
        """Activa o desactiva los campos de Distance Context"""
        for  key, input_field in self.distance_inputs.items():
            if key == "rvolc":
                continue  # No bloquear Rvolc, siempre debe estar activo
            input_field.setReadOnly(not enabled)  
            input_field.setStyleSheet("background-color: lightgray;" if not enabled else "")
            
        for  key, input_field in self.earthquake_inputs.items():
            if key == "magnitude":
                input_field.setReadOnly(not enabled)  
                input_field.setStyleSheet("background-color: lightgray;" if not enabled else "")
            else: 
                 continue   
        self.vs30_input.setReadOnly(not enabled)
        self.vs30_input.setStyleSheet("background-color: lightgray;" if not enabled else "")
                
    def load_and_plot(self):
        """
        Grafica curvas de amenaza según la base de datos elegida:
          - SGC: CSV con skiprows=2, 'stat'=='mean', columnas 'iml_...'
          - USGS/Others: Formato 3 líneas por IMT
        Guarda en self.HzCurvess_Dict para que plot_uhs funcione.
        """
        file_path = self.file_input.text()
        city = self.ciudad_combo.currentText().strip()
        
        if city != 'Other' and self.radio_sgc.isChecked():
            self.imput_parameters_context(False)
            
        if self.radio_usgs.isChecked():
            self.imput_parameters_context(True)        
        
        if not file_path and city == 'Other' and self.radio_sgc.isChecked():
            self.show_warning("Please select a file.")
            return
        if not file_path and self.radio_usgs.isChecked():
            self.show_warning("Please select a file.")
            return
        if not file_path and self.radio_others.isChecked():
            self.show_warning("Please select a file.")
            return          
        self.hazard_ax1.clear()
        self.hazard_figure1.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.85)

        try:        
            # ----------------- SGC -----------------
            if self.radio_sgc.isChecked():
                folder = os.path.join('Colombia', city)
                List_Desag_Source = os.listdir(folder)
                hcurves_files = [file for file in List_Desag_Source if file.startswith("hcurves")][0]
                file_path = os.path.join(folder,hcurves_files)
                df = pd.read_csv(file_path, skiprows=2)
                df_filtrado = df[df['stat'] == 'mean'].reset_index(drop=True)

                # Columnas de la 4 en adelante => GM (quitando "iml_")
                cols = df_filtrado.columns[4:]
                gm_values = [float(c.replace('iml_', '')) for c in cols]

                # Construimos un diccionario hazard_data en el mismo formato
                hazard_data = {}
                for i in range(df_filtrado.shape[0]):
                    imt_name = '0.01' if df_filtrado['imt'].iloc[i] == 'PGA' \
                               else df_filtrado['imt'].iloc[i].replace('SA(','').replace(')','')
                    aep_vals = df_filtrado.iloc[i, 4:].values
                    df_temp = pd.DataFrame({
                        'GM': gm_values,
                        'AEP': aep_vals
                    })
                    hazard_data[imt_name] = df_temp

                self.HzCurvess_Dict = hazard_data

                # Graficamos
                cmap = cm.get_cmap("viridis", len(hazard_data))
                norm = mcolors.Normalize(vmin=0, vmax=len(hazard_data) - 1)

                for i, (imt, df_haz) in enumerate(hazard_data.items()):
                    self.hazard_ax1.plot(
                        df_haz['GM'], df_haz['AEP'],
                        marker='o', linestyle='-',
                        label=imt,
                        color=cmap(norm(i))
                    )
                self.hazard_ax1.set_title("Hazard Curves (SGC)", fontsize=14)

            # ----------------- USGS / Others -----------------
            else:
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                imts = [line.strip().split(' s ')[0] for line in lines[::3]]
                gms  = [list(map(float, line.strip().split(',')[1:])) for line in lines[1::3]]
                aeps = [list(map(float, line.strip().split(',')[1:])) for line in lines[2::3]]

                hazard_data = {}
                for i, imt in enumerate(imts):
                    hazard_data[imt] = pd.DataFrame({'GM': gms[i], 'AEP': aeps[i]})

                self.HzCurvess_Dict = hazard_data

                cmap = cm.get_cmap("viridis", len(hazard_data))
                norm = mcolors.Normalize(vmin=0, vmax=len(hazard_data) - 1)

                for i, (imt, df_haz) in enumerate(hazard_data.items()):
                    label_text = "PGA" if imt == "Peak Ground Acceleration" else imt
                    self.hazard_ax1.plot(
                        df_haz['GM'], df_haz['AEP'],
                        marker='o', linestyle='-',
                        label=label_text,
                        color=cmap(norm(i))
                    )
                self.hazard_ax1.set_title("Hazard Curves (USGS)", fontsize=14)

            self.hazard_ax1.set_xlabel("Ground Motion (g)", fontsize=12)
            self.hazard_ax1.set_ylabel("Annual Frequency of Exceedance", fontsize=12)
            self.hazard_ax1.set_xscale("log")
            self.hazard_ax1.set_yscale("log")    
            self.hazard_ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
            self.hazard_ax1.legend(
                fontsize=10,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.1),
                title="IMTs",
                ncol=1,
                borderaxespad=1
            )
            self.hazard_figure1.tight_layout()
            self.hazard_canvas1.draw()

        except Exception as e:
            self.show_error(f"Error while processing the file: {e}")
    
    def plot_disaggregation_results(self):
        """Genera gráficos sincronizados en un mismo QTabWidget"""
    
        try:
            # ----------------- SGC -----------------
            if self.radio_sgc.isChecked():
                city = self.ciudad_combo.currentText().strip()
                folder = os.path.join('Colombia', city)
                List_Desag_Source = os.listdir(folder)
    
                # Cargar archivos
                magdist_file = [file for file in List_Desag_Source if file.startswith("MagDist")][0]
                source_file = [file for file in List_Desag_Source if file.startswith("Trt")][0]
    
                magdist_path = os.path.join(folder, magdist_file)
                source_path = os.path.join(folder, source_file)
    
                Desag_MagDist = pd.read_csv(magdist_path, skiprows=1)
                Desag_Ambientes = pd.read_csv(source_path, skiprows=1)
    
                Desag_MagDist['TR'] = np.round(Desag_MagDist.poe**-1, 0)                               
                TRint = 475
                self.Disaggregation_Tabs.clear()  # Limpiar todas las pestañas previas
                plt.close('all')  # Cerrar figuras abiertas
                
                tr_layout = QHBoxLayout()
                tr_label = QLabel("Select Return Period (TR):")
                self.tr_selector = QComboBox()
                self.tr_selector.addItems(["31", "100", "225", "475", "975"])
                
                tr_layout.addWidget(tr_label)
                tr_layout.addWidget(self.tr_selector)
                tr_layout.addStretch()
   
                imt_list = ['PGA', 'SA(0.2)', 'SA(0.3)','SA(0.5)', 'SA(0.7)', 'SA(1.0)', 'SA(1.5)', 'SA(2.0)', 'SA(2.5)', 'SA(3.0)', 'SA(4.0)']
    
                MagCols = Desag_MagDist.columns[5:-1]
                Mags = [np.average([float(el.split('_')[1].split('-')[0]), float(el.split('_')[1].split('-')[1])]) for el in MagCols]
    
                Desag_MagDist['Rprom'] = [np.average([float(r.split('-')[0]), float(r.split('-')[1])]) for r in Desag_MagDist['dist'].values]
    
                # Extraer valores de TRs desde las columnas del archivo de fuentes sísmicas
                TRs = [np.round(float(el.split('_')[1])**-1, 0) for el in Desag_Ambientes.columns[4:]]
                Col_TR = 4 + TRs.index(TRint)
    
                for imt_int in imt_list:
                    SaInt_TRint = Desag_MagDist[(Desag_MagDist.imt == imt_int) & (Desag_MagDist.TR == TRint)]
                    SaInt = Desag_Ambientes[Desag_Ambientes.imt == imt_int].copy()
                    SaInt = SaInt.sort_values(by='trt')
    
                    # --------- Generar Figura 3D de Magnitud-Distancia ----------
                    fig1 = plt.figure(figsize=(6, 6))
                    ax1 = fig1.add_subplot(111, projection='3d')
    
                    Contrib = SaInt_TRint[MagCols].values
                    Contrib = 100 * Contrib / np.sum(Contrib)
    
                    bar_width = (max(SaInt_TRint.Rprom) - min(SaInt_TRint.Rprom)) / 40
                    bar_depth = (max(Mags) - min(Mags)) / 40
                    
    
                    sorted_indices = np.argsort(SaInt_TRint.Rprom.values)
                    Mmean = 0
                    Rmean = 0
                    color = '#440154'
                    for i_R in sorted_indices:
                        for i_Mag in range(len(Mags)):
                            R_int = SaInt_TRint.Rprom.iloc[i_R]
                            Rmean += (R_int*Contrib[i_R, i_Mag])/100
                            Mag_int = Mags[i_Mag]
                            Mmean += (Mag_int*Contrib[i_R, i_Mag])/100
                            C_int = Contrib[i_R, i_Mag]
                            if C_int > 0:
                                ax1.bar3d(R_int, Mag_int, 0, bar_width, bar_depth, C_int, color=color, alpha=1.0, linewidth=0.2, edgecolor='grey')
                    Mmean = round(Mmean, 1)
                    Rmean = round(Rmean, 1)
                    ax1.view_init(elev=35, azim=-60)  
                    ax1.set_xlabel('Closest distance,rRup (km)', fontsize=8.5)
                    ax1.set_title(f'Magnitude - Distance TR {TRint} [{imt_int}]\n'
                                 rf'$\mathbf{{M_{{Mean}}}}$: {Mmean} - $\mathbf{{R_{{Mean}}}}$: {Rmean}', fontsize=12)
                    ax1.set_ylabel('Magnitude (Mw)', fontsize=8.5)
                    ax1.set_zlabel('% Contribution to hazard')
                    ax1.set_xlim([0, max(SaInt_TRint.Rprom)])
                    ax1.set_ylim([5, 9])
                    ax1.set_zlim([0, 20])
                    ax1.xaxis.pane.fill = False
                    ax1.yaxis.pane.fill = False
                    ax1.zaxis.pane.fill = False
                    ax1.view_init(elev=35, azim=-60)
                    fig1.tight_layout()
    
                    # --------- Generar Gráfico de Barras para Contribución de Fuentes ----------
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    colors = plt.cm.viridis(np.linspace(0, 1, 4))
                    Contrib = np.round(100 * SaInt.iloc[:, Col_TR] / np.sum(SaInt.iloc[:, Col_TR]), 1)
    
                    Trt_labels = ['Crustal', 'Nido', 'Interface', 'Intraslab']
                    temp_DF = pd.DataFrame({'Trt': Trt_labels, 'Contrib': list(Contrib)})
    
                    bars = ax2.bar(temp_DF['Trt'], temp_DF['Contrib'], color=colors, zorder = 10000)
    
                    for bar in bars:
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}%', ha='center', va='bottom', fontsize=8)
    
                    ax2.set_xticks(range(len(Trt_labels)))
                    ax2.set_xticklabels(Trt_labels, rotation=90, fontsize=12)
                    ax2.set_ylim([0, max(temp_DF['Contrib']) + 5])
                    ax2.grid(visible=True, which='both', axis='both', alpha=0.15)
                    ax2.set_title(f"Source Contribution TR {TRint} [{imt_int}]", fontsize=12)
                    ax2.set_ylabel("% Contribution to Hazard", fontsize=10)
                    fig2.tight_layout()
    
                    # --------- Crear Tab con Ambas Figuras ----------
                    tab = QWidget()
                    tab_layout = QHBoxLayout()
    
                    canvas1 = FigureCanvas(fig1)
                    canvas2 = FigureCanvas(fig2)
    
                    tab_layout.addWidget(canvas1)
                    tab_layout.addWidget(canvas2)
                    tab.setLayout(tab_layout)
    
                    self.Disaggregation_Tabs.addTab(tab, f"{imt_int}")
                    plt.close(fig1)
                    plt.close(fig2)
                    
            else:
                self.Disaggregation_Tabs.clear()
                dissaggregation_folder = self.folder_input_hazard.text()
                List_Desag_Source = os.listdir(dissaggregation_folder)
                for dissaggregation_file in List_Desag_Source:
                    # Leer el archivo línea por línea
                    with open(os.path.join(dissaggregation_folder, dissaggregation_file), "r", encoding="utf-8") as file:
                        lines = file.readlines()
                    
                    # Encontrar la línea de inicio (cabecera de la tabla) y la línea de fin
                    start_line = None
                    end_line = None
                    
                    for i, line in enumerate(lines):
                        if "Distance (km), Magnitude (Mw)" in line:
                            start_line = i
                        elif "Disaggregation Contributions" in line:
                            end_line = i
                            break
                    # Extraer solo las líneas de la tabla y cargar en DataFrame
                    if start_line is not None and end_line is not None:
                        table_lines = lines[start_line:end_line]
                        
                        # Convertir la lista en un formato legible por pandas
                        table_data = StringIO("".join(table_lines))
                        df = pd.read_csv(table_data, header=0, encoding="utf-8", engine="python")
                        df.columns = df.columns.str.strip() 
                        
                    else:
                        print("No se pudo encontrar el inicio o fin de la tabla en el archivo.") 
                        
                    Rprom = df["Distance (km)"].values  
                    Mags = df["Magnitude (Mw)"].values  
                    Contrib = df["ε total"].values
                    
                    total_contrib = np.sum(Contrib)
                
                    if total_contrib > 0:
                        Mmean = np.sum(Mags * Contrib) / total_contrib
                        Rmean = np.sum(Rprom * Contrib) / total_contrib
                    else:
                        Mmean, Rmean = 0, 0 
                    
                    Mmean = round(Mmean, 1)
                    Rmean = round(Rmean, 1)
                    
                    bar_width = (max(Rprom) - min(Rprom)) / 40
                    bar_depth = (max(Mags) - min(Mags)) / 40
                    
                    # Crear figura y ejes 3D
                    fig1 = plt.figure(figsize=(6, 6))
                    ax1 = fig1.add_subplot(111, projection='3d')
                    
                    # Graficar barras 3D en una sola llamada sin iteraciones anidadas
                    ax1.bar3d(Rprom, Mags, np.zeros_like(Contrib), bar_width, bar_depth, Contrib,
                             color='#440154', alpha=1.0, linewidth=0.2, edgecolor='grey')
                    
                    # Configuración de la vista y etiquetas
                    ax1.view_init(elev=35, azim=-60)
                    ax1.set_xlabel('Closest distance, rRup (km)', fontsize=8.5)
                    ax1.set_ylabel('Magnitude (Mw)', fontsize=8.5)
                    ax1.set_zlabel('% Contribution to hazard', fontsize=10)
                    ax1.set_title(f'Magnitude - Distance\n'
                            rf'$\mathbf{{M_{{Mean}}}}$: {Mmean} - $\mathbf{{R_{{Mean}}}}$: {Rmean}', fontsize=12)
                    
                    # Ajustes de límites
                    ax1.set_xlim([0, max(Rprom)+20])
                    ax1.set_ylim([4.5, 10]) 
                    ax1.set_zlim([0, 10]) 
                    
                    # Ocultar los paneles de fondo
                    ax1.xaxis.pane.fill = False
                    ax1.yaxis.pane.fill = False
                    ax1.zaxis.pane.fill = False
                    
                    # Ajustar el diseño
                    fig1.tight_layout() 
                    
                    # --------- Generar Gráfico de Barras para Contribución de Fuentes ----------
                    # Leer el archivo línea por línea
                    with open(os.path.join(dissaggregation_folder, dissaggregation_file), "r", encoding="utf-8") as file:
                        lines = file.readlines()
                    
                    # Diccionario para almacenar la contribución total por Source Type
                    source_contributions = {}
                    
                    # Variables auxiliares
                    current_source = None
                    inside_table = False
                    table_lines = []
                    header_line = None
                    
                    # Recorrer las líneas del archivo
                    for i, line in enumerate(lines):
                        line = line.strip()
                    
                        # Detectar nueva fuente
                        if line.startswith("** Disaggregation Component: Source Type:"):
                            current_source = line.split(":")[-1].strip()  # Extraer el nombre del Source Type
                            inside_table = False  # Resetear estado de lectura de tabla
                            table_lines = []  # Resetear contenido de la tabla
                            header_line = None  # Resetear encabezado
                    
                        # Detectar inicio de la tabla "Disaggregation Contributions"
                        elif line.startswith("Disaggregation Contributions"):
                            inside_table = True
                            header_line = i + 1  # La segunda línea después de esta es la cabecera
                    
                        # Capturar la cabecera de la tabla y comenzar a guardar datos
                        elif inside_table:
                            if header_line and i == header_line:
                                header_text = line  # La línea siguiente es la cabecera
                            elif line == "":
                                inside_table = False  # Línea vacía marca el fin de la tabla
                            else:
                                table_lines.append(line)  # Almacenar líneas de datos
                    
                        # Procesar la tabla cuando termine
                        if not inside_table and table_lines and header_line is not None:
                            # Convertir la tabla en un DataFrame
                            table_text = "\n".join([header_text] + table_lines)  # Incluir cabecera
                            table_data = StringIO(table_text)
                            # Leer la tabla con separador de comas
                            df_table = pd.read_csv(table_data, sep=",", skipinitialspace=True)
                    
                            # Eliminar columnas con solo NaN (vacías en la tabla)
                            df_table = df_table.dropna(axis=1, how="all")
                    
                            # Renombrar columnas para asegurar compatibilidad
                            df_table.columns = df_table.columns.str.strip()
                    
                            # Filtrar solo los valores donde "Type" sea "SET"
                            if "Type" in df_table.columns and "%" in df_table.columns:
                                df_filtered = df_table[df_table["Type"].str.strip().str.upper() == "SET"]
                    
                                # Sumar la columna de "% Contribution"
                                total_contribution = df_filtered["%"].sum()
                    
                                # Guardar la contribución en el diccionario
                                source_contributions[current_source] = total_contribution
                    
                            # Resetear la tabla para la siguiente fuente
                            table_lines = []
                            header_line = None
                    
                    # Convertir resultados en DataFrame para graficar
                    df_contributions = pd.DataFrame(source_contributions.items(), columns=["Source Type", "Total Contribution"])
                    df_contributions = df_contributions.dropna(subset=["Source Type"])
                    df_contributions = df_contributions[df_contributions["Source Type"].str.strip() != ""]
                    df_contributions["Source Type"] = df_contributions["Source Type"].str.replace(r"\*", "", regex=True)                                                          
                    
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    colors = plt.cm.viridis(np.linspace(0, 1, 4))
                    
                    Trt_labels = df_contributions['Source Type']
                    
                    bars = ax2.bar(df_contributions['Source Type'], df_contributions['Total Contribution'], color=colors, zorder = 10000)
                    for bar in bars:
                                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height(): .1f}%', ha='center', va='bottom', fontsize=8)
                    ax2.set_xticks(range(len(Trt_labels)))
                    ax2.set_xticklabels(Trt_labels, rotation=90, fontsize=12)
                    ax2.set_ylim([0, max(df_contributions['Total Contribution']) + 5])
                    ax2.grid(visible=True, which='both', axis='both', alpha=0.15)
                    ax2.set_title("Source Contribution", fontsize=12)
                    ax2.set_ylabel("% Contribution to Hazard", fontsize=10)
                    fig2.tight_layout()

                    # --------- Crear Tab con Ambas Figuras ----------
                    tab = QWidget()
                    tab_layout = QHBoxLayout()
    
                    canvas1 = FigureCanvas(fig1)
                    canvas2 = FigureCanvas(fig2)
    
                    tab_layout.addWidget(canvas1)
                    tab_layout.addWidget(canvas2)
                    tab.setLayout(tab_layout)
    
                    self.Disaggregation_Tabs.addTab(tab, dissaggregation_file.rsplit('.', 1)[0])
                    plt.close(fig1)
                
        except Exception as e:
            self.show_error(f"Error while processing the file: {e}")

    
    def plot_uhs(self):
        """
        Lógica de gráfica UHS usando self.HzCurvess_Dict y 
        los TR que el usuario introduzca en self.tr_input.
        """
        if not self.HzCurvess_Dict:
            self.show_warning("Please load hazard curves first.")
            return

        # Leer la lista de TR que escribió el usuario:
        tr_text = self.tr_input.text().strip()
        if not tr_text:
            self.show_warning("Please enter at least one Return Period.")
            return

        # Convertimos "475, 975" en [475, 975], etc.
        try:
            tr_list = [int(x.strip()) for x in tr_text.split(",") if x.strip()]
        except ValueError:
            self.show_warning("Invalid Return Periods. Please enter comma-separated integers.")
            return

        self.hazard_ax2.clear()
        self.hazard_figure2.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.85)

        try:
            hazard_data = self.HzCurvess_Dict

            # Eliminamos la IMT "Peak Ground Acceleration" si no la queremos
            # en el UHS, o puedes dejarla.
            Keys_imt = [imt for imt in hazard_data.keys()
                        if imt != "Peak Ground Acceleration"]

            # Intentamos convertir las IMTs a float (si representan períodos).
            # Para SGC, podrían ser strings con periodos; para USGS, podrían ser "0.2", "1.0", etc.
            # Ajusta según tu convención real.
            try:
                Pers_UHS = [float(k) for k in Keys_imt]
            except ValueError:
                # Si no se puede convertir, solo enumeramos
                Pers_UHS = range(len(Keys_imt))

            cmap = cm.get_cmap("viridis", len(tr_list))
            norm = mcolors.Normalize(vmin=0, vmax=len(tr_list) - 1)

            for i, TR_int in enumerate(tr_list):
                SaUHS = []
                for K in Keys_imt:
                    df_curv = hazard_data[K]

                    # Interp en escala log10
                    f_HazC = interpolate.interp1d(
                        np.log10(df_curv['AEP'].astype(float)),
                        np.log10(df_curv['GM'].astype(float)),
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    # 1/TR => log10(...) => interpolar => 10^(...)
                    freq = 1.0 / float(TR_int)
                    gm_log = f_HazC(np.log10(freq))
                    gm_val = 10 ** gm_log
                    SaUHS.append(gm_val)

                self.hazard_ax2.plot(
                    Pers_UHS, SaUHS,
                    marker="o",
                    linestyle="-",
                    label=f"{TR_int} yr.",
                    color=cmap(norm(i))
                )

            self.hazard_ax2.set_title("Uniform Hazard Spectrum (UHS)", fontsize=14)
            self.hazard_ax2.set_xlabel("Period (s)", fontsize=12)
            self.hazard_ax2.set_ylabel("Spectral Acceleration (g)", fontsize=12)
            self.hazard_ax2.set_yscale("log")
            self.hazard_ax2.set_xscale("log")

            self.hazard_ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
            self.hazard_ax2.legend(
                fontsize=10,
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
                title="TR",
                ncol=1,
                borderaxespad=1
            )
            self.hazard_figure2.tight_layout()
            self.hazard_canvas2.draw()

        except Exception as e:
            self.show_error(f"Error while plotting UHS: {e}")
            
    def hazard_calculation(self):
        self.plot_uhs
        self.plot_disaggregation_results
        self.load_and_plot
        
    def show_warning(self, message):
        QMessageBox.warning(self, "Warning", message)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
    
        
    def perform_cms_calculation(self, imt_tag, folder_disaggregation, source_type, database_group, magnitude_text, distance_text):
        """ Implementa el cálculo del CMS con espectro de respuesta generado por GMPEs """
    
        # Obtener parámetros del terremoto ingresados por el usuario
        try:
            zhypo_analysis = float(self.earthquake_inputs["zhypo"].text().strip())
            Width_analysis = float(self.earthquake_inputs["width"].text().strip())
            Rake_analysis = float(self.earthquake_inputs["rake"].text().strip())
            Dip_analysis = float(self.earthquake_inputs["dip"].text().strip())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numerical values for earthquake parameters.")
            return  # Asegurar que la función se detiene si hay un error
        
        try:
            vs30_measured = self.vs30measured_combo.currentText() == "True"  # Convertir a booleano
            z1pt0_value = float(self.z1_input.text().strip())
            z2pt5_value = float(self.z25_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numerical values for site parameters.")
        
        if not self.HzCurvess_Dict:
            QMessageBox.warning(self, "Error", "Hazard curves data is missing.")
            return
        
        rvolc_text = self.distance_inputs["rvolc"].text().strip()
        rvolc = float(rvolc_text) if rvolc_text else 0.0
        backarc_ = self.backarc_combo.currentText() == "True" 
        
        if source_type == "Crustal":
            correlation_func = Jayaram_Baker
            hazard_data = self.HzCurvess_Dict
            Keys_imt = [imt for imt in hazard_data.keys()
                    if imt != "Peak Ground Acceleration"] 
            try:
                periods = [float(k) for k in Keys_imt]
            except ValueError:
                # Si no se puede convertir, solo enumeramos
                periods = range(len(Keys_imt))
                
            if database_group == 'USGS':     
                imt_tags = ['Sa(0p01s)','Sa(0p02s)','Sa(0p03s)','Sa(0p05s)', 'Sa(0p075s)', 'Sa(0p1s)', 'Sa(0p15s)', 'Sa(0p2s)', 'Sa(0p25s)', 'Sa(0p3s)', 'Sa(0p4s)', 'Sa(0p5s)', 'Sa(0p75s)',
                    'Sa(1s)', 'Sa(1p5s)', 'Sa(2s)', 'Sa(3s)', 'Sa(4s)', 'Sa(5s)', 'Sa(7p5s)', 'Sa(10s)']
            
            elif database_group == 'SGC':
                imt_tags = ['Sa(0p01s)', 'Sa(0p1s)', 'Sa(0p2s)', 'Sa(0p3s)', 'Sa(0p5s)', 'Sa(0p7s)',
                    'Sa(1s)', 'Sa(1p5s)', 'Sa(2s)', 'Sa(2p5s)','Sa(3s)', 'Sa(4s)', 'Sa(5s)']
                
            Key_Hz_Dict = dict(zip(imt_tags, Keys_imt))
                
        elif source_type in ["Interface", "Intraslab"]:
            correlation_func = Macedo_Liu20
            hazard_data = self.HzCurvess_Dict
            
            if database_group == 'USGS':
                # Define periods as strings to serve as dictionary keys
                Keys_imt = ['0.01','0.05', '0.075', '0.1', '0.15', '0.2', '0.25', '0.3', '0.4', '0.5', '0.75', '1', '1.5', '2', '3', '4', '5']
                
                # Convert period keys from strings to floats for calculations
                periods = [float(el) for el in Keys_imt]
                
                imt_tags = ['Sa(0p01s)','Sa(0p05s)', 'Sa(0p075s)', 'Sa(0p1s)', 'Sa(0p15s)', 'Sa(0p2s)', 'Sa(0p25s)', 'Sa(0p3s)', 'Sa(0p4s)', 'Sa(0p5s)', 'Sa(0p75s)',
                    'Sa(1s)', 'Sa(1p5s)', 'Sa(2s)', 'Sa(3s)', 'Sa(4s)', 'Sa(5s)']
            
            elif database_group == 'SGC':
                Keys_imt = [imt for imt in hazard_data.keys()
                        if imt != "Peak Ground Acceleration"] 
                try:
                    periods = [float(k) for k in Keys_imt]
                except ValueError:
                    # Si no se puede convertir, solo enumeramos
                    periods = range(len(Keys_imt))
                  
                imt_tags = ['Sa(0p01s)', 'Sa(0p1s)', 'Sa(0p2s)', 'Sa(0p3s)', 'Sa(0p5s)', 'Sa(0p7s)',
                    'Sa(1s)', 'Sa(1p5s)', 'Sa(2s)', 'Sa(2p5s)','Sa(3s)', 'Sa(4s)', 'Sa(5s)']
                
            Key_Hz_Dict = dict(zip(imt_tags, Keys_imt))
        else:
            QMessageBox.warning(self, "Error", "Fuente sísmica desconocida. Seleccione 'Crustal', 'Interface' o 'Intraslab'.")
            return

        # Verificar si el usuario ha agregado GMPEs
        if not self.gmpe_weights:
            QMessageBox.warning(self, "Error", "No GMPEs have been added. Please add at least one.")
            return

        
        cms_data = pd.DataFrame({'Period': periods})
        sigma_cms_data = pd.DataFrame({'Period': periods})
        uhs_data = pd.DataFrame({'Period': periods})
        Rho_data = pd.DataFrame({'Period': periods})
        Sa_gmpe = pd.DataFrame({'Period': periods})
        Sa_gmpe_avg = pd.DataFrame({'Period': periods})
        Sigma_gmpe = pd.DataFrame({'Period': periods})
        SigmaFileGMM = pd.DataFrame({'Period': periods})
        Mags_Desag = []
        Rs_Desag = []
        
        for i in range(len(self.tr_list)):
            tr = self.tr_list[i]
            print(f"✅ Running CMS for TR{tr}...")
            if database_group == 'USGS':
                if not folder_disaggregation:
                    magnitude_vector = [float(mag) for mag in magnitude_text.split(",")]
                    Mean_Magnitud = magnitude_vector[i]
                    Mags_Desag.append(Mean_Magnitud)
                    distance_vector = [float(mag) for mag in distance_text.split(",")]
                    Mean_Distance = distance_vector[i]
                    Rs_Desag.append(Mean_Distance)
                    Vs30_GMM = float(self.vs30_input.text().strip())
                else:
                    List_Desag_Source = os.listdir(folder_disaggregation)
                    List_Desag_imt = [match for match in List_Desag_Source if imt_tag in match]
                    FileDesag = [match for match in List_Desag_imt if f"_{tr}." in match or f"-{tr}." in match][0]
                    
                    file_path = os.path.join(folder_disaggregation, FileDesag)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    
                    Mean_Magnitud = float(re.findall("[+-]?\\d+\\.\\d+", lines[20])[0])
                    Mags_Desag.append(Mean_Magnitud)
                    Mean_Distance = float(re.findall("[+-]?\\d+\\.\\d+", lines[21])[0])
                    Rs_Desag.append(Mean_Distance)
                    Vs30_GMM = float(lines[2].split(',')[-1])
                
            elif database_group == 'SGC':
                if not folder_disaggregation:
                    magnitude_vector = [float(mag) for mag in magnitude_text.split(",")]
                    Mean_Magnitud = magnitude_vector[i]
                    Mags_Desag.append(Mean_Magnitud)
                    distance_vector = [float(mag) for mag in distance_text.split(",")]
                    Mean_Distance = distance_vector[i]
                    Rs_Desag.append(Mean_Distance)                    
                    Vs30_GMM = float(self.vs30_input.text().strip())
                else:
                    List_Desag_Source = os.listdir(folder_disaggregation)
                    List_Desag_imt = [match for match in List_Desag_Source if imt_tag in match][0]
                    file_path = os.path.join(folder_disaggregation, List_Desag_imt)
                    df_dessaggregation = pd.read_csv(file_path)
                    df_dessaggregation = df_dessaggregation[df_dessaggregation['TR'] == tr]
                    
                    Mean_Magnitud = float(df_dessaggregation["MMean"].iloc[0])
                    Mags_Desag.append(Mean_Magnitud)
                    Mean_Distance = float(df_dessaggregation["RMean"].iloc[0])
                    Rs_Desag.append(Mean_Distance)  
                    Vs30_GMM = float(df_dessaggregation["Vs30"].iloc[0])
                
            site_ctx = SitesContext()
            site_ctx.vs30 = [Vs30_GMM]
            site_ctx.vs30measured = [vs30_measured]
            site_ctx.z1pt0 = [z1pt0_value]
            site_ctx.z2pt5 = [z2pt5_value]
            site_ctx.sids = [0]
            site_ctx.backarc = backarc_
            
            rup_ctx = RuptureContext()
            rup_ctx.mag = Mean_Magnitud
            rup_ctx.width = Width_analysis
            rup_ctx.ztor = zhypo_analysis - rup_ctx.width/2
            rup_ctx.hypo_depth = zhypo_analysis
            rup_ctx.rake = Rake_analysis
            rup_ctx.dip = Dip_analysis
            
            dist_ctx = DistancesContext()
            dist_ctx.rhypo = [Mean_Distance]
            dist_ctx.rrup = [Mean_Distance]
            dist_ctx.rvolc = rvolc
            dist_ctx.rjb = [np.round(np.sqrt(dist_ctx.rrup[0]**2 - rup_ctx.ztor**2), 1)]
            dist_ctx.repi = [np.round(np.sqrt(Mean_Distance**2 - zhypo_analysis**2), 1)]
            dist_ctx.rx = [Mean_Distance]
            dist_ctx.ry0 = [Mean_Distance]
            
            print(f"{'-' * 7} mag = {rup_ctx.mag} for TR = {tr} {'-' * 7}")
            print(f"{'-' * 7} Dist = {dist_ctx.rrup} for TR = {tr} {'-' * 7}")
            

            total_weight = sum(self.gmpe_weights.values())
            print(self.gmpe_weights)
            
            for i, (gmpe, weight) in enumerate(self.gmpe_weights.items()):
                if total_weight > 0:
                    gmpe_full_name = gmpe_map[gmpe]  # Obtener el nombre completo
                    module_name, class_name = gmpe_full_name.rsplit(".", 1)
                    module = importlib.import_module(f"openquake.hazardlib.gsim.{module_name}")
                    gmpe_class = getattr(module, class_name)()
                    weighted_rsp = np.zeros_like(periods)
                    weighted_sigma = np.zeros_like(periods)
                    for idx, period in enumerate(periods):
                        if period == 0.01:
                            imt = PGA()
                        else:
                            imt = SA(period)                        
                        mean, sigma = gmpe_class.get_mean_and_stddevs(site_ctx, rup_ctx, dist_ctx, imt, ['Total'])
                        weighted_rsp[idx] += np.exp(mean[0])
                        weighted_sigma[idx] += sigma[0][0]
                    Sa_gmpe[gmpe] = weighted_rsp
                    Sigma_gmpe[gmpe] = weighted_sigma
                    wLT = weight
                    if i == 0:
                        Sa_gmpe['weighted average'] = Sa_gmpe[gmpe]*wLT
                        Sigma_gmpe['weighted average'] = Sigma_gmpe[gmpe]*wLT
                    else: 
                        Sa_gmpe['weighted average'] = Sa_gmpe['weighted average'] + Sa_gmpe[gmpe]*wLT
                        Sigma_gmpe['weighted average'] = Sigma_gmpe['weighted average'] + Sigma_gmpe[gmpe]*wLT

            T0 = float(Key_Hz_Dict[imt_tag])
            Sa_gmpe_T0 = Sa_gmpe[Sa_gmpe.Period == T0]
            print(Sa_gmpe_T0)
            Sigma_gmpe_T0 = Sigma_gmpe[Sigma_gmpe.Period == T0]
            print(Sigma_gmpe_T0)
            
            Hazard_Curve_IM = hazard_data[Key_Hz_Dict[imt_tag]]
            f_HazC = interpolate.interp1d(np.log10(Hazard_Curve_IM['AEP'].astype(float)),
                                          np.log10(Hazard_Curve_IM['GM'].astype(float)))
            freq = 1.0 / float(tr)
            gm_log = f_HazC(np.log10(freq))
            Sa_Haz_Curve = 10 ** gm_log
            Rho_vals = np.array(correlation_func(periods, T0))
            Rho_vals[Rho_vals >= 1] = 1 / Rho_vals[Rho_vals >= 1]
            
            Epsilon_T0 = (np.log(Sa_Haz_Curve) - np.log(Sa_gmpe_T0['weighted average'])) / Sigma_gmpe_T0['weighted average'] 
            Epsilon_Bar = Epsilon_T0.values*np.array(Rho_vals)
            CMS = Sa_gmpe['weighted average'] * np.exp(Sigma_gmpe['weighted average'] * Epsilon_Bar)
            Sigma_CMS = Sigma_gmpe['weighted average'] * np.sqrt(np.ones_like(Rho_vals) - Rho_vals ** 2)
            cms_data[f'TR{tr}'] = CMS
            Sa_gmpe_avg[f'TR{tr}'] = Sa_gmpe['weighted average']
            sigma_cms_data[f'TR{tr}'] = Sigma_CMS
            SigmaFileGMM[f'TR{tr}'] = Sigma_gmpe['weighted average']
            sa_values = []
            
            for K in Keys_imt:
                hazard_curve = hazard_data[K]
                f_interp = interpolate.interp1d(np.log10(hazard_curve['AEP'].astype(float)),
                                              np.log10(hazard_curve['GM'].astype(float)))
                sa_values.append(10**f_interp(np.log10(freq)))
            
            uhs_data[f'TR{tr}'] = sa_values
            Rho_data[f'TR{tr}'] = Rho_vals
            
        self.ax2.clear()
        self.figure2.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.85)
        cmap = cm.get_cmap("viridis", len(self.tr_list))
        norm = mcolors.Normalize(vmin=0, vmax=len(self.tr_list) - 1)
        for i in range(len(self.tr_list)):
            tr = self.tr_list[i]
            self.ax2.loglog(periods, cms_data[f'TR{tr}'], linestyle='--', lw = 2.0, color = cmap(norm(i)))
            self.ax2.loglog(periods, uhs_data[f'TR{tr}'], linestyle='-', lw = 2.0, color = cmap(norm(i)), label=f'{tr} yr')
        
        self.ax2.axvline(x = T0, ls = ':', lw = 3.0, color = 'black')
        self.ax2.set_xlabel('Period (s)', fontsize=12)
        self.ax2.set_ylabel('Spectral Acceleration (g)', fontsize=12)
        self.ax2.set_title(f'CMS-UHS for {imt_tag}', fontsize=12)
        self.ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
        self.ax2.legend(
            fontsize=10,
            loc="upper left",
            bbox_to_anchor=(1.0, 1),
            title="TR",
            ncol=1,
            borderaxespad=1
        )
        self.canvas2.draw()
        
        self.cms_tabs.clear()
        
        # Generar y agregar pestañas por cada TR
        for i, tr in enumerate(self.tr_list):
            fig, ax = plt.subplots(figsize=(7, 6))  # Crear una figura para cada pestaña
            ax.loglog(periods, uhs_data[f'TR{tr}'], '-', color='navy', linewidth=2, label='UHS')
        
            ax.loglog(periods, Sa_gmpe_avg[f'TR{tr}'], '-', color='k', linewidth=2, label='GMM Avg')
        
            ax.loglog(periods, cms_data[f'TR{tr}'], '-', color='g', linewidth=2, label='CMS')
            ax.fill_between(periods, 
                            cms_data[f'TR{tr}'] * np.exp(-2.5 * sigma_cms_data[f'TR{tr}']), 
                            cms_data[f'TR{tr}'] * np.exp(2.5 * sigma_cms_data[f'TR{tr}']), 
                            color='g', alpha=0.15, label='CMS ±2.5σ')
        
            ax.set_xlabel('Period (s)', fontsize=12)
            ax.set_ylabel('Spectral Acceleration (g)', fontsize=12)
            ax.set_title(f'CMS for TR {tr} yr', fontsize=12)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # Crear canvas de matplotlib
            canvas = FigureCanvas(fig)
        
            # Crear un nuevo widget para la pestaña
            tab = QWidget()
            tab_layout = QVBoxLayout()
            tab_layout.addWidget(canvas)
            tab.setLayout(tab_layout)
        
            # Agregar la pestaña al QTabWidget
            self.cms_tabs.addTab(tab, f"TR {tr}")
        
        # Asegurar que la primera pestaña esté activa al finalizar
        if self.tr_list:
            self.cms_tabs.setCurrentIndex(0)
            
        return periods, cms_data, uhs_data, SigmaFileGMM, Rho_data, T0, Mags_Desag, Rs_Desag
    

        
    def calculate_cms(self):
        """Calcula el CMS basado en la amenaza sísmica, GMPEs y correlaciones."""
        
        # Verificar que hay datos de amenaza sísmica cargados
        if not self.HzCurvess_Dict:
            QMessageBox.warning(self, "Error", "Please load hazard curves first.")
            return
    
        # Obtener el Source (Tipo de fuente) y el IMT seleccionado
        imt_tag = self.im_target_input.text()
    
        # Obtener la lista de períodos de retorno desde la pestaña Hazard
        tr_text = self.tr_input.text().strip()
        if not tr_text:
            QMessageBox.warning(self, "Error", "Please enter at least one Return Period.")
            return
    
        try:
            self.tr_list = [int(x.strip()) for x in tr_text.split(",") if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid Return Periods format.")
            return
    
    
        # Obtener la carpeta de desagregación ingresada por el usuario
        if self.radio_usgs.isChecked():
            folder_disaggregation = self.folder_input.text().strip()
            magnitude_text = None 
            distance_text = None 
            if not folder_disaggregation:          
                magnitude_text = self.earthquake_inputs["magnitude"].text().strip()
                distance_text = self.distance_inputs["rrup"].text().strip()
                if not magnitude_text and not distance_text:
                    QMessageBox.warning(self, "Error", "Please enter Magnitude and Rrup o Disaggregation folder")
                    
        elif self.radio_sgc.isChecked(): 
            city = self.ciudad_combo.currentText().strip()
            if city != "Others":
                folder_disaggregation = os.path.join('Colombia', city, 'Dessaggregation')
                magnitude_text = None 
                distance_text = None 
            else: 
                folder_disaggregation = self.folder_input.text().strip()
                magnitude_text = None 
                distance_text = None 
                if not folder_disaggregation:
                    magnitude_text = self.earthquake_inputs["magnitude"].text().strip()
                    distance_text = self.distance_inputs["rrup"].text().strip()
                    if not magnitude_text and not distance_text:
                        QMessageBox.warning(self, "Error", "Please enter Magnitude and Rrup o Disaggregation folder")                    
        
        source_type = self.source_combo.currentText().strip()
        if self.radio_sgc.isChecked():
         database_group = 'SGC'
        elif self.radio_usgs.isChecked():
         database_group = 'USGS'  
        else:
         database_group = 'Others'  
        periods, cms, uhs, SigmaFileGMM, Rho_data, T0, Mags_Desag, Rs_Desag = self.perform_cms_calculation(imt_tag, folder_disaggregation, source_type, database_group, magnitude_text, distance_text)
        
        try:
            os.mkdir(os.path.join('Selection_%s'%(imt_tag)))
            uhs.to_excel(os.path.join('Selection_%s'%(imt_tag),'UHS_for_%s.xlsx'%(imt_tag)), index = False)
            cms.to_excel(os.path.join('Selection_%s'%(imt_tag),'CMS_for_%s.xlsx'%(imt_tag)), index = False)
            SigmaFileGMM.to_excel(os.path.join('Selection_%s'%(imt_tag),'SigmaGMM_for_%s.xlsx'%(imt_tag)), index = False)
            Rho_data.to_excel(os.path.join('Selection_%s'%(imt_tag),'Rho_for_%s.xlsx'%(imt_tag)), index = False)
            
        except:
            uhs.to_excel(os.path.join('Selection_%s'%(imt_tag),'UHS_for_%s.xlsx'%(imt_tag)), index = False)
            cms.to_excel(os.path.join('Selection_%s'%(imt_tag),'CMS_for_%s.xlsx'%(imt_tag)), index = False)
            SigmaFileGMM.to_excel(os.path.join('Selection_%s'%(imt_tag),'SigmaGMM_for_%s.xlsx'%(imt_tag)), index = False)
            Rho_data.to_excel(os.path.join('Selection_%s'%(imt_tag),'Rho_for_%s.xlsx'%(imt_tag)), index = False)
        
        shutil.copyfile('ScenarioSpectra_2017.exe', os.path.join('Selection_%s'%(imt_tag),'ScenarioSpectra_2017.exe'))
        
        UHS_File = open('UHS_Inp.txt', 'r')
        UHS_File_Content = UHS_File.readlines()
        Input_UHS_File = []
        for iline in range(len(UHS_File_Content)):
            
            if iline == 5:
                line = UHS_File_Content[iline].replace('8', '%0.0f'%(len(self.tr_list)))
                line = line.replace('\n', '')
                Input_UHS_File.append(line)
            
            elif iline == 9:
                for i_TR in range(len(self.tr_list)):
                    line_aux =  '\t%0.2e\t%0.0f'%(float(self.tr_list[i_TR])**-1, float(self.tr_list[i_TR]))
                    line_aux = line_aux.replace('e','E')
                    Input_UHS_File.append(line_aux)
        
            elif iline > 9 and iline < 17:
                line = UHS_File_Content[iline].replace('\n', '')
        
            elif iline == 18:
                line = UHS_File_Content[iline].replace('21', '%0.0f'%(uhs.shape[0]))
                line = line.replace('\n', '')
                Input_UHS_File.append(line)
            
            elif iline == 21:
                for i_Per in range(uhs.shape[0]):
                    line_aux =  '\t%0.3f	%0.0f	DCPP_run_all_12x.out3'%(uhs['Period'].iloc[i_Per], i_Per+1)
                    Input_UHS_File.append(line_aux)
            
            elif iline > 21 and iline < 42:
                line = UHS_File_Content[iline].replace('\n', '')
        
            elif iline == 43:
                line = ' Testing Levels   :  '
                for i_TR in range(len(self.tr_list)):
                    line += '%0.6f\t'%(float(self.tr_list[i_TR])**-1)
                Input_UHS_File.append(line)
            
            elif iline == 44:
                line = ' Return Period(yr):  '
                for i_TR in range(len(self.tr_list)):
                    line += '%0.0f\t\t'%(float(self.tr_list[i_TR]))
                Input_UHS_File.append(line)
        
            elif iline == 46:
                for i_Per in range(uhs.shape[0]):
                    line = '%0.0f\t%0.3f'%(i_Per + 1, uhs['Period'].iloc[i_Per])
                    for i_TR in range(uhs.shape[1]-1):
                        if np.isnan(float(uhs.iloc[i_Per, i_TR + 1])):
                            line += '\t0.000000'
                        else:
                            line += '\t%0.6f'%(float(uhs.iloc[i_Per, i_TR + 1]))
                  
                        
                    Input_UHS_File.append(line)
        
            elif iline > 46 :
                line = UHS_File_Content[iline].replace('\n', '')
        
        
            else:
                line = UHS_File_Content[iline].replace('\n', '')
                Input_UHS_File.append(line)
                
        filename_UHS = 'UHS_for_%s.txt'%(imt_tag).replace('(','_')
        filename_UHS = filename_UHS.replace(')','')
        np.savetxt(os.path.join('Selection_%s'%(imt_tag),filename_UHS), Input_UHS_File, fmt='%s')
        
        CMS_File = open('CMS_Inp.txt', 'r')
        CMS_File_Content = CMS_File.readlines()
        Input_CMS_File = []
        for iline in range(len(CMS_File_Content)):
            
            if iline == 5:
                line = CMS_File_Content[iline].replace('8', '%0.0f'%(len(self.tr_list)))
                line = line.replace('\n', '')
                Input_CMS_File.append(line)
            
            elif iline == 9:
                for i_TR in range(len(self.tr_list)):
                    line_aux =  '\t%0.2e\t%0.0f'%(float(self.tr_list[i_TR])**-1, float(self.tr_list[i_TR]))
                    line_aux = line_aux.replace('e','E')
                    Input_CMS_File.append(line_aux)
        
            elif iline > 9 and iline < 17:
                line = CMS_File_Content[iline].replace('\n', '')
        
            elif iline == 18:
                line = CMS_File_Content[iline].replace('21', '%0.0f'%(cms.shape[0]))
                line = line.replace('\n', '')
                Input_CMS_File.append(line)
            
            elif iline == 21:
                for i_Per in range(cms.shape[0]):
                    line_aux =  '\t%0.3f	%0.0f	DCPP_run_all_12x.out3'%(cms['Period'].iloc[i_Per], i_Per+1)
                    Input_CMS_File.append(line_aux)
            
            elif iline > 21 and iline < 42:
                line = CMS_File_Content[iline].replace('\n', '')
        
            elif iline == 43:
                line = ' Testing Levels   :  '
                for i_TR in range(len(self.tr_list)):
                    line += '%0.6f\t'%(float(self.tr_list[i_TR])**-1)
                Input_CMS_File.append(line)
            
            elif iline == 44:
                line = ' Return Period(yr):  '
                for i_TR in range(len(self.tr_list)):
                    line += '%0.0f\t\t'%(float(self.tr_list[i_TR]))
                Input_CMS_File.append(line)
        
            elif iline == 46:
                for i_Per in range(cms.shape[0]):
                    line = '%0.0f\t%0.3f'%(i_Per + 1, cms['Period'].iloc[i_Per])
                    for i_TR in range(cms.shape[1]-1):
                        if np.isnan(float(cms.iloc[i_Per, i_TR + 1])):
                            line += '\t0.000000'
                        else:
                            line += '\t%0.6f'%(float(cms.iloc[i_Per, i_TR + 1]))
                  
                    Input_CMS_File.append(line)
        
            elif iline > 46 and iline < 67:
                line = CMS_File_Content[iline].replace('\n', '')
        
            elif iline == 69:
                for i_Per in range(cms.shape[0]):
                    line = '%0.0f\t%0.3f'%(i_Per + 1, cms['Period'].iloc[i_Per])
                    for i_TR in range(cms.shape[1]-1):
                        line += '\t***'
                        
                    Input_CMS_File.append(line)
        
            elif iline > 69 and iline < 90:
                line = CMS_File_Content[iline].replace('\n', '')
        
            elif iline == 92:
                for i_Per in range(SigmaFileGMM.shape[0]):
                    line = '%0.0f\t%0.3f'%(i_Per + 1, SigmaFileGMM['Period'].iloc[i_Per])
                    for i_TR in range(SigmaFileGMM.shape[1]-1):
                        if np.isnan(float(SigmaFileGMM.iloc[i_Per, i_TR + 1])):
                            line += '\t0.000'
                        else:
                            line += '\t%0.3f'%(float(SigmaFileGMM.iloc[i_Per, i_TR + 1]))
                            
                    Input_CMS_File.append(line)
        
            elif iline > 92 and iline < 113:
                line = CMS_File_Content[iline].replace('\n', '')
        
        
            elif iline == 115:
                for i_Per in range(Rho_data.shape[0]):
                    line = '%0.0f\t%0.3f'%(i_Per + 1, Rho_data['Period'].iloc[i_Per])
                    for i_TR in range(Rho_data.shape[1]-1):
                        if np.isnan(float(Rho_data.iloc[i_Per, i_TR + 1])):
                            line += '\t0.000'
                        else:
                            line += '\t%0.3f'%(float(Rho_data.iloc[i_Per, i_TR + 1]))
                            
                    Input_CMS_File.append(line)
        
            elif iline > 115:
                line = CMS_File_Content[iline].replace('\n', '')
        
        
            else:
                line = CMS_File_Content[iline].replace('\n', '')
                Input_CMS_File.append(line)
        
        filename_CMS = 'CMS_for_%s.txt'%(imt_tag).replace('(','_')
        filename_CMS = filename_CMS.replace(')','')
        np.savetxt(os.path.join('Selection_%s'%(imt_tag),filename_CMS), Input_CMS_File, fmt='%s')
        
        Main_File = open('Hz_Inp.txt', 'r')
        Inp_File_Content = Main_File.readlines()
        
        Inp_File_Content[2] = Inp_File_Content[2].replace('0.75\t', '%0.3f\t'%(T0)) #T*
        Inp_File_Content[4] = Inp_File_Content[4].replace('0.15\t1.125\t', '%0.3f\t%0.3f\t'%(0.2*T0, 2.0*T0)) 
        Inp_File_Content[5] = Inp_File_Content[5].replace('2.0\t8.0\t', '%0.1f\t%0.1f\t'%(np.min(Mags_Desag) - 1.0, np.max(Mags_Desag) + 1.0))
        Inp_File_Content[7] = Inp_File_Content[7].replace('0.0    1000.0\t', '%0.1f    %0.1f\t'%(np.max([np.min(Rs_Desag)]) - 10, np.max(Rs_Desag) + 50))
        Inp_File_Content[8] = Inp_File_Content[8].replace('300.0  435.0\t', '%0.1f    %0.1f\t'%(0, 3000))
        Inp_File_Content[12] = Inp_File_Content[12].replace('32\t', '%0.0f'%(32))
        
        
        Inp_File_Content[18] = Inp_File_Content[18].replace('UHS_0p75s_Seattle.inp\n', '%s\n'%(filename_UHS))
        

        Inp_File_Content[19] = Inp_File_Content[19].replace('CMS_0p75s_Seattle.txt\n', '%s\n'%(filename_CMS))
        
        Inp_File_Content[20] = Inp_File_Content[20].replace('flatfile.csv\n', '%s\n'%('flatfile_%s.csv'%(source_type)))
        
        for iline in range(22,28):
            im_tag = imt_tag.replace('(','_')
            im_tag = im_tag.replace(')','')
            Inp_File_Content[iline] = Inp_File_Content[iline].replace('CSS_Seattle_0p75s', 'CSS_%s'%(im_tag))
        
        for iline in range(len(Inp_File_Content)):
            Inp_File_Content[iline] = Inp_File_Content[iline].replace('\n', '')
        
        filename_Main = 'Main_CSS.txt'
        np.savetxt(os.path.join('Selection_%s'%(imt_tag),filename_Main), Inp_File_Content, fmt='%s')
        
        
        if source_type == "Crustal" :
            csv_name = "flatfile_NGA_West2_FullSet_V2.csv"
            df_flatfile = pd.read_csv(os.path.join('Records_NGA', csv_name), skiprows = 3)
        elif source_type == "Intraslab" :
            csv_name = "NGAsubIntraslabFullSet.csv"
            df_flatfile = pd.read_csv(os.path.join('Records_NGA_Sub', csv_name))
        else:
            csv_name = "NGAsubInterfaceFullSet.csv"
            df_flatfile = pd.read_csv(os.path.join('Records_NGA_Sub', csv_name))           
                
        
        df_header= pd.read_csv(os.path.join('Records_NGA','flatfile_NGA_West2_FullSet_V2.csv'), nrows=3, header=None)
        line1 = df_header.iloc[0].fillna('').tolist()
        line2 = df_header.iloc[1].fillna('').tolist()
        line3 = df_header.iloc[2].fillna('').tolist()
        data_as_list = [line1, line2, line3]
        
        df_filtered = df_flatfile[
            (df_flatfile['Earthquake Magnitude'] >= (np.min(Mags_Desag) - 1.0)) & 
            (df_flatfile['Earthquake Magnitude'] <= (np.max(Mags_Desag) + 1.0))
        ]
        
        df_filtered = df_filtered[
            (df_filtered['ClstD (km)'] >= (np.min(Rs_Desag) - 10)) & 
            (df_filtered['ClstD (km)'] <= (np.max(Rs_Desag) + 50))
        ]
        
        if df_filtered.shape[0] <= 595:
           data_as_list.append(list(df_filtered.columns))  
           data_as_list.extend(df_filtered.values.tolist())
           result_df = pd.DataFrame(data_as_list)
           result_df.to_csv(os.path.join('Selection_%s'%(imt_tag),'flatfile_%s.csv'%(source_type)), index=None, header=None)
        else: 
           df_filtered = df_filtered.sample(n=595, random_state=42)
           df_filtered = df_filtered.sort_values(by='Record Sequence Number')
           data_as_list.append(list(df_filtered.columns))  
           data_as_list.extend(df_filtered.values.tolist())
           result_df = pd.DataFrame(data_as_list)  
           result_df.to_csv(os.path.join('Selection_%s'%(imt_tag),'flatfile_%s.csv'%(source_type)), index=None, header=None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HazardApp()
    window.show()
    sys.exit(app.exec_())
