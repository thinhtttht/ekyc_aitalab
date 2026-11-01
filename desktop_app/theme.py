"""Design system and modern dark QSS stylesheet for eKYC Desktop Application."""

DARK_STYLESHEET = """
/* Global Application Styles */
QWidget {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
    selection-background-color: #38bdf8;
    selection-color: #0f172a;
}

/* Main Window */
QMainWindow {
    background-color: #0b1120;
}

/* ScrollBars */
QScrollBar:vertical {
    border: none;
    background: #1e293b;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #475569;
    min-height: 25px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #64748b;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #1e293b;
    height: 8px;
    margin: 0px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #475569;
    min-width: 25px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #64748b;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Sidebar Navigation Buttons */
QPushButton.nav-btn {
    background-color: transparent;
    color: #94a3b8;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 14px;
    font-weight: 600;
    text-align: left;
}
QPushButton.nav-btn:hover {
    background-color: rgba(56, 189, 248, 0.08);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.2);
}
QPushButton.nav-btn:checked {
    background-color: rgba(56, 189, 248, 0.16);
    color: #38bdf8;
    border: 1px solid #38bdf8;
}

/* Primary Action Buttons */
QPushButton.btn-primary {
    background-color: #0284c7;
    color: #ffffff;
    border: 1px solid #38bdf8;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton.btn-primary:hover {
    background-color: #0369a1;
    border-color: #7dd3fc;
}
QPushButton.btn-primary:pressed {
    background-color: #075985;
}
QPushButton.btn-primary:disabled {
    background-color: #334155;
    border-color: #475569;
    color: #64748b;
}

/* Success Action Buttons */
QPushButton.btn-success {
    background-color: #059669;
    color: #ffffff;
    border: 1px solid #10b981;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton.btn-success:hover {
    background-color: #047857;
}

/* Danger Buttons */
QPushButton.btn-danger {
    background-color: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 12px;
}
QPushButton.btn-danger:hover {
    background-color: #ef4444;
    color: #ffffff;
}

/* Secondary / Ghost Buttons */
QPushButton.btn-secondary {
    background-color: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 13px;
}
QPushButton.btn-secondary:hover {
    background-color: #334155;
    color: #ffffff;
}

/* Input Fields */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1e293b;
    color: #ffffff;
    border: 1.5px solid #475569;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 500;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1.5px solid #38bdf8;
    background-color: #0b1120;
    color: #ffffff;
}
QLineEdit[text=""] {
    color: #94a3b8;
}
QComboBox::drop-down {
    border: none;
    padding-right: 10px;
}
QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: #ffffff;
    border: 1px solid #475569;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
    padding: 6px;
}

/* Card Container */
QFrame.card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
}
QFrame.card-highlight {
    background-color: #1e293b;
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 12px;
    padding: 16px;
}

/* Table Widget */
QTableWidget {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    gridline-color: #334155;
    selection-background-color: rgba(56, 189, 248, 0.15);
    selection-color: #ffffff;
}
QTableWidget::item {
    padding: 8px 12px;
    border-bottom: 1px solid #283548;
}
QHeaderView::section {
    background-color: #0f172a;
    color: #94a3b8;
    font-weight: 600;
    font-size: 12px;
    padding: 10px 12px;
    border: none;
    border-bottom: 2px solid #38bdf8;
}

/* Progress Bar */
QProgressBar {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-weight: 600;
    height: 18px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
    border-radius: 5px;
}

/* Status Bar */
QStatusBar {
    background-color: #0b1120;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
    font-size: 12px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #334155;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #38bdf8;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f8fafc;
    border: 2px solid #0284c7;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
"""
