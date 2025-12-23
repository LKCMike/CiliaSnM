"""
:Date        : 2025-12-04 08:37:06
:LastEditTime: 2025-12-11 11:55:32
:Description : CiliaSnM - Minimal Version
"""
#!/usr/bin/env python3

import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    """
    Start Qt Application
    """
    app = QApplication(sys.argv)
    app.setApplicationName("CiliaSnM")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
