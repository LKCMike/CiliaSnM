# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os

PACKAGES_TO_COLLECT = [
    'multiprocessing',
    'xsdata',
    'xsdata_pydantic_basemodel',
    'ome_types',
    'aicsimageio',
    'pydantic',
    'imagecodecs',
    'tifffile',
    'xarray',
    'zarr',
]

def collect_all_submodules(package_names):
    all_modules = []
    for package in package_names:
        try:
            modules = collect_submodules(package)
            all_modules.extend(modules)
            print(f"✓ Collected {len(modules)} modules from {package}")
        except Exception as e:
            print(f"⚠ Warning: Could not collect {package}: {e}")
    return all_modules

hiddenimports = collect_all_submodules(PACKAGES_TO_COLLECT)

# 添加特定的重要模块（确保它们被包含）
essential_imports = [
    'xsdata_pydantic_basemodel.hooks',
    'xsdata_pydantic_basemodel.hooks.class_type',
    'ome_types._converter',
    'imagecodecs._shared',
]

hiddenimports.extend(essential_imports)

hiddenimports = list(set(hiddenimports))
print(f"\nTotal unique hidden imports: {len(hiddenimports)}")

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,  # 添加这一行
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CiliaSnM',
    contents_directory='lib',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Cilia Statistic and Measurement',
)
