# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Data files to include
data_files = [
    ('frontend/dist', 'frontend/dist'),
]

# Add additional files if specified
data_files.append(('darkcircuit_agent_modular.py', '.'))
data_files.append(('agent_utils.py', '.'))
data_files.append(('Rag_tool.py', '.'))
data_files.append(('streaming_handler.py', '.'))

a = Analysis(
    ['local_app_direct_ssh.py'],
    pathex=[],
    binaries=[],
    datas=data_files,
    hiddenimports=['langchain_core', 'langchain_openai', 'langgraph', 'paramiko'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DarkCircuit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='frontend/public/cyberlabs.png',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DarkCircuit',
)
