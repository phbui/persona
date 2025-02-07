from setuptools import setup, find_packages

setup(
    name="dm_engine",
    version="0.1.0",
    packages=find_packages(include=["dm_engine", "dm_engine.*"]),
    py_modules=["llm", "conversation", "persona", "playermodel"], 
    entry_points={
        "console_scripts": [
            "dm-cli=dm_engine.cli:main", 
            "dm-gui=dm_engine.gui:main", 
        ],
    },
)
