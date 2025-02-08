from setuptools import setup, find_packages

setup(
    name="dm_engine",
    version="0.1.0",
    packages=find_packages(include=["dm_engine", "dm_engine.*"]),
    py_modules=["llm", "conversation", "persona", "player_model"], 
    entry_points={
        "console_scripts": [
            "dm-desktop=dm_engine.desktop:main",
        ],
    },
)
