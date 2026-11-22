from pathlib import Path

import pytest

from codeblue.dataroom import load_data_room

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def room():
    return load_data_room(ROOT / "data_rooms" / "harbourline")
