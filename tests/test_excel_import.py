from io import BytesIO

import pandas as pd

from tis.core import load_table


def _xlsx_bytes():
    stream = BytesIO()
    pd.DataFrame({"time_s": [0.0, 1.0], "signal": [2.0, 3.0]}).to_excel(
        stream, index=False, engine="openpyxl"
    )
    return stream.getvalue()


def test_load_table_xlsx_uses_first_row_as_header_by_default():
    frame = load_table(_xlsx_bytes(), "sample.xlsx")
    assert list(frame.columns) == ["time_s", "signal"]
    assert frame.shape == (2, 2)


def test_load_table_xlsx_supports_header_none():
    frame = load_table(_xlsx_bytes(), "sample.xlsx", header=None)
    assert frame.shape == (3, 2)
    assert frame.iloc[0].tolist() == ["time_s", "signal"]
