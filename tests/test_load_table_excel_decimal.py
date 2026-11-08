import io

import pandas as pd

from tis.core import load_table


def test_load_excel_respects_decimal_separator():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"signal": ["1,25", "2,50"]}).to_excel(
            writer, index=False
        )

    frame = load_table(
        buffer.getvalue(),
        "decimal-comma.xlsx",
        decimal=",",
    )

    assert frame["signal"].tolist() == [1.25, 2.50]
