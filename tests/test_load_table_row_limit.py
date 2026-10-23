import io

import pandas as pd
import pytest

from tis.core import load_table


def test_csv_row_limit_is_enforced():
    with pytest.raises(ValueError, match="row import limit"):
        load_table(
            b"time,signal\n0,1\n1,2\n2,3\n",
            "sample.csv",
            max_rows=2,
        )


def test_excel_row_limit_is_enforced():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"signal": [1, 2, 3]}).to_excel(writer, index=False)

    with pytest.raises(ValueError, match="row import limit"):
        load_table(buffer.getvalue(), "sample.xlsx", max_rows=2)
