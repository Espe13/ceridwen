"""Q2-010: ``Spectrum.display()`` prints a short head/tail table by default, not 80 pixels."""
from __future__ import annotations

import numpy as np

from ceridwen.observation.observation import Spectrum


def _spec(n):
    w = np.linspace(4000.0, 8000.0, n)
    return Spectrum(wavelength=w, flux=np.ones(n), uncertainty=0.1 * np.ones(n), name="s")


def _pixel_rows(txt):
    return [ln for ln in txt.splitlines() if ln.strip()[:1].isdigit() and "True" in ln]


def test_default_is_ten_pixel_rows_and_says_how_to_see_all():
    txt = _spec(600).display(return_str=True)
    rows = _pixel_rows(txt)
    assert len(rows) == 10
    assert rows[0].split()[0] == "0" and rows[-1].split()[0] == "599"
    assert "590 more pixels; display(max_rows=600) prints all" in txt
    assert "stats over 600/600" in txt


def test_max_rows_still_controls_the_table():
    assert len(_pixel_rows(_spec(600).display(max_rows=600, return_str=True))) == 600
    assert len(_pixel_rows(_spec(600).display(max_rows=40, return_str=True))) == 40
    assert len(_pixel_rows(_spec(8).display(return_str=True))) == 8
