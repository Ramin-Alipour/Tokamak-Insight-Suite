"""Real-browser ECE workflow test against a running Streamlit server.

The complete test suite requires Playwright and an installed Chromium browser.
"""
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright, expect


def test_live_streamlit_ece_response_workflow_upload_and_download(tmp_path):
    csv = "time_s,ece_80,ece_90,ece_100\n" + "\n".join(
        f"{i * 1e-6},{10 + i * 0.1},{20 + i * 0.1},{30 + i * 0.1}"
        for i in range(32)
    )
    csv_path = tmp_path / "ece_live_smoke.csv"
    csv_path.write_text(csv, encoding="utf-8")

    server = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless=true", "--server.port=8501",
            "--browser.gatherUsageStats=false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        healthy = False
        for _ in range(90):
            if server.poll() is not None:
                raise AssertionError("Streamlit server exited before becoming healthy.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as response:
                    healthy = response.status == 200
                    if healthy:
                        break
            except Exception:
                time.sleep(1)
        assert healthy, "Streamlit server did not become healthy within 90 seconds."

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(accept_downloads=True)
            page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded", timeout=60_000)
            uploader = page.locator('input[type="file"]').first
            uploader.set_input_files(str(csv_path))
            expect(page.get_by_text("ece_live_smoke.csv", exact=False)).to_be_visible(timeout=30_000)

            page.get_by_text("Plasma Parameters", exact=True).first.click()
            page.get_by_text("Reflectometry / ECE", exact=True).click()
            workflow = page.get_by_role("combobox", name="ECE workflow")
            workflow.click()
            # Streamlit's custom select menu does not expose a consistently
            # clickable option node in headless Chromium. Keyboard navigation
            # exercises the actual widget and selects its final documented mode.
            workflow.press("End")
            workflow.press("Enter")
            expect(page.get_by_text("Convolve calibrated spectral radiance with a supplied receiver response", exact=False)).to_be_visible(timeout=15_000)

            page.get_by_label("Spectral frequency grid (Hz; comma-separated, increasing)").fill(
                "80000000000,85000000000,90000000000,95000000000,100000000000"
            )
            page.get_by_label("Calibrated spectral radiance values (same length; W m⁻² sr⁻¹ Hz⁻¹)").fill(
                "1e-10,1e-10,1e-10,1e-10,1e-10"
            )
            page.get_by_label("Response frequency grid (Hz; comma-separated, increasing)").fill(
                "85000000000,90000000000,95000000000"
            )
            page.get_by_label("Response weights: one comma-separated row per channel, rows separated by semicolons").fill(
                "0,1,0;0,0.5,0.5"
            )
            page.get_by_role("button", name="Convolve spectrum with receiver response").click()
            expect(page.get_by_text("ECE receiver-response convolution completed using the supplied kernel.", exact=False)).to_be_visible(timeout=30_000)
            with page.expect_download(timeout=30_000) as download_info:
                page.get_by_role("button", name="Export ECE receiver-response convolution").click()
            download = download_info.value
            assert download.suggested_filename == "tis_ece_receiver_response.json"
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
