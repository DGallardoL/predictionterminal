## Tasks

### Pre-flight
- [ ] **Verify slug returns history.** Query Kalshi directly:
      ```bash
      .venv/bin/python -c "import sys; sys.path.insert(0,'src'); \
        from pfm.sources.kalshi import KalshiClient, fetch_factor_history; \
        c = KalshiClient(); \
        df = fetch_factor_history(c, 'KXLCPIMAXYOY-27'); \
        c.close(); print(f'bars={len(df)}', df.head(3).to_dict())"
      ```
      Expect ≥100 bars; abort if not.

### Implementation
- [ ] **Add yaml entry** to `api/src/pfm/factors.yml` under the Kalshi block:
      ```yaml
      - id: k_cpi_max_yoy_27
        name: "Max US CPI YoY in 2027 (Kalshi)"
        slug: "KXLCPIMAXYOY-27"
        source: kalshi
        theme: macro
        description: >
          Highest YoY headline CPI print expected during 2027. Direct read
          on inflation regime — drives rate expectations, hits long-duration
          tech, and pairs with `no_fed_cuts_2026` / `eleven_fed_cuts` for
          inflation-rate composite. CFTC-regulated, has daily volume + OI.
      ```

### Verification
- [ ] **Tests still pass.** Run `cd api && .venv/bin/pytest -q` — expect 51/51.
- [ ] **Server picks up the entry.** Restart uvicorn and curl
      `GET /factors`; confirm `k_cpi_max_yoy_27` is listed with
      `source="kalshi"` and `theme="macro"`.
- [ ] **UI smoke-test.** Reload `localhost:8000`, go to Curated → macro tab,
      confirm the new card with green "K" badge and a sparkline.
- [ ] **Regression smoke-test.** Run `/fit` for NVDA with only the new
      factor selected — expect non-zero R², no errors.

### Cleanup
- [ ] **Archive change** with `openspec archive add-cpi-kalshi-factor`
      to apply the spec delta into `openspec/specs/factors-catalog/spec.md`.
