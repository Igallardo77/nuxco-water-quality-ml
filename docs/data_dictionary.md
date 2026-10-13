# Data dictionary

The analysis worksheet is `01_Dataset_ML` in `data/Base_Datos_Calidad_Agua_Integrada.xlsx`. There are 65 observations: five repeated sites over 13 campaigns (February 2016–March 2017; no March 2016).

Identification columns: `ID_muestra` (unique sample identifier), `Fecha` (nominal first day of month), `Anio`, `Mes_num`, `Mes`, `Periodo` (YYYY-MM), `Orden_temporal` (1–13), `Punto` (1–5), and `Zona_exploratoria` (P1–P3 / P4–P5).

| Variable | Meaning | Unit |
|---|---|---|
| `Temperatura_C` | Water temperature | °C |
| `pH` | pH | dimensionless |
| `CE_mS_cm` | Electrical conductivity | mS/cm |
| `SDT_mg_L` | Total dissolved solids | mg/L |
| `Salinidad_UPS` | Salinity | PSU |
| `OD_mg_L` | Dissolved oxygen | mg/L |
| `DBO5_mg_L` | Five-day biological oxygen demand | mg/L |
| `DQO_mg_L` | Chemical oxygen demand | mg/L |
| `SST_mg_L` | Total suspended solids | mg/L |
| `SS_mL_L` | Settleable solids | mL/L (verify laboratory unit) |
| `NH3_N_mg_L` | Ammoniacal nitrogen | mg/L |
| `NO2_mg_L` | Nitrite | mg/L |
| `NO3_mg_L` | Nitrate | mg/L |

**Unit discrepancy:** a manuscript version describes settleable solids in mg/L, while the original data column is `SS_mL_L`. No values should be converted without laboratory confirmation.
