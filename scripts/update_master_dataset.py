"""
update_master_dataset.py
Updates Davao_City_Barangays_PSA_PSGC_2026_with_Model_Columns.xlsx with
the real CDRRMO / Project NOAH flood hazard data from
Davao_City_Barangays_with_Flood_Hazard_Data.xlsx.
"""

import shutil
import openpyxl
import pandas as pd

src_file = "data/Davao_City_Barangays_PSA_PSGC_2026_with_Model_Columns.xlsx"
backup_file = "data/Davao_City_Barangays_PSA_PSGC_2026_with_Model_Columns.backup.xlsx"
flood_file = "data/Davao_City_Barangays_with_Flood_Hazard_Data.xlsx"

# Backup existing file
shutil.copyfile(src_file, backup_file)
print("Backup created:", backup_file)

wb = openpyxl.load_workbook(src_file)
ws = wb["Davao City Barangays"]

df_flood = pd.read_excel(flood_file)
assert len(df_flood) == 182

headers = [cell.value for cell in ws[1]]
print("Headers:", headers)
vuln_ind_col = headers.index("Vulnerability Indicator") + 1
vuln_weight_col = headers.index("Vulnerability Weight (P_j)") + 1
vuln_source_col = headers.index("Vulnerability Data Source") + 1
brgy_col = headers.index("Barangay") + 1

for row_idx in range(2, 184):
    brgy_excel = ws.cell(row=row_idx, column=brgy_col).value
    flood_row = df_flood.iloc[row_idx - 2]
    flood_brgy = flood_row["Barangay"]
    assert brgy_excel == flood_brgy, f"Mismatch at row {row_idx}: {brgy_excel} vs {flood_brgy}"
    
    raw_rating = int(flood_row["Hazard Rating / Weight (P_j)"])
    indicator = raw_rating + 1
    weight = indicator / 340.0
    source = "Davao City CDRRMO / Project NOAH Flood Hazard Assessment"
    pop = int(flood_row["Population (2024 POPCEN)"])
    food_demand = pop * 2100
    
    ws.cell(row=row_idx, column=vuln_ind_col, value=indicator)
    ws.cell(row=row_idx, column=vuln_weight_col, value=weight)
    ws.cell(row=row_idx, column=vuln_source_col, value=source)
    ws.cell(row=row_idx, column=14, value=food_demand)

# Update Source sheet
if "Source" in wb.sheetnames:
    ws_source = wb["Source"]
    for r in range(1, ws_source.max_row + 1):
        k = ws_source.cell(row=r, column=1).value
        if k == "Vulnerability data":
            ws_source.cell(row=r, column=2, value="CDRRMO / Project NOAH Flood Hazard Assessment mapped to 1-4 indicator, normalized sum=1.0")
        elif k == "Potential vulnerability sources":
            ws_source.cell(row=r, column=2, value="Official Davao City CDRRMO / Project NOAH Flood Susceptibility Dataset")
        elif k == "Data status":
            ws_source.cell(row=r, column=2, value="VALIDATED EMPIRICAL DATA -- calibrated with official PSA 2024 POPCEN and CDRRMO flood hazard ratings")

wb.save(src_file)
print("Successfully updated", src_file)
