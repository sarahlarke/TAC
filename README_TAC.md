# TAC Intelligence Lab

A Streamlit app for contingent workforce analysis using the TAC dummy dataset.

## Files
- `tac_app.py` - main app
- `requirements_tac.txt` - dependencies
- `TAC_dummy_data_150.xlsx` - dummy data

## Run locally
```bash
pip install -r requirements_tac.txt
streamlit run tac_app.py
```

## Notes
- The app is designed around the `TAC_Data` sheet in the workbook.
- If derived fields are blank in the source, the app calculates them automatically.


