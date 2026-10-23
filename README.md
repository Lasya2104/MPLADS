<<<<<<< HEAD
# MPLADS Work Intelligence — Streamlit Prototype

## What this app contains

1. Executive dashboard
2. Particular work explorer
3. Expenditure/fund anomaly detection using Isolation Forest
4. Duplicate work detection using TF-IDF + cosine similarity
5. Ground-reality evidence upload
6. Public feedback
7. Officer verification / human-in-the-loop
8. Red-Amber-Green decision support
9. Closed-loop audit tracking

## Databases

- **SQLite**: local structured records for feedback, alerts and officer audits.
- **MongoDB**: optional evidence/log storage. MongoDB Compass is only the GUI; the app connects to the MongoDB server using `MONGODB_URI`.

## Important data note

The supplied prototype CSV has 1,000 rows. The data dictionary marks several monitoring/model fields as prototype/synthetic, including some expenditure/monitoring/detection/feedback fields. Replace those with verified official records before claiming production accuracy.

## Run on Windows

Open Command Prompt in this folder:

```text
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is installed but `app.py` is not found, make sure Command Prompt is opened in the project folder:

```text
cd path\to\mplads_webapp
streamlit run app.py
```

## MongoDB

If MongoDB is installed locally:

```text
mongodb://localhost:27017/
```

Set it in a `.env` file or environment variable:

```text
MONGODB_URI=mongodb://localhost:27017/
```

Then restart Streamlit.

## Recommended production additions

- Official MPLADS portal ingestion/API or verified bulk-data process
- Role-based login
- District/MP/admin permissions
- Actual payment transaction history
- Real GIS coordinates
- Proper YOLO model trained for asset classes
- Speech-to-text and IVRS integration
- Immutable audit trail
- Model monitoring and threshold calibration
=======
# MPLADS
>>>>>>> 5018b4220ca3a38dae66860bb0ff3c7912846521
