# AEGIS Mission Control UI

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
streamlit run ui/app.py
```

The app has seven modules: Mission Control, Algorithm Laboratory, Rival
Intelligence, Environment Laboratory, AI Command Centre, Research Analytics,
and System Architecture. Controls dispatch to Parts A through D. No Part E
implementation exists in this repository, so the architecture page identifies
the current boundary instead of inventing Part E results.
