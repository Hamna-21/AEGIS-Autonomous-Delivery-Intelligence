# Part A

Part A contains the environment and world model:

- terrain grid loading and cost maps
- obstacle and movement rules
- one-way street transitions
- fog and visibility
- movement slips
- optional proposed flood module
- PEAS and environment classification
- unit tests

Run it from the project root:

```powershell
python -m part_a.run
python -m unittest discover -s tests -v
```

Search algorithms are intentionally reserved for a later part.
