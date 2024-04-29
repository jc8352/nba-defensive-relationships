import pandas as pd
import json
import csv
from pathlib import Path

cwd = Path.cwd()
root = cwd.parent.absolute()

with open(root / 'data' / 'team_defense_advanced.json') as file:
     team_defense_advanced = json.load(file)
     team_defense_headers = team_defense_advanced['resultSets'][0]['headers']
     team_defense_advanced = team_defense_advanced['resultSets'][0]['rowSet']

with open(root / 'data' / 'team_wide_open_shots.json') as file2:
     opp_wide_open_shots = json.load(file2)
     wide_open_headers = opp_wide_open_shots['resultSets'][0]['headers']
     wide_open_headers[5:] = ["WIDE_OPEN_"+header for header in wide_open_headers[5:]]
     opp_wide_open_shots = opp_wide_open_shots['resultSets'][0]['rowSet']

with open(root / 'data' / 'team_open_shots.json') as file3:
     opp_open_shots = json.load(file3)
     open_headers = opp_open_shots['resultSets'][0]['headers']
     open_headers[5:] = ["OPEN_"+header for header in open_headers[5:]]
     opp_open_shots = opp_open_shots['resultSets'][0]['rowSet']

NUM_TEAMS = 30
combined_headers = []
excluded_cols = []
for i, header in enumerate(team_defense_headers+wide_open_headers+open_headers):
     if header not in combined_headers:
          combined_headers.append(header)
     else:
          excluded_cols.append(i)


team_defense_advanced = sorted(team_defense_advanced, key = lambda x: x[0])
opp_wide_open_shots = sorted(opp_wide_open_shots, key = lambda x: x[0])
opp_open_shots = sorted(opp_open_shots, key = lambda x: x[0])

#print(team_defense_advanced)
#print('\n\n\n')
#print(opp_wide_open_shots)

combined_data = []
for i in range(NUM_TEAMS):
     combined_data.append([])
     for j, elem in enumerate(team_defense_advanced[i]+opp_wide_open_shots[i]+opp_open_shots[i]):
          if j not in excluded_cols:
               combined_data[i].append(elem)



with open(root / 'data' / 'opponent.csv', 'w') as file:
	writer = csv.writer(file)
	writer.writerow(combined_headers)
	for team in combined_data:
		writer.writerow(team)


