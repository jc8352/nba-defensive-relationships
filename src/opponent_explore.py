import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy import stats
from pathlib import Path

cwd = Path.cwd()
root = cwd.parent.absolute()

opponent_shooting = pd.read_csv(root / 'data' / 'opponent.csv')
opponent_shooting_23 = pd.read_csv(root / 'data' / 'opponent_2023.csv')


print("2023-24 correlation with defensive rating")
print(opponent_shooting.corr()['DEF_RATING'])
#OPP_PTS_PAINT 0.776817 correlation with defensive rating
#WIDE_OPEN_FG3A 0.131525 correlation with defensive rating
#OPEN_FG3A -0.009537 correlation with defensive rating
print('\n')

print("2022-23 correlation with defensive rating")
print(opponent_shooting_23.corr()['DEF_RATING'])
#OPP_PTS_PAINT 0.697227 correlation with defensive rating
#WIDE_OPEN_FG3A 0.073471 correlation with defensive rating
#OPEN_FG3A -0.014871 correlation with defensive rating
print('\n')


#determining what weights being applied to wide open and open 3's allowed and added to opponent points in the paints results in highest correlation with defensive rating
"""
max_corr = 0
max_corr_ij = []
max_corr_23 = 0
max_corr_ij_23 = []

combined_max = [0,0,0]
combined_max_ij = []
for i in range(0, 100, 1):
	for j in range(0, 100, 1):
		corr = stats.pearsonr(opponent_shooting['OPP_PTS_PAINT']+opponent_shooting['WIDE_OPEN_FG3A']*i/100+opponent_shooting['OPEN_FG3A']*j/100, opponent_shooting['DEF_RATING']).statistic
		if corr > max_corr:
			max_corr = corr
			max_corr_ij = [i/100, j/100]
		corr_23 = stats.pearsonr(opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A']*i/100+opponent_shooting_23['OPEN_FG3A']*j/100, opponent_shooting_23['DEF_RATING']).statistic
		if corr_23 > max_corr_23:
			max_corr_23 = corr_23
			max_corr_ij_23 = [i/100, j/100]
		if (corr+corr_23)/2 > combined_max[0]:
			combined_max = [(corr+corr_23)/2, corr, corr_23]
			combined_max_ij = [i/100, j/100]
print("max correlation 2023-24:", max_corr)
print("wide open 3's multiplier 2023-24:", max_corr_ij[0], "| open 3's multiplier 2023-24:", max_corr_ij[1])
print("\n")
print("max correlation 2022-23:", max_corr_23)
print("wide open 3's multiplier 2022-23:", max_corr_ij_23[0], "| open 3's multiplier 2022-23:", max_corr_ij_23[1])
print("\n")
print("combined max correlation:", combined_max[0], "| correlation 2023-24:", combined_max[1], "| correlation 2022-23:", combined_max[2])
print("wide open 3's multiplier combined", combined_max_ij[0], "| open 3's multiplier combined:", combined_max_ij[1])
print("\n")
"""

"""
max correlation 2023-24: 0.8687443766973358
wide open 3's multiplier 2023-24: 0.6 | open 3's multiplier 2023-24: 0.99

max correlation 2022-23: 0.7572933923802677
wide open 3's multiplier 2022-23: 0.1 | open 3's multiplier 2022-23: 0.96

combined max correlation: 0.8047839130615981 | correlation 2023-24: 0.861061221982255 | correlation 2022-23: 0.7485066041409412
wide open 3's multiplier combined 0.36 | open 3's multiplier combined: 0.98
"""

#visualizing relationships between opponent points in the paint, wide open 3's allowed, and open 3's allowed and defensive rating
plt.scatter(opponent_shooting['OPP_PTS_PAINT'], opponent_shooting['DEF_RATING'])
plt.xlabel("Opponent Points in the Paint 2023-24")
plt.ylabel("Defensive Rating 2023-24")
plt.text(52, 110, "correlation: "+str(int(stats.pearsonr(opponent_shooting['OPP_PTS_PAINT'], opponent_shooting['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.show()


norm = (opponent_shooting-opponent_shooting.mean())/opponent_shooting.std()
plt.scatter(norm['OPP_PTS_PAINT']+norm['WIDE_OPEN_FG3A'], norm['DEF_RATING'])
plt.xlabel("Normalized opponent points in the paint plus normalized wide open 3's 2023-24")
plt.ylabel("Defensive Rating 2023-24")
plt.text(1, -1.5, "correlation: "+str(int(stats.pearsonr(norm['OPP_PTS_PAINT']+norm['WIDE_OPEN_FG3A'], norm['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.show()

plt.scatter(opponent_shooting['OPP_PTS_PAINT']+opponent_shooting['WIDE_OPEN_FG3A'], opponent_shooting['DEF_RATING'])
plt.xlabel("Opponent points in the paint plus wide open 3's 2023-24")
plt.ylabel("Defensive Rating 2023-24")
plt.text(72, 110, "correlation: "+str(int(stats.pearsonr(opponent_shooting['OPP_PTS_PAINT']+opponent_shooting['WIDE_OPEN_FG3A'], opponent_shooting['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.show()

plt.scatter(opponent_shooting['OPP_PTS_PAINT']+opponent_shooting['WIDE_OPEN_FG3A']+opponent_shooting['OPEN_FG3A'], opponent_shooting['DEF_RATING'])
plt.xlabel("Opponent points in the paint plus wide open 3's plus open 3's 2023-24")
plt.ylabel("Defensive Rating 2023-24")
plt.text(85, 110, "correlation: "+str(int(stats.pearsonr(opponent_shooting['OPP_PTS_PAINT']+opponent_shooting['WIDE_OPEN_FG3A']+opponent_shooting['OPEN_FG3A'], opponent_shooting['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.show()

opponent_shooting_23['PITP_PLUS_WIDE'] = opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A']
plt.scatter(opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A'], opponent_shooting_23['DEF_RATING'])
plt.xlabel("Opponent points in the paint plus wide open 3's 2022-23")
plt.ylabel("Defensive Rating 2022-23")
plt.text(71, 111, "correlation: "+str(int(stats.pearsonr(opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A'], opponent_shooting_23['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.show()

plt.scatter(opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A']+opponent_shooting_23['OPEN_FG3A'], opponent_shooting_23['DEF_RATING'])
plt.xlabel("Opponent points in the paint plus wide open 3's plus open 3's 2022-23")
plt.ylabel("Defensive Rating 2022-23")
plt.text(83, 111, "correlation: "+str(int(stats.pearsonr(opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A']+opponent_shooting_23['OPEN_FG3A'], opponent_shooting_23['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.show()

plt.scatter(opponent_shooting['OPP_PTS_PAINT']+opponent_shooting['WIDE_OPEN_FG3A']*.36+opponent_shooting['OPEN_FG3A']*.98, opponent_shooting['DEF_RATING'])
plt.xlabel("Opponent points in the paint plus wide open 3's*.36 plus open 3's*.98 2023-24")
plt.ylabel("Defensive Rating 2023-24")
plt.text(72, 111, "correlation: "+str(int(stats.pearsonr(opponent_shooting['OPP_PTS_PAINT']+opponent_shooting['WIDE_OPEN_FG3A']*.36+opponent_shooting['OPEN_FG3A']*.98, opponent_shooting['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.show()

plt.scatter(opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A']*.36+opponent_shooting_23['OPEN_FG3A']*.98, opponent_shooting_23['DEF_RATING'])
plt.xlabel("Opponent points in the paint plus wide open 3's*.36 plus open 3's*.98 2022-23")
plt.ylabel("Defensive Rating 2022-23")
plt.text(72, 111, "correlation: "+str(int(stats.pearsonr(opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A']*.36+opponent_shooting_23['OPEN_FG3A']*.98, opponent_shooting_23['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.show()


both_years = pd.concat([opponent_shooting, opponent_shooting_23])

plt.scatter(opponent_shooting['OPP_PTS_PAINT'], opponent_shooting['DEF_RATING'], color='red', label='2023-24')
plt.scatter(opponent_shooting_23['OPP_PTS_PAINT'], opponent_shooting_23['DEF_RATING'], color='blue', label='2022-23')
plt.xlabel('Opponent points in the paint')
plt.ylabel('Defensive Rating')
plt.text(55, 110, "correlation: "+str(int(stats.pearsonr(both_years['OPP_PTS_PAINT'], both_years['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.legend()
plt.show()

plt.scatter(opponent_shooting['OPP_PTS_PAINT']+opponent_shooting['WIDE_OPEN_FG3A']+opponent_shooting['OPEN_FG3A'], opponent_shooting['DEF_RATING'], color='red', label='2023-24')
plt.scatter(opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A']+opponent_shooting_23['OPEN_FG3A'], opponent_shooting_23['DEF_RATING'], color='blue', label='2022-23')
plt.xlabel("Opponent points in the paint plus wide open 3's plus open 3's")
plt.ylabel('Defensive Rating')
plt.text(85, 110, "correlation: "+str(int(stats.pearsonr(both_years['OPP_PTS_PAINT']+both_years['WIDE_OPEN_FG3A']+both_years['OPEN_FG3A'], both_years['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.legend()
plt.show()

plt.scatter(opponent_shooting['OPP_PTS_PAINT']+opponent_shooting['WIDE_OPEN_FG3A']*.36+opponent_shooting['OPEN_FG3A']*.98, opponent_shooting['DEF_RATING'], color='red', label='2023-24')
plt.scatter(opponent_shooting_23['OPP_PTS_PAINT']+opponent_shooting_23['WIDE_OPEN_FG3A']*.36+opponent_shooting_23['OPEN_FG3A']*.98, opponent_shooting_23['DEF_RATING'], color='blue', label='2022-23')
plt.xlabel("Opponent points in the paint plus wide open 3's*.36 plus open 3's*.98")
plt.ylabel('Defensive Rating')
plt.text(74, 110, "correlation: "+str(int(stats.pearsonr(both_years['OPP_PTS_PAINT']+both_years['WIDE_OPEN_FG3A']*.36+both_years['OPEN_FG3A']*.98, both_years['DEF_RATING']).statistic*1000)/1000), fontsize=8, color='black')
plt.legend()
plt.show()




print('\n')
#regression to predict defensive rating using opponent points in the paint with test data from the 2022-23 season
X = opponent_shooting[['OPP_PTS_PAINT']]
Y = opponent_shooting['DEF_RATING']

X_test = opponent_shooting_23[['OPP_PTS_PAINT']]
Y_test = opponent_shooting_23['DEF_RATING']

model = LinearRegression().fit(X, Y)
print('regression using opponent points in the paint to predict defensive rating')
print('score:', model.score(X, Y))
print('Coefficients:', model.coef_)
print('Y-Intercept:', model.intercept_)
pred_Y = model.predict(X)
mse = mean_squared_error(Y, pred_Y)
print('Mean absolute error:', np.sqrt(mse))
print('test score: ', model.score(X_test, Y_test))
print('\n')
i = 0
print('2023-24 predicted defensive ratings vs actual:')
for prediction in pred_Y:
	print(opponent_shooting.iloc[i, 1], '-', 'predicted:', round(prediction,1), 'actual:', opponent_shooting.iloc[i, 7])
	i+=1
print('\n')



#regression to predict defensive rating using opponent points in the paint, wide open 3's allowed, and open 3's allowed with test data from the 2022-23 season
X = opponent_shooting[['OPP_PTS_PAINT', 'WIDE_OPEN_FG3A', 'OPEN_FG3A']]
X_test = opponent_shooting_23[['OPP_PTS_PAINT', 'WIDE_OPEN_FG3A', 'OPEN_FG3A']]

model = LinearRegression().fit(X, Y)
print("regression using opponent points in the paint, wide open 3's allowed, and open 3's allowed to predict defensive rating")
print('score:', model.score(X, Y))
print('Coefficients:', model.coef_)
print('Y-Intercept:', model.intercept_)
pred_Y = model.predict(X)
mse = mean_squared_error(Y, pred_Y)
print('Mean absolute error:', np.sqrt(mse))
print('test score: ', model.score(X_test, Y_test))
print('\n')
i = 0
print('2023-24 predicted defensive ratings vs actual:')
for prediction in pred_Y:
	print(opponent_shooting.iloc[i, 1], '-', 'predicted:', round(prediction,1), 'actual:', opponent_shooting.iloc[i, 7])
	i+=1
print("\n")

pred_test_Y = model.predict(X_test)
average_error = (pred_test_Y-Y_test).mean() #2023-24 model predicts that 2022-23 defensive ratings are .61 higher on average than they actually were
mse = mean_squared_error(Y_test, pred_test_Y)
i = 0
print('2022-23 predicted defensive ratings vs actual:')
for prediction in pred_test_Y:
	print(opponent_shooting_23.iloc[i, 1], '-', 'predicted:', round(prediction,1), 'actual:', opponent_shooting_23.iloc[i, 7])
	i+=1
print('2022-23 mean absolute error:', np.sqrt(mse))

#doesn't generalize well, not training on enough data. also difficult to account for league-wide changes from year-to-year

