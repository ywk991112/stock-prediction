from solver import Logistic_Solver, MLP_Solver, Random_Forest_Solver
from feature import get_feature

filename = 'data/Combined_News_DJIA.csv'
# features = get_feature('2-gram', filename)
# features = get_feature('tfidf', filename)
features = get_feature('wv', filename)
# solver = Logistic_Solver(features, C=0.8)
solver = MLP_Solver(features, 10)
# solver = Random_Forest_Solver(features)
solver.fit()
print(solver.evaluate())
