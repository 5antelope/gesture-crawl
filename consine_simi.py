import numpy as np

from scipy import spatial

def consine_similarity(x, y):
	a = np.asarray(x).reshape(-1)
	b = np.asarray(y).reshape(-1)

	return spatial.distance.cosine(a, b)