#################################################
# logRegression: Logistic Regression
# Author : zouxy
# Date   : 2014-03-02
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import matplotlib.pyplot as plt
import time


# calculate the sigmoid function
def sigmoid(inX):
	return 1.0 / (1 + exp(-inX))


# train a logistic regression model using some optional optimize algorithm
# input: train_x is a mat datatype, each row stands for one sample
#		 train_y is mat datatype too, each row is the corresponding label
#		 opts is optimize option include step and maximum number of iterations
def trainLogRegres(train_x, train_y, opts):
	# calculate training time
	startTime = time.time()

	numSamples, numFeatures = shape(train_x)
	alpha = opts['alpha']; maxIter = opts['maxIter']
	weights = ones((numFeatures, 1))

	# optimize through gradient descent algorilthm
	for k in range(maxIter):
		if opts['optimizeType'] == 'gradDescent': # gradient descent algorilthm
			output = sigmoid(train_x * weights)
			error = train_y - output
			weights = weights + alpha * train_x.transpose() * error
		elif opts['optimizeType'] == 'stocGradDescent': # stochastic gradient descent
			for i in range(numSamples):
				output = sigmoid(train_x[i, :] * weights)
				error = train_y[i, 0] - output
				weights = weights + alpha * train_x[i, :].transpose() * error
		elif opts['optimizeType'] == 'smoothStocGradDescent': # smooth stochastic gradient descent
			# randomly select samples to optimize for reducing cycle fluctuations
			dataIndex = range(numSamples)
			for i in range(numSamples):
				alpha = 4.0 / (1.0 + k + i) + 0.01
				randIndex = int(random.uniform(0, len(dataIndex)))
				output = sigmoid(train_x[randIndex, :] * weights)
				error = train_y[randIndex, 0] - output
				weights = weights + alpha * train_x[randIndex, :].transpose() * error
				del(dataIndex[randIndex]) # during one interation, delete the optimized sample
		else:
			raise NameError('Not support optimize method type!')


	print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
	return weights


# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
	numSamples, numFeatures = shape(test_x)
	matchCount = 0
	for i in xrange(numSamples):
		predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
		if predict == bool(test_y[i, 0]):
			matchCount += 1
	accuracy = float(matchCount) / numSamples
	return accuracy


# show your trained logistic regression model only available with 2-D data
def showLogRegres(weights, train_x, train_y):
	# notice: train_x and train_y is mat datatype
	numSamples, numFeatures = shape(train_x)
	if numFeatures != 3:
		print "Sorry! I can not draw because the dimension of your data is not 2!"
		return 1

	# draw all samples
	for i in xrange(numSamples):
		if int(train_y[i, 0]) == 0:
			plt.plot(train_x[i, 1], train_x[i, 2], 'or')
		elif int(train_y[i, 0]) == 1:
			plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

	# draw the classify line
	min_x = min(train_x[:, 1])[0, 0]
	max_x = max(train_x[:, 1])[0, 0]
	weights = weights.getA()  # convert mat to array
	y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
	y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
	plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()

#################################################
# logRegression: Logistic Regression
# Author : zouxy
# Date   : 2014-03-02
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import matplotlib.pyplot as plt
import time

def loadData():
	train_x = []
	train_y = []
	fileIn = open('E:/Python/Machine Learning in Action/testSet.txt')
	for line in fileIn.readlines():
		lineArr = line.strip().split()
		train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
		train_y.append(float(lineArr[2]))
	return mat(train_x), mat(train_y).transpose()


## step 1: load data
print "step 1: load data..."
train_x, train_y = loadData()
test_x = train_x; test_y = train_y

## step 2: training...
print "step 2: training..."
opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
optimalWeights = trainLogRegres(train_x, train_y, opts)

## step 3: testing
print "step 3: testing..."
accuracy = testLogRegres(optimalWeights, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
showLogRegres(optimalWeights, train_x, train_y)

'''
-0.017612	14.053064	0
-1.395634	4.662541	1
-0.752157	6.538620	0
-1.322371	7.152853	0
0.423363	11.054677	0
0.406704	7.067335	1
0.667394	12.741452	0
-2.460150	6.866805	1
0.569411	9.548755	0
-0.026632	10.427743	0
0.850433	6.920334	1
1.347183	13.175500	0
1.176813	3.167020	1
-1.781871	9.097953	0
-0.566606	5.749003	1
0.931635	1.589505	1
-0.024205	6.151823	1
-0.036453	2.690988	1
-0.196949	0.444165	1
1.014459	5.754399	1
1.985298	3.230619	1
-1.693453	-0.557540	1
-0.576525	11.778922	0
-0.346811	-1.678730	1
-2.124484	2.672471	1
1.217916	9.597015	0
-0.733928	9.098687	0
-3.642001	-1.618087	1
0.315985	3.523953	1
1.416614	9.619232	0
-0.386323	3.989286	1
0.556921	8.294984	1
1.224863	11.587360	0
-1.347803	-2.406051	1
1.196604	4.951851	1
0.275221	9.543647	0
0.470575	9.332488	0
-1.889567	9.542662	0
-1.527893	12.150579	0
-1.185247	11.309318	0
-0.445678	3.297303	1
1.042222	6.105155	1
-0.618787	10.320986	0
1.152083	0.548467	1
0.828534	2.676045	1
-1.237728	10.549033	0
-0.683565	-2.166125	1
0.229456	5.921938	1
-0.959885	11.555336	0
0.492911	10.993324	0
0.184992	8.721488	0
-0.355715	10.325976	0
-0.397822	8.058397	0
0.824839	13.730343	0
1.507278	5.027866	1
0.099671	6.835839	1
-0.344008	10.717485	0
1.785928	7.718645	1
-0.918801	11.560217	0
-0.364009	4.747300	1
-0.841722	4.119083	1
0.490426	1.960539	1
-0.007194	9.075792	0
0.356107	12.447863	0
0.342578	12.281162	0
-0.810823	-1.466018	1
2.530777	6.476801	1
1.296683	11.607559	0
0.475487	12.040035	0
-0.783277	11.009725	0
0.074798	11.023650	0
-1.337472	0.468339	1
-0.102781	13.763651	0
-0.147324	2.874846	1
0.518389	9.887035	0
1.015399	7.571882	0
-1.658086	-0.027255	1
1.319944	2.171228	1
2.056216	5.019981	1
-0.851633	4.375691	1
-1.510047	6.061992	0
-1.076637	-3.181888	1
1.821096	10.283990	0
3.010150	8.401766	1
-1.099458	1.688274	1
-0.834872	-1.733869	1
-0.846637	3.849075	1
1.400102	12.628781	0
1.752842	5.468166	1
0.078557	0.059736	1
0.089392	-0.715300	1
1.825662	12.693808	0
0.197445	9.744638	0
0.126117	0.922311	1
-0.679797	1.220530	1
0.677983	2.556666	1
0.761349	10.693862	0
-2.168791	0.143632	1
1.388610	9.341997	0
0.317029	14.739025	0
'''


#################################################
# kmeans: k-means cluster
# Author : zouxy
# Date   : 2013-12-25
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import time
import matplotlib.pyplot as plt


# calculate Euclidean distance
def euclDistance(vector1, vector2):
	return sqrt(sum(power(vector2 - vector1, 2)))

# init centroids with random samples
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids

# k-means cluster
def kmeans(dataSet, k):
	numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
	clusterAssment = mat(zeros((numSamples, 2)))
	clusterChanged = True

	## step 1: init centroids
	centroids = initCentroids(dataSet, k)

	while clusterChanged:
		clusterChanged = False
		## for each sample
		for i in xrange(numSamples):
			minDist  = 100000.0
			minIndex = 0
			## for each centroid
			## step 2: find the centroid who is closest
			for j in range(k):
				distance = euclDistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist  = distance
					minIndex = j

			## step 3: update its cluster
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist**2

		## step 4: update centroids
		for j in range(k):
			pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] = mean(pointsInCluster, axis = 0)

	print 'Congratulations, cluster complete!'
	return centroids, clusterAssment

# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
	numSamples, dim = dataSet.shape
	if dim != 2:
		print "Sorry! I can not draw because the dimension of your data is not 2!"
		return 1

	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	if k > len(mark):
		print "Sorry! Your k is too large! please contact Zouxy"
		return 1

	# draw all samples
	for i in xrange(numSamples):
		markIndex = int(clusterAssment[i, 0])
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	# draw the centroids
	for i in range(k):
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)

	plt.show()

#################################################
# kmeans: k-means cluster
# Author : zouxy
# Date   : 2013-12-25
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import time
import matplotlib.pyplot as plt

## step 1: load data
print "step 1: load data..."
dataSet = []
fileIn = open('E:/Python/Machine Learning in Action/testSet.txt')
for line in fileIn.readlines():
	lineArr = line.strip().split('\t')
	dataSet.append([float(lineArr[0]), float(lineArr[1])])

## step 2: clustering...
print "step 2: clustering..."
dataSet = mat(dataSet)
k = 4
centroids, clusterAssment = kmeans(dataSet, k)

## step 3: show the result
print "step 3: show the result..."
showCluster(dataSet, k, centroids, clusterAssment)
'''
1.658985	4.285136
-3.453687	3.424321
4.838138	-1.151539
-5.379713	-3.362104
0.972564	2.924086
-3.567919	1.531611
0.450614	-3.302219
-3.487105	-1.724432
2.668759	1.594842
-3.156485	3.191137
3.165506	-3.999838
-2.786837	-3.099354
4.208187	2.984927
-2.123337	2.943366
0.704199	-0.479481
-0.392370	-3.963704
2.831667	1.574018
-0.790153	3.343144
2.943496	-3.357075
-3.195883	-2.283926
2.336445	2.875106
-1.786345	2.554248
2.190101	-1.906020
-3.403367	-2.778288
1.778124	3.880832
-1.688346	2.230267
2.592976	-2.054368
-4.007257	-3.207066
2.257734	3.387564
-2.679011	0.785119
0.939512	-4.023563
-3.674424	-2.261084
2.046259	2.735279
-3.189470	1.780269
4.372646	-0.822248
-2.579316	-3.497576
1.889034	5.190400
-0.798747	2.185588
2.836520	-2.658556
-3.837877	-3.253815
2.096701	3.886007
-2.709034	2.923887
3.367037	-3.184789
-2.121479	-4.232586
2.329546	3.179764
-3.284816	3.273099
3.091414	-3.815232
-3.762093	-2.432191
3.542056	2.778832
-1.736822	4.241041
2.127073	-2.983680
-4.323818	-3.938116
3.792121	5.135768
-4.786473	3.358547
2.624081	-3.260715
-4.009299	-2.978115
2.493525	1.963710
-2.513661	2.642162
1.864375	-3.176309
-3.171184	-3.572452
2.894220	2.489128
-2.562539	2.884438
3.491078	-3.947487
-2.565729	-2.012114
3.332948	3.983102
-1.616805	3.573188
2.280615	-2.559444
-2.651229	-3.103198
2.321395	3.154987
-1.685703	2.939697
3.031012	-3.620252
-4.599622	-2.185829
4.196223	1.126677
-2.133863	3.093686
4.668892	-2.562705
-2.793241	-2.149706
2.884105	3.043438
-2.967647	2.848696
4.479332	-1.764772
-4.905566	-2.911070
'''

#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import time
import matplotlib.pyplot as plt


# calulate kernel value
def calcKernelValue(matrix_x, sample_x, kernelOption):
	kernelType = kernelOption[0]
	numSamples = matrix_x.shape[0]
	kernelValue = mat(zeros((numSamples, 1)))

	if kernelType == 'linear':
		kernelValue = matrix_x * sample_x.T
	elif kernelType == 'rbf':
		sigma = kernelOption[1]
		if sigma == 0:
			sigma = 1.0
		for i in xrange(numSamples):
			diff = matrix_x[i, :] - sample_x
			kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma**2))
	else:
		raise NameError('Not support kernel type! You can use linear or rbf!')
	return kernelValue


# calculate kernel matrix given train set and kernel type
def calcKernelMatrix(train_x, kernelOption):
	numSamples = train_x.shape[0]
	kernelMatrix = mat(zeros((numSamples, numSamples)))
	for i in xrange(numSamples):
		kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
	return kernelMatrix


# define a struct just for storing variables and data
class SVMStruct:
	def __init__(self, dataSet, labels, C, toler, kernelOption):
		self.train_x = dataSet # each row stands for a sample
		self.train_y = labels  # corresponding label
		self.C = C             # slack variable
		self.toler = toler     # termination condition for iteration
		self.numSamples = dataSet.shape[0] # number of samples
		self.alphas = mat(zeros((self.numSamples, 1))) # Lagrange factors for all samples
		self.b = 0
		self.errorCache = mat(zeros((self.numSamples, 2)))
		self.kernelOpt = kernelOption
		self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)


# calculate the error for alpha k
def calcError(svm, alpha_k):
	output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)
	error_k = output_k - float(svm.train_y[alpha_k])
	return error_k


# update the error cache for alpha k after optimize alpha k
def updateError(svm, alpha_k):
	error = calcError(svm, alpha_k)
	svm.errorCache[alpha_k] = [1, error]


# select alpha j which has the biggest step
def selectAlpha_j(svm, alpha_i, error_i):
	svm.errorCache[alpha_i] = [1, error_i] # mark as valid(has been optimized)
	candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0] # mat.A return array
	maxStep = 0; alpha_j = 0; error_j = 0

	# find the alpha with max iterative step
	if len(candidateAlphaList) > 1:
		for alpha_k in candidateAlphaList:
			if alpha_k == alpha_i:
				continue
			error_k = calcError(svm, alpha_k)
			if abs(error_k - error_i) > maxStep:
				maxStep = abs(error_k - error_i)
				alpha_j = alpha_k
				error_j = error_k
	# if came in this loop first time, we select alpha j randomly
	else:
		alpha_j = alpha_i
		while alpha_j == alpha_i:
			alpha_j = int(random.uniform(0, svm.numSamples))
		error_j = calcError(svm, alpha_j)

	return alpha_j, error_j


# the inner loop for optimizing alpha i and alpha j
def innerLoop(svm, alpha_i):
	error_i = calcError(svm, alpha_i)

	### check and pick up the alpha who violates the KKT condition
	## satisfy KKT condition
	# 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
	# 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
	# 3) yi*f(i) <= 1 and alpha == C (between the boundary)
	## violate KKT condition
	# because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
	# 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
	# 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
	# 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
	if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or\
		(svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

		# step 1: select alpha j
		alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
		alpha_i_old = svm.alphas[alpha_i].copy()
		alpha_j_old = svm.alphas[alpha_j].copy()

		# step 2: calculate the boundary L and H for alpha j
		if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
			L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
			H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
		else:
			L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
			H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
		if L == H:
			return 0

		# step 3: calculate eta (the similarity of sample i and j)
		eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] \
				  - svm.kernelMat[alpha_j, alpha_j]
		if eta >= 0:
			return 0

		# step 4: update alpha j
		svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

		# step 5: clip alpha j
		if svm.alphas[alpha_j] > H:
			svm.alphas[alpha_j] = H
		if svm.alphas[alpha_j] < L:
			svm.alphas[alpha_j] = L

		# step 6: if alpha j not moving enough, just return
		if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
			updateError(svm, alpha_j)
			return 0

		# step 7: update alpha i after optimizing aipha j
		svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
								* (alpha_j_old - svm.alphas[alpha_j])

		# step 8: update threshold b
		b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
													* svm.kernelMat[alpha_i, alpha_i] \
							 - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
													* svm.kernelMat[alpha_i, alpha_j]
		b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
													* svm.kernelMat[alpha_i, alpha_j] \
							 - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
													* svm.kernelMat[alpha_j, alpha_j]
		if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
			svm.b = b1
		elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
			svm.b = b2
		else:
			svm.b = (b1 + b2) / 2.0

		# step 9: update error cache for alpha i, j after optimize alpha i, j and b
		updateError(svm, alpha_j)
		updateError(svm, alpha_i)

		return 1
	else:
		return 0


# the main training procedure
def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('rbf', 1.0)):
	# calculate training time
	startTime = time.time()

	# init data struct for svm
	svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)

	# start training
	entireSet = True
	alphaPairsChanged = 0
	iterCount = 0
	# Iteration termination condition:
	# 	Condition 1: reach max iteration
	# 	Condition 2: no alpha changed after going through all samples,
	# 				 in other words, all alpha (samples) fit KKT condition
	while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
		alphaPairsChanged = 0

		# update alphas over all training examples
		if entireSet:
			for i in xrange(svm.numSamples):
				alphaPairsChanged += innerLoop(svm, i)
			print '---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)
			iterCount += 1
		# update alphas over examples where alpha is not 0 & not C (not on boundary)
		else:
			nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
			for i in nonBoundAlphasList:
				alphaPairsChanged += innerLoop(svm, i)
			print '---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)
			iterCount += 1

		# alternate loop over all examples and non-boundary examples
		if entireSet:
			entireSet = False
		elif alphaPairsChanged == 0:
			entireSet = True

	print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
	return svm


# testing your trained svm model given test set
def testSVM(svm, test_x, test_y):
	test_x = mat(test_x)
	test_y = mat(test_y)
	numTestSamples = test_x.shape[0]
	supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
	supportVectors 		= svm.train_x[supportVectorsIndex]
	supportVectorLabels = svm.train_y[supportVectorsIndex]
	supportVectorAlphas = svm.alphas[supportVectorsIndex]
	matchCount = 0
	for i in xrange(numTestSamples):
		kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.kernelOpt)
		predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + svm.b
		if sign(predict) == sign(test_y[i]):
			matchCount += 1
	accuracy = float(matchCount) / numTestSamples
	return accuracy


# show your trained svm model only available with 2-D data
def showSVM(svm):
	if svm.train_x.shape[1] != 2:
		print "Sorry! I can not draw because the dimension of your data is not 2!"
		return 1

	# draw all samples
	for i in xrange(svm.numSamples):
		if svm.train_y[i] == -1:
			plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'or')
		elif svm.train_y[i] == 1:
			plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob')

	# mark support vectors
	supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
	for i in supportVectorsIndex:
		plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')

	# draw the classify line
	w = zeros((2, 1))
	for i in supportVectorsIndex:
		w += multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T)
	min_x = min(svm.train_x[:, 0])[0, 0]
	max_x = max(svm.train_x[:, 0])[0, 0]
	y_min_x = float(-svm.b - w[0] * min_x) / w[1]
	y_max_x = float(-svm.b - w[0] * max_x) / w[1]
	plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
	plt.show()
#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import SVM

################## test svm #####################
## step 1: load data
print "step 1: load data..."
dataSet = []
labels = []
fileIn = open('E:/Python/Machine Learning in Action/testSet.txt')
for line in fileIn.readlines():
	lineArr = line.strip().split('\t')
	dataSet.append([float(lineArr[0]), float(lineArr[1])])
	labels.append(float(lineArr[2]))

dataSet = mat(dataSet)
labels = mat(labels).T
train_x = dataSet[0:81, :]
train_y = labels[0:81, :]
test_x = dataSet[80:101, :]
test_y = labels[80:101, :]

## step 2: training...
print "step 2: training..."
C = 0.6
toler = 0.001
maxIter = 50
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))

## step 3: testing
print "step 3: testing..."
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
SVM.showSVM(svmClassifier)


#########################################
# kNN: k Nearest Neighbors

# Input:      newInput: vector to compare to existing dataset (1xN)
#             dataSet:  size m data set of known vectors (NxM)
#             labels: 	data set labels (1xM vector)
#             k: 		number of neighbors to use for comparison

# Output:     the most popular class label
#########################################

from numpy import *
import operator

# create a dataset which contains 4 samples with 2 classes
def createDataSet():
	# create a matrix: each row as a sample
	group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
	labels = ['A', 'A', 'B', 'B'] # four samples and two classes
	return group, labels

# classify using kNN
def kNNClassify(newInput, dataSet, labels, k):
	numSamples = dataSet.shape[0] # shape[0] stands for the num of row

	## step 1: calculate Euclidean distance
	# tile(A, reps): Construct an array by repeating A reps times
	# the following copy numSamples rows for dataSet
	diff = tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise
	squaredDiff = diff ** 2 # squared for the subtract
	squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row
	distance = squaredDist ** 0.5

	## step 2: sort the distance
	# argsort() returns the indices that would sort an array in a ascending order
	sortedDistIndices = argsort(distance)

	classCount = {} # define a dictionary (can be append element)
	for i in xrange(k):
		## step 3: choose the min k distance
		voteLabel = labels[sortedDistIndices[i]]

		## step 4: count the times labels occur
		# when the key voteLabel is not in dictionary classCount, get()
		# will return 0
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

	## step 5: the max voted class will return
	maxCount = 0
	for key, value in classCount.items():
		if value > maxCount:
			maxCount = value
			maxIndex = key

	return maxIndex

import kNN
from numpy import *

dataSet, labels = kNN.createDataSet()

testX = array([1.2, 1.0])
k = 3
outputLabel = kNN.kNNClassify(testX, dataSet, labels, 3)
print "Your input is:", testX, "and classified to class: ", outputLabel

testX = array([0.1, 0.3])
outputLabel = kNN.kNNClassify(testX, dataSet, labels, 3)
print "Your input is:", testX, "and classified to class: ", outputLabel



###########################################################

# Example of Naive Bayes implemented from Scratch in Python
###########################################################
import csv
import random
import math

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	filename = 'pima-indians-diabetes.data.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()
