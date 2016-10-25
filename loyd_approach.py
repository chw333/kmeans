import math
import numpy as np
import plotly as plotly
import plotly.plotly as py
from plotly.graph_objs import *


def CalculateDistance(p1, p2):
    xDiff = (p2[0, 0] - p1[0, 0]) ** 2
    yDiff = (p2[0, 1] - p2[0, 1]) ** 2
    return math.sqrt(xDiff + yDiff)


def CreateScatterTrace(data, color, marker='point', size=10):
    trace = Scatter(
        x=data[:, 0],
        y=data[:, 1],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            symbol=marker,
            line=dict(
                width=2)
        )
    )
    return trace


def Plot2Categories(cat1Data, cat2Data):
    traceCat1 = CreateScatterTrace(cat1Data, 'rgba(152, 0, 0, .8)')
    traceCat2 = CreateScatterTrace(cat2Data, 'rgba(0, 152, 0, .8)')
    py.plot([traceCat1, traceCat2], filename='blog/twocategories')
    # plotly.offline.plot({
    #    "data": [traceCat1, traceCat2],
    #    "layout": Layout(title="2 Fake Categories")
    # })


def PlotSingleCategory(data):
    trace = CreateScatterTrace(data, 'rgba(152, 0, 0, .8)')
    plotly.offline.plot({
        "data": [trace],
        "layout": Layout(title="2 Fake Categories")
    })


def PlotKMeansSingle(data, cen1, cen2):
    traceCat1 = CreateScatterTrace(data, 'rgba(0, 0, 152, .8)')
    traceCen1 = CreateScatterTrace(cen1, 'rgba(152, 0, 0, .8)', 'star', 15)
    traceCen2 = CreateScatterTrace(cen2, 'rgba(0, 152, 0, .8)', 'star', 15)
    plotly.offline.plot({
        "data": [traceCat1, traceCen1, traceCen2],
        "layout": Layout(title="KMeans")
    })


def PlotKMeans(cat1Data, cat2Data, cen1, cen2):
    traceCat1 = CreateScatterTrace(cat1Data, 'rgba(152, 0, 0, .8)')
    traceCat2 = CreateScatterTrace(cat2Data, 'rgba(0, 152, 0, .8)')
    traceCen1 = CreateScatterTrace(cen1, 'rgba(152, 0, 0, .8)', 'star', 15)
    traceCen2 = CreateScatterTrace(cen2, 'rgba(0, 152, 0, .8)', 'star', 15)
    py.plot([traceCat1, traceCat2, traceCen1, traceCen2], filename='blog/KMeansPlot')
    # plotly.offline.plot({
    #    "data": [traceCat1, traceCat2, traceCen1, traceCen2],
    #    "layout": Layout(title="KMeans")
    # })


cat1 = 5.5
cat2 = 3.5
# Create 2 A-Symmetrical Gaussian Distributions
cat1Data = np.column_stack((np.random.normal(cat1, 1.0, 50),
                            np.random.normal(cat1, 1.0, 50)))
cat2Data = np.column_stack((np.random.normal(cat2, 1.0, 50),
                            np.random.normal(cat2, 1.0, 50)))
# Plot 2 categories known
Plot2Categories(cat1Data, cat2Data)

# merge categories and display as k-means would see them.
mergedData = np.concatenate((cat1Data, cat2Data))
PlotSingleCategory(mergedData)

###################
# K-Means Round 1 #
###################
# pick 2 initial centroids intelligently
xSplit = np.percentile(mergedData[:, 0], 50.0)
ySplit = np.percentile(mergedData[:, 1], 50.0)
xStd = np.std(mergedData[:, 0])
yStd = np.std(mergedData[:, 1])
cat1Cen = np.array([xSplit - xStd, ySplit - yStd]).reshape([1, 2])
cat2Cen = np.array([xSplit + xStd, ySplit + yStd]).reshape([1, 2])
# Do dumber centroid selection, for longer convergence
# This is for demonstration
cat1Cen = np.array([1, 1]).reshape([1, 2])
cat2Cen = np.array([7, 7]).reshape([1, 2])

# Plot before category assignments
PlotKMeansSingle(mergedData, cat1Cen, cat2Cen)

# assign points to categories
cat1 = np.empty([2, ])
cat2 = np.empty([2, ])
for i in range(0, mergedData[:, 0].size):
    distToCen1 = CalculateDistance(cat1Cen, mergedData[i, :].reshape([1, 2]))
    distToCen2 = CalculateDistance(cat2Cen, mergedData[i, :].reshape([1, 2]))
    if distToCen1 < distToCen2:
        cat1 = np.concatenate((cat1, mergedData[i, :]))
    else:
        cat2 = np.concatenate((cat2, mergedData[i, :]))
# endfor
# delete first row, as it contains zeros from initialization
# Also just reshape while we are at it.
cat1 = np.delete(cat1, (0, 1))
cat1 = cat1.reshape([cat1.size / 2, 2])
cat2 = np.delete(cat2, (0, 1))
cat2 = cat2.reshape([cat2.size / 2, 2])

# Plot first round
PlotKMeans(cat1, cat2, cat1Cen, cat2Cen)

###################
# K-Means Round 2 #
###################
# Move centroids to average of data they have
cat1Cen = np.array([np.mean(cat1[:, 0]), np.mean(cat1[:, 1])]).reshape([1, 2])
cat2Cen = np.array([np.mean(cat2[:, 0]), np.mean(cat2[:, 1])]).reshape([1, 2])
# Plot before re-assignment
PlotKMeans(cat1, cat2, cat1Cen, cat2Cen)
# Re-assign
mergedData = np.concatenate((cat1, cat2))
cat1 = np.empty([2, ])
cat2 = np.empty([2, ])
for i in range(0, mergedData[:, 0].size):
    distToCen1 = CalculateDistance(cat1Cen, mergedData[i, :].reshape([1, 2]))
    distToCen2 = CalculateDistance(cat2Cen, mergedData[i, :].reshape([1, 2]))
    if distToCen1 < distToCen2:
        cat1 = np.concatenate((cat1, mergedData[i, :]))
    else:
        cat2 = np.concatenate((cat2, mergedData[i, :]))
# endfor
cat1 = np.delete(cat1, (0, 1))
cat1 = cat1.reshape([cat1.size / 2, 2])
cat2 = np.delete(cat2, (0, 1))
cat2 = cat2.reshape([cat2.size / 2, 2])
# plot after
PlotKMeans(cat1, cat2, cat1Cen, cat2Cen)
