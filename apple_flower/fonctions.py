import numpy as np
def P(x , a,b, deg):
    p = np.polyfit(a,b,deg)
    res = p[deg]
    print(res)
    for i in range(deg) :
        res+= p[i]*pow(x,deg-i)
    return  res

def ContourXY(contourCoord):
    x=[]
    y=[]

    for i in range(len(contourCoord)):
        ax =contourCoord[i][0][1]
        x.append( contourCoord[i][0][0])
        y.append( contourCoord[i][0][1])
    return (x,y)

def partitionement(contour, points) :
    parts = []
    end= points[0]
    for i in range(len(points)-1) :
        start = points[i]
        end = points[i+1]
        p = contour[start:end]
        parts.append(p)
    return parts


def divided_diff(x, y):
  
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = \
           (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef

def newton_poly(coef, x_data, x):
  
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p

def curvature_based_segmentation(contour):
  curvature = np.zeros(len(contour))
  for i in range(len(contour)):
    prev_pt = contour[(i-1)%len(contour)]
    curr_pt = contour[i]
    next_pt = contour[(i+1)%len(contour)]
    
    # Calculate the curvature using the angle between the previous, current, and next points
    angle = np.arccos(np.dot(next_pt-curr_pt, prev_pt-curr_pt) / 
                      (np.linalg.norm(next_pt-curr_pt) * np.linalg.norm(prev_pt-curr_pt)))
    curvature[i] = angle
  
  # Find the peaks in the curvature
  peaks = []
  for i in range(1, len(contour)-1):
    if curvature[i] > curvature[i-1] and curvature[i] > curvature[i+1]:
      peaks.append(i)
      
  # Partition the contour into segments based on the peaks in the curvature
  segments = []
  start = 0
  for i in range(len(peaks)):
    end = peaks[i]
    segments.append(contour[start:end])
    start = end
  segments.append(contour[start:])
  
  return segments

from scipy.optimize import least_squares

def residuals(params, x, y):
    f = np.poly1d(params)
    return f(x) - y

def least_squares_approximation(x, y, degree):
    initial_guess = np.zeros(degree + 1)
    results = least_squares(residuals, initial_guess, args=(x, y))
    return results.x


def calculate_error_norms(original_segment, approximated_curve):
    error = original_segment - approximated_curve
    L1_norm = np.sum(np.abs(error))
    L2_norm = np.sqrt(np.sum(error**2))
    Linf_norm = np.max(np.abs(error))
    return L1_norm, L2_norm, Linf_norm
