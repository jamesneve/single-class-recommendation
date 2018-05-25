# -*- coding: UTF-8 -*-

import numpy as np
import math

delta = 0.1
l = 0.2
d = 2

def make_uniform_weight_matrix(data_matrix_r):
  w = np.zeros((len(data_matrix_r), len(data_matrix_r[0])))
  for i, u in enumerate(data_matrix_r):
    for j, v in enumerate(u):
      if v == 1:
        w[i][j] = 1
      else:
        w[i][j] = delta

  return w

def rank(r):
  return np.linalg.matrix_rank(r)

def init_u(data_matrix_r):
  return np.random.normal(0, 0.01, (len(data_matrix_r), d))

def init_v(data_matrix_r):
  return np.random.normal(0, 0.01, (len(data_matrix_r[0]), d))

def update_u(r, w, u, v):
  nextu = np.zeros(u.shape)
  for i, ui in enumerate(u):
    ri = np.zeros((len(r), len(r[0])))
    ri[i] = r[i]
    wdashi = make_wdashi(w[i])
    riwdashi = np.matmul(ri, wdashi)
    riwdashiv = np.matmul(riwdashi, v)  # First term
    
    vt = np.transpose(v)
    vtwdashi = np.matmul(vt, wdashi)
    vtwdashiv = np.matmul(vtwdashi, v)  # Second term
    
    lambdawi = make_lambdawi(w[i])  # Third term

    spdm = vtwdashiv + lambdawi
    ispdm = np.linalg.inv(spdm)

    res = np.matmul(riwdashiv, ispdm)
    nextu[i] = res[i]

  return nextu

def update_v(r, w, u, v):
  nextv = np.zeros(v.shape)
  for j, vj in enumerate(v):
    rj = np.zeros((len(r[0]), len(r)))
    rj[j] = np.transpose(r)[j]
    wdashj = make_wdashi(np.transpose(w)[j])
    rjwdashj = np.matmul(rj, wdashj)
    rjwdashju = np.matmul(rjwdashj, u)  # First term

    ut = np.transpose(u)
    utwdashj = np.matmul(ut, wdashj)
    utwdashju = np.matmul(utwdashj, u)  # Second term

    lambdawj = make_lambdawi(np.transpose(w)[j])  # Third term

    spdm = utwdashju + lambdawj
    ispdm = np.linalg.inv(spdm)

    res = np.matmul(rjwdashju, ispdm)
    nextv[j] = res[j]

  return nextv

def make_wdashi(wi):
  wdash = np.zeros((len(wi), len(wi)))
  for i, u in enumerate(wi):
    wdash[i][i] = u

  return wdash

def make_lambdawi(wi):
  wij = np.sum(wi)
  lwij = l * wij

  lambdawi = np.zeros((d, d))
  for i, u in enumerate(lambdawi):
    lambdawi[i][i] = lwij

  return lambdawi

def main():
  r = [[1, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 1, 0, 0, 0], [1, 1, 0, 1, 0], [1, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0]]
  if len(r) == 0 or len(r[0]) == 0:
    print("Empty matrix")
    return

  w = make_uniform_weight_matrix(r)
  u = init_u(r)
  v = init_v(r)

  for i in range(1, 1000):
    u = update_u(r, w, u, v)
    v = update_v(r, w, u, v)

  x = np.matmul(u, np.transpose(v))
  print(x)
  
main()
