#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Edited from
# https://github.com/yuma-m/matplotlib-draggable-plot

import math
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy import interpolate
import numpy as np
from time import gmtime, strftime
from tkinter import filedialog
import csv

class interactive_plot(object):

  def __init__(self):
    self._figure, self._axes = None, None
    self._line, self._sample, self._spline = None, None, None
    self._sample_dt = 3600 # seconds
    self._sample_size = 24*60*60 // self._sample_dt
    self._sp_x = np.linspace(0, 24, 1000)
    self._sa_x = [ (0.5 + i)*self._sample_dt/3600 for i in range(self._sample_size)]
    self._fine_rate = 30
    self._dragging_point = None
    self._points = {}

    self._init_plot()

  def _update_sample_size(self, val):
    self._sample_dt = self._slider_dt.val*60
    self._sample_size = int(24*60*60 / self._sample_dt)
    self._sa_x = [ (0.5 + i)*self._sample_dt/3600 for i in range(self._sample_size)]
    self._update_plot()

  def _update_rate(self, val):
    self._fine_rate = self._slider_rate.val
    self._update_plot()

  def _export_json(self, val):
    print('Config JSON timetable\n{')
    print('  "creation_dt" : {},'.format(self._fine_rate))
    print('  "creation_rate" : {},'.format( [ int(y) for y in self._sa_y ]) )
    #print('  "creation_rate" : {},'.format( [ int(y/self._sample_dt*self._fine_rate) for y in self._sa_y ]) )
    print('}')

  def _reset(self, val):
    self._points = {}
    self._update_plot()

  def _save_shape(self, val):
    if self._points:
      with open( strftime('shape_%Y_%m_%d_%H_%M_%S.txt', gmtime()), 'w') as shapeout:
        shapeout.write('X\tY\n')
        for x, y in self._points.items():
          shapeout.write('{}\t{}\n'.format(x,y))
    else:
      print('No shape to save. Points #{}'.format(len(self._points)))

  def _load_shape(self, val):
    fname = filedialog.askopenfilename(initialdir='./')
    with open(fname) as shape:
      shapedata = csv.reader(shape, delimiter='\t')
      next(shapedata) # skip header
      self._points = {}
      for row in shapedata:
        self._points[ float(row[0])] = float(row[1])
    self._update_plot()


  def _init_plot(self):
    self._figure = plt.figure('Plot editor')
    plt.subplots_adjust(left=0.25, bottom=0.25)
    axes = plt.subplot(1, 1, 1)
    axes.set_xlim(-1, 25)
    axes.set_ylim(-1, 1000)
    axes.set_xticks( [ i*self._sample_dt/3600 for i in range(self._sample_size+1)], minor=False)
    axes.set_xticklabels( [ strftime("%H:%M", gmtime( i*self._sample_dt ) ) for i in range(self._sample_size+1)], fontdict=None, minor=False, rotation=-45 )
    axes.grid(which="both")

    color = 'lightgoldenrodyellow'
    ## Sliders
    slider_h = 0.03
    slider_offset_v = 0.05
    slider_w = 0.65
    slider_offset_h = 0.25
    # coarse dt slider
    self._ax_slider_dt = plt.axes([slider_offset_h, slider_offset_v, slider_w, slider_h], facecolor=color)
    self._slider_dt = Slider(self._ax_slider_dt, 'Coarse dt (min)', 5, 120, valinit=self._sample_dt//60, valstep=5)
    self._slider_dt.on_changed(self._update_sample_size)
    slider_offset_v += 2*slider_h
    # coarse dt slider
    self._ax_slider_rate = plt.axes([slider_offset_h, slider_offset_v, slider_w, slider_h], facecolor=color)
    self._slider_rate = Slider(self._ax_slider_rate, 'Fine dt (sec)', 5, 120, valinit=self._fine_rate, valstep=5)
    self._slider_rate.on_changed(self._update_rate)
    slider_offset_v += 2*slider_h
    ## Buttons
    button_h = 0.05
    button_offset_v = 0.8
    button_w = 0.15
    button_offset_h = 0.025
    # save shape
    self._ax_button_save = plt.axes([button_offset_h, button_offset_v, button_w, button_h], facecolor=color)
    self._button_save = Button(self._ax_button_save, 'Save Shape')
    self._button_save.on_clicked(self._save_shape)
    button_offset_v -= 2*button_h
    # load shape
    self._ax_button_load = plt.axes([button_offset_h, button_offset_v, button_w, button_h], facecolor=color)
    self._button_load = Button(self._ax_button_load, 'Load Shape')
    self._button_load.on_clicked(self._load_shape)
    button_offset_v -= 2*button_h
    # export to json button
    self._ax_button_export = plt.axes([button_offset_h, button_offset_v, button_w, button_h], facecolor=color)
    self._button_export = Button(self._ax_button_export, 'Export JSON')
    self._button_export.on_clicked(self._export_json)
    button_offset_v -= 2*button_h
    # reset button
    self._ax_button_reset = plt.axes([button_offset_h, button_offset_v, button_w, button_h], facecolor=color)
    self._button_reset = Button(self._ax_button_reset, 'Reset')
    self._button_reset.on_clicked(self._reset)
    button_offset_v -= 2*button_h

    self._axes = axes
    self._figure.canvas.mpl_connect('button_press_event', self._on_click)
    self._figure.canvas.mpl_connect('button_release_event', self._on_release)
    self._figure.canvas.mpl_connect('motion_notify_event', self._on_motion)

    plt.show()

  def _update_plot(self):
    # update ticks
    self._axes.set_xticks( [ i*self._sample_dt/3600 for i in range(self._sample_size+1)], minor=False)
    self._axes.set_xticklabels( [ strftime("%H:%M", gmtime( i*self._sample_dt ) ) for i in range(self._sample_size+1)], fontdict=None, minor=False, rotation=-45 )

    if not self._points:
      x = []
      y = []
    else:
      x, y = zip(*sorted(self._points.items()))

    if len(x) == 1:
      self._sa_y = np.array([0] * len(self._sa_x))
      self._sp_y = np.array([0] * len(self._sp_x))

    try:
      tck = interpolate.splrep(x, y, s=0)
      self._sa_y = np.array([ 0 if y < 0 else y for y in interpolate.splev(self._sa_x, tck, der=0)])
      self._sp_y = np.array(interpolate.splev(self._sp_x, tck, der=0))
    except Exception as e:
      #print('Fail spline : too few points')
      self._sa_y = np.array([0] * len(self._sa_x))
      self._sp_y = np.array([0] * len(self._sp_x))

    # Add new plot
    if not self._line:
      self._line, = self._axes.plot(x, y, 'b', marker='o', markersize=10)
      self._spline, = self._axes.plot(self._sp_x, self._sp_y, 'g')
      self._sample, = self._axes.plot(self._sa_x, self._sa_y, 'r', marker='o', markersize=7)
    # Update current plot
    else:
      self._line.set_data(x, y)
      self._sample.set_data(self._sa_x, self._sa_y)
      self._spline.set_data(self._sp_x, self._sp_y)

    self._figure.canvas.draw()

  def _add_point(self, x, y=None):
    if isinstance(x, MouseEvent):
        x, y = x.xdata, x.ydata
    self._points[x] = y
    return x, y

  def _remove_point(self, x, _):
    if x in self._points:
      self._points.pop(x)

  def _find_neighbor_point(self, event):
    u""" Find point around mouse position

    :rtype: ((int, int)|None)
    :return: (x, y) if there are any point around mouse else None
    """

    xl, xr = self._axes.get_xlim()
    dx = xr - xl
    yb, yt = self._axes.get_ylim()
    dy = yt - yb
    distance_threshold = 0.05
    min_distance = math.sqrt(2)
    nearest_point = None
    for x, y in self._points.items():
      distance = math.hypot( (event.xdata - x) / dx , (event.ydata - y) / dy )
      if distance < min_distance:
        min_distance = distance
        nearest_point = (x, y)
    if min_distance < distance_threshold:
      return nearest_point
    return None

  def _on_click(self, event):
    u""" callback method for mouse click event

    :type event: MouseEvent
    """
    # left click
    if event.button == 1 and event.inaxes in [self._axes]:
      point = self._find_neighbor_point(event)
      if point:
        self._dragging_point = point
      else:
        self._add_point(event)
      self._update_plot()
    # right click
    elif event.button == 3 and event.inaxes in [self._axes]:
      point = self._find_neighbor_point(event)
      if point:
        self._remove_point(*point)
        self._update_plot()

  def _on_release(self, event):
    u""" callback method for mouse release event

    :type event: MouseEvent
    """
    if event.button == 1 and event.inaxes in [self._axes] and self._dragging_point:
      self._dragging_point = None
      self._update_plot()

  def _on_motion(self, event):
    u""" callback method for mouse motion event

    :type event: MouseEvent
    """
    if not self._dragging_point:
      return
    if event.xdata is None or event.ydata is None:
      return
    self._remove_point(*self._dragging_point)
    self._dragging_point = self._add_point(event)
    self._update_plot()


if __name__ == "__main__":
  plot = interactive_plot()
