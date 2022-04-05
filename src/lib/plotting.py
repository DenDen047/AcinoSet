import os
import sys
import math
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
from typing import Dict, List
from cv2 import Rodrigues
from matplotlib import collections  as mc
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QGridLayout, QSizePolicy
from .misc import get_pose_params
from .points import common_image_points
from .utils import create_board_object_pts, load_scene

plt.style.use(os.path.join('..', 'configs', 'mplstyle.yaml'))
pg.setConfigOptions(antialias=True)


def create_camera(color=[0.1]*3):
    ## plot camera
    bx, by, bz = 0.15, 0.1, 0.1
    m = max([bx, by, bz])
    verts = np.array([
        [bx, by, bz], [bx, by, -bz], [bx, -by, bz], [bx, -by, -bz],
        [-bx, by, bz], [-bx, by, -bz], [-bx, -by, bz], [-bx, -by, -bz],
        [2 * bx, 2 * by, 2 * bz], [2 * bx, -2 * by, 2 * bz],
        [-2 * bx, 2 * by, 2 * bz], [-2 * bx, -2 * by, 2 * bz],
        [0, 0, 0], [5*m, 0, 0], [0, 5*m, 0], [0, 0, 5*m]
    ])
    edges = np.array([
        [0,1], [0,2], [0,4], [1,3], [1,5], [2,3],
        [2,6], [3,7], [4,5], [4,6], [5,7], [6,7],
        [0,8], [2,9], [4,10],[6,11],[8,9], [8,10],
        [9,11], [10,11], [12,13], [12,14], [12,15]
    ]).flatten()
    colors = np.array([color for _ in edges])
    colors[[40,41]] = (1, 0, 0)
    colors[[42,43]] = (0, 1, 0)
    colors[[44,45]] = (0, 0, 1)

    mesh = gl.GLLinePlotItem(pos=verts[edges], color=colors, width=1, antialias=True, mode='lines')
    return mesh


def create_grid(obj_points, board_shape, color=[0.5]*3):
    cols = board_shape[0]
    rows = board_shape[1]
    xyz_quiver = np.array([
        [0]*3,
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1]
    ])
    verts = np.vstack([obj_points, xyz_quiver])

    edges = []
    for r in range(rows):
        for c in range(cols-1):
            edges.append(c+r*cols)
            edges.append(c+r*cols+1)
    for c in range(cols):
        for r in range(rows-1):
            edges.append(c+r*cols)
            edges.append(c+(r+1)*cols)
    edges.extend(np.array([0,1,0,2,0,3])+(rows*cols))

    colors = np.array([color for _ in range(len(edges))])
    colors[[-6,-5]] = (1, 0, 0)
    colors[[-4,-3]] = (0, 1, 0)
    colors[[-2,-1]] = (0, 0, 1)

    mesh = gl.GLLinePlotItem(pos=verts[edges], color=colors, width=1.0, antialias=True, mode='lines')
    return mesh


def plot_calib_board(img_points, board_shape, camera_resolution, frame_fpath=None, obj_corners=None):
    corners = np.array(img_points, dtype=np.float32)
    if obj_corners is not None:
        obj_corners = np.array(obj_corners, dtype=np.float32)
    plt.figure(figsize=(16, 9), dpi=120)
    if frame_fpath:
        plt.imshow(plt.imread(frame_fpath))

    for pts in corners:
        pts = pts.reshape(-1, 2)
        cols = board_shape[0]
        rows = board_shape[1]
        edges = []
        for r in range(rows):
            for c in range(cols - 1):
                edges.append(c + r * cols)
                edges.append(c + r * cols + 1)
        for c in range(cols):
            for r in range(rows - 1):
                edges.append(c + r * cols)
                edges.append(c + (r + 1) * cols)
        lc = mc.LineCollection(pts[edges].reshape(-1, 2, 2), color='r', linewidths=0.1)
        plt.gca().add_collection(lc)

    if obj_corners is not None:
        for pts in obj_corners:
            pts = pts.reshape(-1, 2)
            cols = board_shape[0]
            rows = board_shape[1]
            edges = []
            for r in range(rows):
                for c in range(cols - 1):
                    edges.append(c + r * cols)
                    edges.append(c + r * cols + 1)
            for c in range(cols):
                for r in range(rows - 1):
                    edges.append(c + r * cols)
                    edges.append(c + (r + 1) * cols)
            lc = mc.LineCollection(pts[edges].reshape(-1, 2, 2), color='b', linewidths=0.1)
            plt.gca().add_collection(lc)

        plt.gca().set_xlim((0, camera_resolution[0]))
        plt.gca().set_ylim((camera_resolution[1], 0))

    plt.show()

def plot_corners(img_points, obj_corners, board_shape, camera_resolution, frame_fpath=None):
    corners = np.array(img_points, dtype=np.float32)
    obj_corners = np.array(obj_corners, dtype=np.float32)
    plt.figure(figsize=(16, 9), dpi=120)
    if frame_fpath:
        plt.imshow(plt.imread(frame_fpath))

    for pts in corners:
        pts = pts.reshape(-1, 2)
        plt.scatter(pts[:, 0], pts[:, 1], color="r", s=2)

    for pts in obj_corners:
        pts = pts.reshape(-1, 2)
        plt.scatter(pts[:, 0], pts[:, 1], color="b", s=2)

    plt.gca().set_xlim((0, camera_resolution[0]))
    plt.gca().set_ylim((camera_resolution[1], 0))

    plt.show()

def plot_confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor=facecolor, **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

class Animation:
    def __init__(self, title, scene_fpath, centered=False, dark_mode=False):
        self.app = QApplication.instance()
        if self.app == None:
            self.app = QApplication([])

        self.centered = False if 'Scene' in title else centered
        self.dark_mode = dark_mode
        self.screen_res = self.app.desktop().screenGeometry()
        self.screen_res = np.array([self.screen_res.width(), self.screen_res.height()])

        theme_colors = ['w','k']
        pg.setConfigOption('background', theme_colors[dark_mode])
        pg.setConfigOption('foreground', theme_colors[not dark_mode])

        self.win = pg.GraphicsLayoutWidget(title=title, size=self.screen_res/2, show=True)
        self.layout = QGridLayout()
        self.win.setLayout(self.layout)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor([255*(not dark_mode)]*3)
        # self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.opts['distance']=15
        self.view.orbit(-135,10)

        # camera pose
        self.k_arr, self.d_arr, self.r_arr, self.t_arr, self.cam_res = load_scene(scene_fpath)
        self.n_cams = len(self.k_arr)
        self.cam_pos = []

        grid = gl.GLGridItem(size=QtGui.QVector3D(50,50,0), color=[abs(55-255*(not dark_mode))]*3)
        self.view.addItem(grid)

        if not self.centered:
            for r, t in zip(self.r_arr, self.t_arr):
                self.plot_camera(r, t)

            scene_center = np.mean(self.cam_pos, axis=0)
            scene_center[2] = 0
            grid.translate(*scene_center)
            self.view.pan(*scene_center)
        else:
            grid.translate(0,0,-0.5)


    def rodrigues_to_vec(self, r):
        ang = 180 / np.pi * np.linalg.norm(r)
        return (ang, *r)

    def plot_camera(self, r, t):
        # https://en.wikipedia.org/wiki/Camera_resectioning#Extrinsic_parameters
        # T is the world origin position in the camera coordinates.
        # The world position of the camera is C = -(R^-1)@T.
        # Similarly, the rotation of the camera in world coordinates is given by R^-1
        R = r.T
        T = -R @ t
        self.cam_pos.append(T)
        # Convert for plotting
        R = Rodrigues(R)[0]
        R = self.rodrigues_to_vec(R)
        cam = create_camera([self.dark_mode]*3)
        cam.rotate(*R)
        cam.translate(*T)
        self.view.addItem(cam)

    def save_snapshot(self, filename, size=None):
        pg.makeQImage(self.view.renderToArray(size if size else self.screen_res)).save(filename)
        self.app.quit()

    def show(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.exec_()


class Scene(Animation):
    def __init__(self, scene_fpath, **kwargs):
        Animation.__init__(self, 'Scene Reconstruction', scene_fpath, **kwargs)
        self.layout.addWidget(self.view, 0, 0, 1, 1)

    def plot_calib_board(self, r, t, board_shape, board_edge_len):
        obj_pts = create_board_object_pts(board_shape, board_edge_len)
        calib_board = create_grid(obj_pts, board_shape)
        calib_board.translate(*t)
        r = self.rodrigues_to_vec(r)
        calib_board.rotate(*r)
        self.view.addItem(calib_board)

    def plot_points(self, points, color=[0]+[0.5]*3, size=None):
        scatter = gl.GLScatterPlotItem(pos=points, color=color, size=size if size else self.screen_res[0]/500, pxMode=True)
        scatter.setGLOptions('translucent')
        self.view.addItem(scatter)


class Cheetah(Animation):
    def __init__(self, multiple_reconstructions, scene_fpath, labels, project_func, hide_lure=False, reprojections=True, **kwargs):
        Animation.__init__(self, 'Cheetah Reconstruction', scene_fpath, **kwargs)
        self.layout.addWidget(self.view, 0, 0, self.n_cams, 1)

        # To add a legend, investigate https://groups.google.com/g/pyqtgraph/c/PfJvmjIF3Dg/m/QVG9xUGk-zgJ

        # indices correspond to joints in 'markers' variable
        lines_idxs = [0,1,0,2,1,2,1,3,0,3,2,3,3,4,4,5,5,6,6,7,
                      3,8,4,8,8,9,9,10,      # left front leg
                      3,11,4,11,11,12,12,13, # right front leg
                      4,14,5,14,14,15,15,16,
                      4,17,5,17,17,18,18,19]

        colours = [[self.dark_mode]*3+[1], # white if dark_mode else black
                   [1,0,1,1],              # fuchsia/magenta
                   [0,1,0,1],              # green
                   [0,0.8,0.8,1]]          # light blue

        self.n_reconstructions = len(multiple_reconstructions)
        assert self.n_reconstructions < 5, 'Cannot plot more than 4 reconstructions at a time'
        self.n_frames = len(multiple_reconstructions[0])
        self.frame = 0

        # assuming lure is the last element - see misc.get_markers
        self.scatter_frames, self.lines_frames, self.scatter_plots, self.line_plots = [], [], [], []
        for i in range(self.n_reconstructions):
            assert self.n_frames==len(multiple_reconstructions[i]), 'Elements along axis 0 of multiple_reconstructions must be of equal length'
            scatter_frames, lines_frames = [], []
            for frame in multiple_reconstructions[i]:
                if self.centered:
                    dots_ave = np.nanmean(frame[:-1], axis=0) # exlude lure from calc
                    frame -= dots_ave
                if hide_lure:
                    frame = frame[:-1]
                scatter_frames.append(frame)
                lines_frames.append(frame[lines_idxs, :])

            self.scatter_frames.append(np.array(scatter_frames))
            self.lines_frames.append(np.array(lines_frames))

            # create dots
            self.scatter_plots.append(gl.GLScatterPlotItem(pos=np.zeros((1,3)), color=colours[i], size=self.screen_res[0]/250, pxMode=True))
            self.scatter_plots[i].setGLOptions('translucent')
            self.view.addItem(self.scatter_plots[i])

            # create links
            self.line_plots.append(gl.GLLinePlotItem(pos=np.zeros((2,3)), color=colours[i], width=self.screen_res[0]/1250, antialias=True, mode='lines'))
            self.line_plots[i].setGLOptions('translucent')
            self.view.addItem(self.line_plots[i])

        # ====== 2D ======
        if self.centered:
            self.reprojections = False
            if reprojections:
                print('Reprojections are not permitted in centered mode')
        else:
            self.reprojections = reprojections

        if self.reprojections:
            self.cam_data, self.cam_lines, cam_w = [], [], []
            for i in range(self.n_cams):
                cam_w.append(pg.PlotWidget())
                cam_w[i].setXRange(0, self.cam_res[0])
                cam_w[i].setYRange(self.cam_res[1], 0)
                cam_w[i].invertY()
                cam_w[i].sizeHint = lambda: pg.QtCore.QSize(self.screen_res[0]/7, self.screen_res[1]/7)
                cam_w[i].setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                cam_w[i].setBackground([255*(not self.dark_mode)]*3)

                cam_i_lines, cam_data = [], []
                for j in range(self.n_reconstructions):
                    cam_data.append(pg.PlotDataItem(connect='pairs', pen=pg.mkPen(255*np.array(colours[j]))))
                    cam_w[i].addItem(cam_data[j])
                    cam_params = [self.k_arr[i], self.d_arr[i], self.r_arr[i], self.t_arr[i]]
                    cam_i_lines.append(project_func(self.lines_frames[j], *cam_params).reshape((self.n_frames, -1, 2)))

                self.cam_lines.append(cam_i_lines)
                self.cam_data.append(cam_data)
                self.layout.addWidget(cam_w[i], i, 1, 1, 1)

    def update(self):
        for i in range(self.n_reconstructions):
            if self.reprojections:
                for j in range(self.n_cams):
                    self.cam_data[j][i].setData(self.cam_lines[j][i][self.frame])

            self.scatter_plots[i].setData(pos=self.scatter_frames[i][self.frame])
            self.line_plots[i].setData(pos=self.lines_frames[i][self.frame])

        self.frame = (self.frame+1) % self.n_frames

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(100) # speed of reconstruction
        self.show()


def plot_extrinsics(scene_fpath, pts_2d, fnames, triangulate_func, manual_points_only=False, **kwargs):
    scene = Scene(scene_fpath, **kwargs)

    colors = [[1,0,0],                        # red: cam pair 0&1
              [0,1,0],                        # green: cam pair 1&2
              [kwargs.get('dark_mode', 0)]*3, # white if dark_mode else black: cam pair 2&3
              [0,0,1],                        # blue: cam pair 3&4
              [0,0.8,0.8],                    # light blue: cam pair 4&5
              [1,0,1]]                        # fuchsia/magenta: cam pair 5&0

    for cam in range(scene.n_cams):
        a, b = cam % scene.n_cams, (cam + 1) % scene.n_cams
        img_pts_1, img_pts_2, _ = common_image_points(pts_2d[a], fnames[a], pts_2d[b], fnames[b])
        try:
            pts_3d = triangulate_func(
                img_pts_1, img_pts_2,
                scene.k_arr[a], scene.d_arr[a], scene.r_arr[a], scene.t_arr[a],
                scene.k_arr[b], scene.d_arr[b], scene.r_arr[b], scene.t_arr[b]
            )
            scene.plot_points(pts_3d, color=colors[cam]+[1]) # must have transparency channel
        except:
            msg = 'Could not triangulate points' if len(img_pts_1) else 'No points exist'
            print(msg, f'for cam pair with indices {[a,b]}')

    scene.show()


def plot_marker_3d(pts_3d, frames=None, fitted_pts_3d=None, fig_title='3D points'):
    if frames is None:
        frames = np.arange(len(pts_3d))

    frames += 1 # frames are 1 based indices
    num_axes = pts_3d.shape[1]

    fig, axs = plt.subplots(1, num_axes, figsize=(num_axes*6,5))
    fig.suptitle(fig_title)
    for ax in range(num_axes):
        axs[ax].plot(frames, pts_3d[:, ax], 'o-', markersize=2)
        axs[ax].set_xlabel('Frames')
        axs[ax].set_ylabel('Position (m)')
        axs[ax].set_title(chr(ord('X') + ax) + ' Axis')

    if fitted_pts_3d is not None:
        for ax in range(num_axes):
            err = pts_3d[:, ax] - fitted_pts_3d[:, ax]
            pad = 2*err.std() # 95%
            axs[ax].plot(frames, fitted_pts_3d[:, ax])
            axs[ax].set_ylim(fitted_pts_3d[:, ax].min() - pad, fitted_pts_3d[:, ax].max() + pad)
            axs[ax].legend(['Original', 'Curve Fit'])

    plt.show(block=False)


def plot_optimized_states(x, idxs, smoothed_x=None, mplstyle_fpath=None):
    x = np.array(x)
    if smoothed_x is not None:
        smoothed_x = np.array(smoothed_x)
    if mplstyle_fpath is not None:
        plt.style.use(mplstyle_fpath)

    _titles = [
        'Lure positions',
        'Head positions', 'Head angles',
        'Neck length',
        'Neck angles',
        'Front torso angle', 'Back torso angles',
        'Tail base angles', 'Tail mid angles',
        'Left shoulder angle', 'Left front knee angle',
        'Right shoulder angles', 'Right front knee angle',
        'Left hip angle', 'Left back knee angle',
        'Right hip angle', 'Right back knee angle'
    ]
    _label_lists = [
        ['x_l', 'y_l', 'z_l'], # exclude lure for now
        ['x_0', 'y_0', 'z_0'],
        ['phi_0', 'theta_0', 'psi_0'],
        ['l_1'],
        ['phi_1', 'theta_1', 'psi_1'],
        ['theta_2'], ['phi_3', 'theta_3', 'psi_3'],
        ['theta_4', 'psi_4'], ['theta_5', 'psi_5'],
        ['theta_6'], ['theta_7'],
        ['theta_8'], ['theta_9'],
        ['theta_10'], ['theta_11'],
        ['theta_12'], ['theta_13']
    ]

    titles = []
    label_lists = []
    for title, lbl in zip(_titles, _label_lists):
        if set(lbl).issubset(idxs.keys()):
            titles.append(title)
            label_lists.append(lbl)

    idxs = [[idxs[l] for l in lbl] for lbl in label_lists]

    if len(titles) == 1:
        plt_shape = [1, 1]
    else:
        plt_shape = [len(titles)//2, 2]
    fig, axs = plt.subplots(*plt_shape, figsize=(plt_shape[1]*7, plt_shape[0]*4))


    for i in range(plt_shape[0]):
        for j in range(plt_shape[1]):
            k = 2*i+j
            if plt_shape[1] > 1:
                ax = axs[i,j] if plt_shape[0] > 1 else axs[j]
            else:
                ax = axs
            ax.set_title(titles[k])
            ax.plot(x[:, idxs[k]])
            lgnd = label_lists[k]
            if smoothed_x is not None:
                ax.plot(smoothed_x[:, idxs[k]])
                lgnd += [l + ' (smoothed)' for l in lgnd]
            ax.legend(lgnd)

    plt.show(block=False)
    return fig, axs


def plot_shutter_delay(data, mplstyle_fpath=None):
    data = np.array(data)
    n_camera, n_frame = data.shape
    if mplstyle_fpath is not None:
        plt.style.use(mplstyle_fpath)

    fig, ax = plt.subplots()

    x = range(1, n_frame+1)
    for c in range(n_camera):
        ax.plot(x, data[c,:]*1e3, label=f'camera_{c+1}')
    ax.legend()
    ax.set_xlabel('frame')
    ax.set_ylabel('time delay (ms)')
    ax.set_title('Shutter Delay')

    return fig, ax


def plot_value_sets(values: List, titles: List[str], labels: List[str] = None, mplstyle_fpath=None):
    assert len(values) == len(titles)

    n_plot = len(titles)

    if mplstyle_fpath is not None:
        plt.style.use(mplstyle_fpath)

    all_values = np.array(values)
    ymin = np.nanmin(all_values) - 0.01
    ymax = np.nanmax(all_values) + 0.01

    plt_shape = [math.ceil(len(titles)/2), 2]
    fig, axs = plt.subplots(*plt_shape, figsize=(plt_shape[1]*7, plt_shape[0]*4))

    for i in range(plt_shape[0]):
        for j in range(plt_shape[1]):
            k = 2*i+j
            ax = axs[i,j] if len(axs.shape) > 1 else axs[j]

            ax.set_title(titles[k])
            x = all_values[k, :]    # [n_frame, v]
            ax.plot(x)
            ax.set_xlim(0, x.shape[0])
            ax.set_ylim(ymin, ymax)
            if labels is not None:
                ax.legend(labels)

            if len(titles) - 1 == k:
                break

    plt.show(block=False)
    return fig, axs
