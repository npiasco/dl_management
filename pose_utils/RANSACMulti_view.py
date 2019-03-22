import setlog
import numpy as np
import sklearn.linear_model
import sklearn.base
import pyopengv
import pose_utils.utils as utils
import torch


logger = setlog.get_logger(__name__)


class angle_between_bearing:
    def __init__(self, poses):
        self.poses = poses

    def __call__(self, proj_bearings, bearings):
        bearings = np.split(bearings, bearings.shape[1]//3, axis=1)
        proj_bearings = np.split(proj_bearings, proj_bearings.shape[1]//3, axis=-1)
        angles = list()
        for i, bearing in enumerate(bearings):
            angles.append(np.arccos(np.sum(bearing * proj_bearings[i], axis=1)))

        angles = np.mean(angles, axis=0)
        return angles


def average_rotation(Rs, weights=None):
    '''
    https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
    '''
    if weights:
        qs = [utils.rot_to_quat(torch.from_numpy(R)).numpy().reshape(4, 1)*weights[i] for i, R in enumerate(Rs)]
    else:
        qs = [utils.rot_to_quat(torch.from_numpy(R)).numpy().reshape(4, 1) for R in Rs]

    Q = np.concatenate(qs, axis=1)
    M = np.matmul(Q, np.transpose(Q))
    w, v = np.linalg.eig(M)

    max_eig = np.argmax(w)
    return utils.quat_to_rot(torch.from_numpy(v[:, max_eig].real)).numpy()


def intersec(x_r, u_dir, weights=None):
    '''
    Compute the position of the point that minimise the distance between line defined by 3D points x_r and vector u_dir
    '''

    n_ex = len(x_r)
    A = np.zeros((3*n_ex, 3))
    B = np.zeros((3*n_ex))

    for i, x in enumerate(x_r):
        u = u_dir[i]
        A[i * 3, 1] = u[2]
        A[i * 3, 2] = -u[1]
        A[i * 3 + 1, 0] = -u[2]
        A[i * 3 + 1, 2] = u[0]
        A[i * 3 + 2, 0] = u[1]
        A[i * 3 + 2, 1] = -u[0]
        B[i * 3] = x[1]*u[2] - x[2]*u[1]
        B[i * 3 + 1] = x[2]*u[0] - x[0]*u[2]
        B[i * 3 + 2] = x[0]*u[1] - x[1]*u[0]

        if weights:
            A[i * 3:i * 3 + 3, :] *= np.sqrt(weights[i])
            B[i * 3:i * 3 + 3] *= np.sqrt(weights[i])

    x = np.linalg.lstsq(A, B, rcond=None)
    return x[0], x[1]


class MultiViewEstimator(sklearn.base.BaseEstimator):
    def __init__(self, camera_poses, pnp_algo, ransac_threshold, iterations):

        self.camera_poses = camera_poses
        self.pnp_algo = pnp_algo
        self.ransac_threshold = ransac_threshold
        self.iterations = iterations
        self.T = np.eye(4)
        self.residual = 0

    def fit(self, ref_bearing, bearings):

        bearings = np.split(bearings, bearings.shape[1]//3, axis=1)
        inliers = list()
        Rr = list()
        x_r = list()
        u_dir = list()

        for i, bearing in enumerate(bearings):
            Tr = pyopengv.relative_pose_ransac(ref_bearing, bearing,
                                               algo_name=self.pnp_algo,
                                               threshold=self.ransac_threshold,
                                               iterations=self.iterations)

            with open("ransac_inliers.txt", 'r') as f:
                inlier = int(f.read())
            inliers.append(inlier)
            Tr_w = np.eye(4)
            Tr_w[:3, :] = Tr
            Tr_w = np.matmul(np.linalg.inv(Tr_w), self.camera_poses[i])
            Rr.append(Tr_w[:3, :3])
            x_r.append(self.camera_poses[i][:3, 3])
            u_dir.append((np.matmul(self.camera_poses[i][:3, :3], -Tr[:3, 3])) / np.linalg.norm(
                np.matmul(self.camera_poses[i][:3, :3], -Tr[:3, 3]))
                         )
        t, residual = intersec(x_r, u_dir, weights=inliers)

        self.residual = residual

        self.T[:3, :3] = average_rotation(
            Rr,
            weights=inliers
        )
        self.T[:3, 3] = t

    def score(self, ref_bearing, bearings):
        projected_bearings = self.predict(ref_bearing)

        bearings = np.split(bearings, bearings.shape[1]//3, axis=1)
        proj_bearings = np.split(projected_bearings, projected_bearings.shape[1]//3, axis=-1)
        #score = 0
        score = list()
        '''
        for i, bearing in enumerate(bearings):
            #score += np.sum(np.sum(bearing * proj_bearings[i], axis=1))
            score.append(np.mean(np.sum(bearing * proj_bearings[i], axis=1)))

        #return np.mean(score) - self.residual
        '''
        print(1 - self.residual)
        print(self.T)
        return 1 - self.residual

    def predict(self, X):
        '''
        projected_bearing = list()
        for i, pose in enumerate(self.camera_poses):
            T_relative = np.matmul(self.T, np.linalg.inv(pose))
            #T_relative = np.matmul(np.linalg.inv(self.T), (pose))
            projected_bearing.append( np.matmul(T_relative[:3, :3],  X.transpose()).transpose()) #   + T_relative[:3, 3]
        return np.concatenate(projected_bearing, axis=1)
        '''
        return np.matmul(self.T[:3, :3], X.transpose()).transpose() # vect directeur  + self.T[:3, 3]

    def mean_intersec_distance(self, bearings, proj_bearing):
        bearings_s = np.split(bearings, bearings.shape[1]//3, axis=1)
        bearings_w = [np.matmul(self.camera_poses[i][:3, :3], vect.transpose()).transpose()
                      for i, vect in enumerate(bearings_s)]
        x_r = [self.T[:3, 3]] + [pose[:3, 3] for pose in self.camera_poses]

        intersec_distance = list()
        for i, vector in enumerate(proj_bearing):
            #u_r = [vector] + [vects[i, :] for vects in bearings_w]
            prj_pt =[intersec([self.T[:3, 3], self.camera_poses[j][:3, 3]], [vector, vects[i, :]])[0] for j, vects in
                     enumerate(bearings_w)]
            intersec_distance.append(np.mean(np.std(prj_pt, axis=0)))
        #print(intersec_distance)
        return np.array(intersec_distance).reshape(len(intersec_distance))


def multi_view_ransac_estimator(bearing, bearings, camera_poses, **kwargs):
    pnp_algo = kwargs.pop('pnp_algo', 'NISTER')
    pnp_ransac_threshold = kwargs.pop('pnp_ransac_threshold', 0.0002)
    pnp_iterations = kwargs.pop('pnp_iterations', 1000)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    inter_loss = angle_between_bearing(camera_poses)

    estimator = MultiViewEstimator(camera_poses,
                                   pnp_algo,
                                   pnp_ransac_threshold,
                                   pnp_iterations)

    ransac = sklearn.linear_model.RANSACRegressor(base_estimator=estimator,
                                                  min_samples=0.3,
                                                  max_trials=100,
                                                  loss=estimator.mean_intersec_distance,
                                                  residual_threshold=0.05)

    ransac.fit(bearing, bearings)

    fscore = ransac.score(bearing, bearings)
    logger.debug('Ransac score {} in {} iteration with {} inliers (on {})'.format(fscore,
                                                                                  ransac.n_trials_,
                                                                                  np.sum(ransac.inlier_mask_),
                                                                                  ransac.inlier_mask_.size))

    return ransac.estimator_.T
