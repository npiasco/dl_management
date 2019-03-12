import setlog
import numpy as np
import sklearn.linear_model
import sklearn.base


logger = setlog.get_logger(__name__)


class PoseEstimator(sklearn.base.BaseEstimator):
    def __init__(self):
        self.threshold = 1e-3
        self.T = np.eye(4,4)

    def fit(self, X, y):
        X_centroid = np.mean(X[:, :3], 0)
        X_centred = X[:, :3] - X_centroid

        y_centroid = np.mean(y[:, :3], 0)
        y_centred = y[:, :3] - y_centroid

        H = np.matmul(y_centred.transpose(1, 0), X_centred).transpose(1, 0)
        U, S, V = np.linalg.svd(H)
        V = V.transpose(1, 0)
        if np.linalg.det(U) * np.linalg.det(V) < 0:
            V[:, -1] *= -1

        R = np.matmul(V, U.transpose(1, 0))

        # translation
        t = y_centroid - np.matmul(R, X_centroid)

        # homogeneous transformation
        T = np.eye(4, 4)
        T[:3, :3] = R
        T[:3, 3] = t
        self.T = T

    def score(self, X, y):
        y_est = self.predict(X)
        '''
        u = np.sum(np.sum((y-y_est)**2, 1))
        v = np.sum(np.sum((y-np.mean(y, 0))**2, 1))
        return 1-u/v
        '''
        '''
        u = np.sum((y - y_est) ** 2, 1)
        v = np.sum((y - np.mean(y, 0)) ** 2, 1)
        return np.average(1 - u / v, weights=v)
        '''
        return  1 - np.mean((np.sum((y-y_est)**2, 1)))

    def predict(self, X):
        return np.matmul(self.T, X.transpose(1,0)).transpose(1,0)


def ransac_pose_estimation(pt, pt_nn):
    ransac = sklearn.linear_model.RANSACRegressor(base_estimator=PoseEstimator(), min_samples=0.5, max_trials=100,
                                                  residual_threshold=0.4, loss='squared_loss')
    ransac.fit(pt.t().cpu().numpy(), pt_nn.t().cpu().numpy())
    fscore = ransac.score(pt.t().cpu().numpy(), pt_nn.t().cpu().numpy())
    logger.debug('Ransac score {} in {} iteration'.format(ransac.score(pt.t().cpu().numpy(), pt_nn.t().cpu().numpy()),
                                                          ransac.n_trials_))
    return {'T': pt.new_tensor(ransac.estimator_.T).unsqueeze(0),
            'score': fscore, 'inliers_ratio': sum(ransac.inlier_mask_)/len(ransac.inlier_mask_)}



