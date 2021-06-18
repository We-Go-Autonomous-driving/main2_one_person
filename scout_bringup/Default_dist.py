import numpy as np

class Default_dist:
    def __init__(self, roi1_th=10000, roi2_th=10000, roi3_th=10000, roi4_th=10000):
        self.roi1_th = roi1_th
        self.roi2_th = roi2_th 
        self.roi3_th = roi3_th
        self.roi4_th = roi4_th

    def default_update(self, depth_frame):
        obs_roi1 = np.array((depth_frame[435:445, 195:205]),dtype=np.float64)
        obs_roi2 = np.array((depth_frame[395:405, 235:245]),dtype=np.float64)
        obs_roi3 = np.array((depth_frame[395:405, 395:405]),dtype=np.float64)
        obs_roi4 = np.array((depth_frame[435:445, 435:445]),dtype=np.float64)

        obs_roi1 = np.where(obs_roi1<1, np.mean(obs_roi1), obs_roi1)
        obs_roi2 = np.where(obs_roi2<1, np.mean(obs_roi2), obs_roi2)
        obs_roi3 = np.where(obs_roi3<1, np.mean(obs_roi3), obs_roi3)
        obs_roi4 = np.where(obs_roi4<1, np.mean(obs_roi4), obs_roi4)

        mean1 = np.mean(obs_roi1)
        mean2 = np.mean(obs_roi2)
        mean3 = np.mean(obs_roi3)
        mean4 = np.mean(obs_roi4)

        self.roi1_th = np.mean([self.roi1_th, mean1])
        self.roi2_th = np.mean([self.roi2_th, mean2])
        self.roi3_th = np.mean([self.roi3_th, mean3])
        self.roi4_th = np.mean([self.roi4_th, mean4])

        print(self.roi1_th, self.roi2_th, self.roi3_th, self.roi4_th)