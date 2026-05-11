import numpy as np
import cv2

class IQAModule:
    @staticmethod
    def compute(img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        L = np.mean(ycrcb[:, :, 0]) / 255.0
        
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        diff = img.astype(np.float32) - blurred.astype(np.float32)
        N = np.std(diff) / 255.0
        
        if L < 0.3:
            bin_id = 'dark'
        elif L > 0.6:
            bin_id = 'bright'
        else:
            bin_id = 'medium'
            
        q = max(0.0, min(1.0, 1.0 - N))
        return L, N, bin_id, q