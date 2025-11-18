import unittest
import os
import pickle
import numpy as np
from tqdm import tqdm

class TestPreprocessingOutput(unittest.TestCase):
    def setUp(self):
        # Set these paths to your reference and test output folders
        self.angle_ref_file = '/pdo/users/jlupoiii/TESS/data/angles/angles_O11-54_data_dic.pkl'
        self.angle_test_file = '/pdo/users/djtufto/data/test_count/test_tess_angles_O11-54_data_dic.pkl'
        self.background_ref_dir = '/pdo/users/jlupoiii/TESS/data/background_avg_ccds_im256x256/'
        self.background_test_dir = '/pdo/users/djtufto/pptestnew/bg/'
        self.reference_dir = '/pdo/users/jlupoiii/TESS/data/processed_images_im256x256/'
        self.test_output_dir = '/pdo/users/djtufto/pptestnew/ccd/'
        self.config_path = 'config/preprocess_config.yaml'
        self.raw_ccd_dir = '/pdo/users/roland/SL_data/'
        self.ccd_files = 0
        self.background_files = 0
        self.tolerance = 1e-1  # Adjust as needed
        self.orbit_range = range(11, 54)
        self.out_of_orbit_files = 0
        self.raw_ccd_files = 0
        self.num_mismatches = 0
    
    def test_angles(self):
        with open(self.angle_ref_file, 'rb') as f:
            ref_angle_data = pickle.load(f)
        with open(self.angle_test_file, 'rb') as f:
            test_angle_data = pickle.load(f)
        self.assertEqual(ref_angle_data, test_angle_data)

    def test_ccd(self):
        ccd_ref_files = [f for f in os.listdir(self.reference_dir) if f.endswith('.pkl')]
        #Check we are making same files
        self.assertTrue(ccd_ref_files, f"No .pkl files found in reference directory: {self.reference_dir}")
        for ref_file in tqdm(ccd_ref_files):
            ref_path = os.path.join(self.reference_dir, ref_file)
            test_path = os.path.join(self.test_output_dir, ref_file)
            if not os.path.exists(test_path):
                self.out_of_orbit_files += 1
                continue
            self.assertTrue(os.path.exists(test_path), f"Missing output file: {test_path}")
            with open(ref_path, 'rb') as f:
                ref_img = pickle.load(f)
            with open(test_path, 'rb') as f:
                test_img = pickle.load(f)
            self.assertEqual(ref_img.shape, test_img.shape, f"Image shape mismatch for file {ref_file}")
            if not np.allclose(ref_img, test_img, atol=1e-3):
                self.num_mismatches += 1
            self.ccd_files += 1
        print(f"Out of orbit files: {self.out_of_orbit_files}")
        print(f"Number of CCD files: {self.ccd_files}")
        print(f"Number of CCD files in reference: {len(ccd_ref_files)}")
        print(f"Number of mismatches: {self.num_mismatches}")
        self.assertEqual(self.ccd_files, len(ccd_ref_files), f"Number of files mismatch: {self.ccd_files} != {len(ccd_ref_files)}")

    def test_background(self):
        background_ref_files = [f for f in os.listdir(self.background_ref_dir) if f.endswith('.pkl')]
        self.assertTrue(background_ref_files, f"No .pkl files found in reference directory: {self.background_ref_dir}")
        for ref_file in background_ref_files:
            ref_path = os.path.join(self.background_ref_dir, ref_file)
            test_path = os.path.join(self.background_test_dir, ref_file)
            if not os.path.exists(test_path):
                orbit = int(ref_file.split('_')[0][1:])
                if orbit not in self.orbit_range:
                    print(f"Missing output file: {test_path}")
                    self.out_of_orbit_files += 1
                    continue
                else:
                    self.assertTrue(os.path.exists(test_path), f"Missing output file: {test_path}")
            with open(ref_path, 'rb') as f:
                ref_img = pickle.load(f)
            with open(test_path, 'rb') as f:
                test_img = pickle.load(f)
            #Check if the images are the same
            self.assertTrue(
                np.allclose(ref_img, test_img, atol=self.tolerance),
                f"Image mismatch for file {ref_file}"
            )
            self.background_files += 1
        self.assertEqual(self.background_files, len(background_ref_files) - self.out_of_orbit_files, f"Number of files mismatch: {self.background_files} != {len(background_ref_files) - self.out_of_orbit_files}")
        
if __name__ == '__main__':
    unittest.main() 