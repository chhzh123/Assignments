import unittest
import numpy as np
from MSELoss_ls import MSELoss_ls

class TestMSELoss_ls(unittest.TestCase):

	def setUp(self):
		pass

	def test1(self):
		x = np.array([[0.8,0.2],[0.2,0.8]])
		target = np.array([0,1])
		sm_factor = 0.1
		self.assertEqual(MSELoss_ls(x,target,sm_factor),((0.8-0.1)**2+(0.2-0.9/1)**2)*2)

	def test2(self):
		x = np.array([[0.4,0.3,0.3]])
		target = np.array([0])
		sm_factor = 0.2
		self.assertEqual(MSELoss_ls(x,target,sm_factor),(0.4-0.2)**2+(0.3-0.8/2)**2+(0.3-0.8/2)**2)

if __name__ == '__main__':
	unittest.main()