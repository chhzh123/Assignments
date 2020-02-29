import unittest
import numpy as np
from Q01 import Solution as Sol1
from Q02 import Solution as Sol2

class TestSolution1(unittest.TestCase):

	def setUp(self):
		self.sol1 = Sol1()

	def test_sigmoid(self):
		self.assertEqual(self.sol1.sigmoid(10),1/(1 + np.exp(-10)))

	def test_tanh(self):
		self.assertTrue(np.abs(self.sol1.tanh(10) - np.tanh(10)) < 1e-6)

	def test_relu(self):
		self.assertEqual(self.sol1.relu(5),5)

	def test_leaky_relu(self):
		self.assertEqual(self.sol1.leaky_relu(0.5,-5),-2.5)

	def test_elu(self):
		self.assertEqual(self.sol1.elu(0.5,0),0)


class TestSolution2(unittest.TestCase):

	def setUp(self):
		self.sol2 = Sol2()

	def test_reverse_numbers(self):
		self.assertEqual(self.sol2.reverse_numbers(123),321)
		self.assertEqual(self.sol2.reverse_numbers(-123),-321)
		self.assertEqual(self.sol2.reverse_numbers(120),21)

	def test_third_maximum_number(self):
		self.assertEqual(self.sol2.third_maximum_number([1,2,3]),1)
		self.assertEqual(self.sol2.third_maximum_number([1,2]),2)
		self.assertEqual(self.sol2.third_maximum_number([2,2,3,1]),1)

if __name__ == '__main__':
	unittest.main()