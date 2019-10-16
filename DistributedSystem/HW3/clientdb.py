import logging

import grpc

from decimal import Decimal
import pyinterpreter_pb2
import pyinterpreter_pb2_grpc

def run():
	with grpc.insecure_channel('localhost:50051') as channel:
		stub = pyinterpreter_pb2_grpc.QueryStub(channel)
		print("gRPC-based MySQL interpreter")
		while True:
			print("mysql> ",end="")
			message = input()
			response = stub.query(pyinterpreter_pb2.EvalRequest(msg=message))
			if response.msg != "":
				result = eval(response.msg)
				for item in result:
					print(item)

if __name__ == '__main__':
	logging.basicConfig()
	run()