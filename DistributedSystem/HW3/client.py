import logging

import grpc

import pyinterpreter_pb2
import pyinterpreter_pb2_grpc

def run():
	with grpc.insecure_channel('localhost:50051') as channel:
		stub = pyinterpreter_pb2_grpc.EvaluationStub(channel)
		print("gRPC-based Python interpreter")
		while True:
			print(">>> ",end="")
			message = input()
			response = stub.eval(pyinterpreter_pb2.EvalRequest(msg=message))
			if response.msg != "":
				print(response.msg)

if __name__ == '__main__':
	logging.basicConfig()
	run()