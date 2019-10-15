from concurrent import futures
import logging

import grpc

import pyinterpreter_pb2
import pyinterpreter_pb2_grpc

class Evaluation(pyinterpreter_pb2_grpc.EvaluationServicer):

	def eval(self,request,context):
		try:
			msg = str(eval(request.msg))
		except:
			exec(request.msg)
			msg = ""
		return pyinterpreter_pb2.EvalReply(msg=msg)

def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	pyinterpreter_pb2_grpc.add_EvaluationServicer_to_server(Evaluation(),server)
	server.add_insecure_port('[::]:50051')
	server.start()
	server.wait_for_termination()

if __name__ == '__main__':
	logging.basicConfig()
	serve()