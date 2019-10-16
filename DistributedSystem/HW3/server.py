from concurrent import futures
import logging
import yaml

import grpc

import numpy as np
import mysql.connector
import pyinterpreter_pb2
import pyinterpreter_pb2_grpc

class Evaluation(pyinterpreter_pb2_grpc.EvaluationServicer):

	def __init__(self):
		print("Created Python interpreter!")

	def eval(self,request,context):
		try:
			msg = str(eval(request.msg))
		except:
			exec(request.msg)
			msg = ""
		return pyinterpreter_pb2.EvalReply(msg=msg)

class SQLQuery(pyinterpreter_pb2_grpc.QueryServicer):

	def __init__(self):
		config = yaml.load(open('config.yaml'))
		print("Connected MySQL '{}'@'{}'!".format(config["user"],config["host"]))
		self.db = mysql.connector.connect(
					  host=config["host"],
					  user=config["user"],
					  passwd=config["passwd"]
					)
		self.cursor = self.db.cursor()

	def query(self,request,context):
		self.cursor.execute("use school;")
		self.cursor.execute(request.msg)
		return pyinterpreter_pb2.EvalReply(msg=str(self.cursor.fetchall()))

def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	pyinterpreter_pb2_grpc.add_EvaluationServicer_to_server(Evaluation(),server)
	pyinterpreter_pb2_grpc.add_QueryServicer_to_server(SQLQuery(),server)
	server.add_insecure_port('[::]:50051')
	server.start()
	server.wait_for_termination()

if __name__ == '__main__':
	logging.basicConfig()
	serve()