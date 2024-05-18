# This file contains all files related to firebase
import pyrebase
from loguru import logger
from constants import Constants as const

firebaseProdConfig = {
	"apiKey": "AIzaSyCyn-mmp-ZLkk2wEKLjEE9gxm7a8Ewx7bw",
	"authDomain": "robsgg-e1f74.firebaseapp.com",
	"databaseURL": "https://robsgg-e1f74-default-rtdb.firebaseio.com",
	"projectId": "robsgg-e1f74",
	"storageBucket": "robsgg-e1f74.appspot.com",
	"messagingSenderId": "223334572997",
	"appId": "1:223334572997:web:b33fbc61caa53af345a1ac"
}

firebase = pyrebase.initialize_app(firebaseProdConfig)

logger.info("----------------------------------------------------------------------")
logger.info("Setting up Firebase Service...in ")
logger.info("----------------------------------------------------------------------")


class FirebaseService:
	
	# 1. Have this db connection constant
	def __init__(self):
		self.db = firebase.database()
	
	# ---------------------- BEGIN RESULTS ----------------------
	def fetch_results(self):
		return self.db.child(const.RESULTS).get().val()
	
	def fetch_results_from_db(self, database_name):
		return self.db.child(database_name).get().val()
	
	# ---------------------- END RESULTS ----------------------
	
	# ---------------------- BEGIN RESULT ----------------------
	
	def fetch_result(self, result_id: str):
		return self.db.child(const.RESULTS).child(result_id).get().val()
	
	def remove_all_results(self):
		self.db.child(const.RESULTS).remove()
	
	def update_result(self, result_id: str, result: dict):
		self.db.child(const.RESULTS).child(result_id).set(result)
		logger.info(f"Updated result in the firebase - {result.__str__()}")
	
	def update_result_to_db(self, database_name, result_id: str, result: dict):
		self.db.child(database_name).child(result_id).set(result)
		logger.info(f"Updated result in the firebase - {result.__str__()}")
	
	def remove_result(self, result_id: str):
		self.db.child(const.RESULTS).child(result_id).remove()


# ---------------------- END RESULT ----------------------


if __name__ == "__main__":
	db_service = FirebaseService()