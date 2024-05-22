# This file contains all files related to firebase
import pyrebase
from loguru import logger
from constants import Constants as const

firebaseProdConfig = {
	"apiKey": "AIzaSyDtk0iVz6h800KpsEHGqGtnWvCdqfnQLsk",
	"authDomain": "captaincook-566a5.firebaseapp.com",
	"databaseURL": "https://captaincook-566a5-default-rtdb.firebaseio.com",
	"projectId": "captaincook-566a5",
	"storageBucket": "captaincook-566a5.appspot.com",
	"messagingSenderId": "644085661673",
	"appId": "1:644085661673:web:297e2e2082e9f2b07c1578"
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
