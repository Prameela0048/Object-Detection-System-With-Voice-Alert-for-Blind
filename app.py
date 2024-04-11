from flask import Flask, render_template
import DistanceEstimation
import threading
import os

app = Flask(__name__)
t1 = None  # Thread for DistanceEstimation.SWAB()
running = False  # Variable to track if DistanceEstimation.SWAB() is running

def run_swab():
    global running
    DistanceEstimation.SWAB()

@app.route("/")
def home():
    return render_template("Main.html")

@app.route("/start_execution")
def start_execution():
    global t1, running
    if not running:
        t1 = threading.Thread(target=run_swab)
        t1.start()
        running = True
        print("Execution started successfully")
    return "Execution started successfully"

@app.route("/stop_execution")
def stop_execution():
    global running
    if running:
        running = False
        os._exit(0)  # Terminate the script
    return "Execution stopped successfully"

if __name__ == "__main__":
    app.run(debug=True)
# from flask import Flask, render_template
# import DistanceEstimation
# import threading

# app = Flask(__name__)
# t1 = None  # Thread for DistanceEstimation.SWAB()
# running = False  # Variable to track if DistanceEstimation.SWAB() is running

# def run_swab():
#     global running
#     while running:
#         DistanceEstimation.SWAB()

# @app.route("/")
# def home():
#     return render_template("Main.html")

# @app.route("/start_execution")
# def start_execution():
#     global t1, running
#     if not running:
#         t1 = threading.Thread(target=run_swab)
#         t1.start()
#         running = True
#         print("Execution started successfully")
#     return "Execution started successfully"

# @app.route("/stop_execution")
# def stop_execution():
#     global running
#     if running:
#         running = False
#         print("Execution stopped successfully")
#     return "Execution stopped successfully"

# if __name__ == "__main__":
#     app.run(debug=True)

