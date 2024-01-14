"""
Flask server collecting metrics for how long it takes students in each section.

The client is the reporting functions in utils.py. Run like:

MLAB_SERVER='http://localhost:5000' MLAB_EMAIL='chris@rdwrs.com' python w2d3_solution.py

Metrics are just dumped to a CSV for later data analysis.
"""
import csv
import datetime
import http
from flask import Flask, request


app = Flask(__name__)

METRICS_FILE = "metrics.csv"
outfile = open(METRICS_FILE, "a", newline="")
writer = csv.writer(outfile)


@app.route("/api/report_success", methods=["POST"])
def report_success():
    now = datetime.datetime.now().isoformat()
    assert request.json is not None
    writer.writerow([now, request.json["email"], request.json["testname"]])
    outfile.flush()
    return ("", http.HTTPStatus.NO_CONTENT)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
    # app.run(host="0.0.0.0", debug=True)
