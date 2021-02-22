"""
Text Clustering: handler

Author: Jinraj K R <jinrajkr@gmail.com>
Created Date: 1-Apr-2020
Modified Date: 1-May-2020
===================================

Execution starts from here...
It takes the parameters as mentioned in the sample below and
returns the clusters and slots
if export_results_to_csv is set true, then save the results into excel file

"""

from model.validation import validate

def _main(params, return_type):
    print("validating...")
    resp = validate(params)

    if type(resp) == str:
        print("validation alert - {}".format(resp))
        return {"status": "400", "message": resp, "data":""}
    else:
        print("process execution started")
        response_data = resp.execute(return_type)
        print("process completed!")
        return response_data
