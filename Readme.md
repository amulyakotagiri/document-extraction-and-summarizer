##How to run this project:

Open the live link at : https://project.amulyakotagiri.tech
You can find the test medical record samples in temp-data folder
Upload any sample record, choose summarization template and submit to get the summarization using AI.



##Conversely, to run this locally, 
Clone the git repo
create virtual env and activate : python -m venv venv && source venv/bin/activate
Install requirements : pip install -r requirements.txt
Run command : uvicorn main:app --port 7000 --reload 