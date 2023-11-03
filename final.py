from re import S
from urllib import response
import streamlit as st
import openai as openai
import time
import os
from langchain.chains import RetrievalQA 
# import openapi-schema-pydantic
import langchain
# from dotenv import load_dotenv
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from pydantic import ( BaseModel, Field )
from langchain.tools import BaseTool
from typing import Type
from langchain.agents import create_sql_agent, AgentType
from langchain.agents import initialize_agent
import os
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains.openai_functions.openapi import get_openapi_chain
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import json
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator



template_string2="""As an Operations Service Assistant Bot, your primary role is to assist daily operations users in managing their tasks related to client transactions. 
These tasks include updating the payment limit for clients and releasing holds for transactions. 
To accomplish this, you need to query the SQL database to get client Master and client transactions data for answering user queries.
For high-risk clients, advise not to release holds or update risk values and advise to request wire payment from the client.
For low-risk clients, suggest releasing holds and updating risk values.
In responding to user queries, follow this format:
If the question is about how the day looks for operational users, check the number of transactions on hold for the current day. Inform them accordingly:
If there are numerous transactions on hold, inform them that their day will be busy due to a high workload.
If there are fewer transactions on hold, mention that their day will be more relaxed as they have fewer tasks to handle.
Important: you Can Update the data for only low-risk clients on user request and Respond only with the data retreived from database.
Note: While Providing Response, Categorize the list of transactions on hold along with client names, addresses, and payment methods into High Risk clients and Low Risk clients based on the risk analysis criteria\
and Do not use technical terminology like Database , system.
Please ensure that your responses are based solely on the data retrieved from our system. Do not create or fabricate data on your own.
Respond in friendly manner.
"""
# Output format:Provide the transaction information in tabular format.
# 
# ClientTransaction contains 

release_prompt= """ let's break it down step by step:
Step 1: Initial Risk Assessment
First, perform an initial assessment to determine whether the client is high risk or low risk based on your previous analysis.
Step 2: Actions for High-Risk Clients
If the client is deemed high risk, respond with "Forwarded request to level 2 for releasing the hold."
Step 3: Actions for Low-Risk Clients
For low-risk clients, update the hold status to 'N' in the clienttransaction table within the database for the respective transaction ID.
Please note that the clienttransaction table contains the transaction IDs. """

# For High risk clients, Respond " Forwarded request to level2 for releasing the hold" and\
#                    For Low risk Clients, Update the hold with N in clienttransaction table in the database for the respective transaction id
# First determine high-risk or low risk and then
template_string= """As an Operations Service Assistant Bot, your primary role is to assist daily operations users in managing their tasks related to client transactions. These tasks include updating the risk value for clients,releasing holds for transactions ans requesting wire payment. To accomplish this, you will retrieve client information and transaction data from the SQL database to answer user queries.
Here's a step-by-step breakdown of your role and the SQL queries involved:
If user asks question on transaction details on hold, use step1 and provide the information.
Step 1: Retrieve transactions on hold from the SQL database.
Example SQL Query:
SELECT * FROM ClientTransaction WHERE hold = 'Y';
Step 2: Retrieve Client Master details for the transactions on hold.
Example SQL Query:
SELECT * FROM ClientMaster WHERE clientid IN (SELECT clientid FROM ClientTransaction WHERE hold = 'Y');
Step 3: Analyze the data to determine the risk level of clients. Consider clients with the following conditions as high-risk:
Approach: 
First retreive the return indicators for the transactions
Example SQL Query:
SELECT COUNT(*) FROM ClientTransaction WHERE clientid = 'IN123' AND returnindicator = 'X'\
Second Retreive creditscore and client created date for the same client
Example SQL Query:
SELECT creditscore, createddate FROM ClientMaster where clientid = 'IN123'\
if below conditions are satisfied, Consider it as high-risk client else consider it as low-risk.
1.Credit score between 0 and 5
2.Created less than 2 months ago
3.More than one return indicator for transactions
Example SQL Query:
SELECT clientid
FROM (
    SELECT c.clientid,
           c.creditscore,
           c.createdate,
           COUNT(t.returnindicator) AS return_count
    FROM ClientMaster c
    JOIN ClientTransaction t ON c.clientid = t.clientid
    WHERE c.creditscore BETWEEN 0 AND 5
      AND c.createdate >= DATEADD(MONTH, -2, GETDATE())
    GROUP BY c.clientid, c.creditscore, c.createdate
) AS RiskyClients
WHERE RiskyClients.return_count > 1;
Step 4: Determine whether a client is high risk or low risk based on the analysis.
Step 5: Provide below suggestions to the operation users based on the risk analysis results.
For high-risk clients, advise not to release holds or update payment limit.
For low-risk clients, suggest releasing holds and updating Payment limit.
Update the data only for low-risk clients and should not update for High-risk clients
In responding to user queries, follow this format:
If the question is about how the day looks for operational users, check the number of transactions on hold for the current day. Inform them accordingly:
If there are numerous transactions on hold, inform them that their day will be busy due to a high workload.
If there are fewer transactions on hold, mention that their day will be more relaxed as they have fewer tasks to handle.
Additionally, categorize the list of transactions on hold along with client names, addresses, and payment methods into High Risk clients and Low Risk clients based on the risk analysis criteria.

"""

def set_png_as_page_bg():
    page_bg_img = '''
    <style>
    .stApp{
    background-image: url("https://adpselect.com/login/images/backgrounds/adpSelect.png");
    background-size: 50%;
    width:100%;
    background-repeat: no-repeat;
    }
    </style>
    ''' 
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
# Note: Do not use Riskvalue column for categorizing the high risk and low risk clients
# ```Categorize List of transactions on hold along with client name, addess and payment method into High Risk clients and Low Risk Clients```\

# promptdb = """System:You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
# \nUnless the user specifies in the question a specific number of examples to obtain, query for at most 10 results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
# \nNever query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
# \nPay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
# \nPay attention to use CURDATE() function to get the current date, if the question involves \"today\".
# \n If user asks the question on transactions on hold,
# \nget the data from the clienttransaction which are on hold from sql database join clientMaster Table on clientid.
# \nIfhold = 'Y' means transaction is on hold\
# \nIf hold = NULL means transaction is not on hold\
# \nIf return indicator = X means transaction for a particular client is returned.\
# \nIF return indicator = NULL means transaction for a particualr client is not returned
# \n\nUse the following format:\n\nQuestion: Question here\nSQLQuery: SQL Query to run\nSQLResult: Result of the SQLQuery\nAnswer: Final answer here.
# \n Question: {input}"""


if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    #    llm = ChatOpenAI()
# if 'buffer_memory' not in st.session_state:
#         st.session_state.buffer_memory=ConversationBufferWindowMemory(k=5,return_messages=True)
_ = load_dotenv(find_dotenv())
# llm = ChatOpenAI( model_name ="gpt-3.5-turbo",temperature=0)
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613",temperature=0)
# llm = ChatOpenAI(model_anme="gpt-3.5-turbo-instruct")
# llm = ChatOpenAI(model_name="gpt-3.5-turbo-instruct-0914")

db = SQLDatabase.from_uri(f"""mysql+pymysql://{os.environ['db_user']}:{os.environ['db_password']}@{os.environ['db_host']}/{os.environ['db_name']}""")



def get_holds(user_question): 
    # prompt_template = ChatPromptTemplate.from_template(template_string)
    # prompt_template.format(role="System")
    # # prompt =PromptTemplate( input_variables=["input"],template=template_string)
    # # system_message = SystemMessagePromptTemplate.from_template(template=template_string)
    # db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, top_k=10, use_query_checker=True ,prompt=prompt_template)
    # response = db_chain.run(user_question)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(llm=llm, toolkit=toolkit,verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS, suffix=template_string)
    response = agent.run(user_question)
    return response
def get_fraud_info(user_question):
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, top_k=5, use_query_checker=True)
    response = db_chain.run(user_question)
    return response
def release_holds(user_question):
    # db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)
    # response = db_chain.run(user_question)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(llm=llm, toolkit=toolkit,verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS, suffix=release_prompt,memory=st.session_state.memory)
    response = agent.run(user_question)
    return response
# def get_pdf   
def get_domain_info(user_question):
    loader = TextLoader('/Users/srikanthdenduluri/Downloads/ADPWIKI.txt', encoding='utf8')
    documents = loader.load()
    # raw_documents = TextLoader('/Users/srikanthdenduluri/Downloads/state_of_the_union.txt').load()
    # index = VectorstoreIndexCreator().from_loaders([loader])
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    db2 = Chroma.from_documents(texts, OpenAIEmbeddings())
    retriever = db2.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = qa.run(user_question)
    return  response

    embedding_vector = OpenAIEmbeddings().embed_query(user_question)
    docs = db2.similarity_search_by_vector(embedding_vector)
class GetholdsInput(BaseModel):
            user_question: str = Field(description="user_question")
class Currentholds(BaseTool):
            name = "get_holds"
            description = """" useful for retrieving client information and transactions from the database based on user queries.
            Useful to get transactions holds information, Payment limit and return history information of client.
            Useful in providing task information to client.
            Useful in updating payment limit of client and releasing the holds.
            If user asks how is his day, check all the transactions on hold and client details. suggest the tasks he need to perform.
            """
            args_schema: Type[BaseModel] = GetholdsInput
            def _run(self, user_question:str):
                response = get_holds(user_question)
                return response
            def _arun(self, user_question:str):
                raise NotImplementedError("error")
class GetDomainInfonput(BaseModel):
            user_question: str = Field(description="user_question")
class DomainInformation(BaseTool):
            name = "get_domain_info"
            description = """" useful for getting the information about ADP.
                """
            args_schema: Type[BaseModel] = GetDomainInfonput
            def _run(self, user_question:str):
                response = get_domain_info(user_question)
                return response
            def _arun(self, user_question:str):
                raise NotImplementedError("error")

class GetfraudInput(BaseModel):
            user_question: str = Field(description="user_question")
class FraudInformation(BaseTool):
            name = "get_fraud_info"
            description = """" useful for getting the fraud transaction information.
                """
            args_schema: Type[BaseModel] = GetfraudInput
            def _run(self, user_question:str):
                response = get_fraud_info(user_question)
                return response
            def _arun(self, user_question:str):
                raise NotImplementedError("error")
class ReleaseholdsInput(BaseModel):
            user_question: str = Field(description="user_question")
class Releaseholds(BaseTool):
            name = "release_holds"
            description = """" Useful to release holds on the transactions for the client.
                               
                """
            args_schema: Type[BaseModel] = ReleaseholdsInput
            def _run(self, user_question:str):
                response = release_holds(user_question)
                return response
            def _arun(self, user_question:str):
                raise NotImplementedError("error")

def main():
    langchain.debug = True
    load_dotenv()
    logo = 'https://images.sftcdn.net/images/t_app-icon-m/p/69c53218-c7a9-11e6-8ce5-5a990553aaad/3424445367/adp-mobile-solutions-logo'
    logo2 ='https://i.pinimg.com/736x/8b/16/7a/8b167af653c2399dd93b952a48740620.jpg'
    # db = SQLDatabase.from_uri("postgressql+psycopg2://postgres:dhoni@localhost:5432/postgres")
    st.set_page_config(page_title="Ops Care",page_icon=logo) 
    set_png_as_page_bg()
    st.header("Ops Care")
    if 'area_key' not in st.session_state:
     st.session_state.area_key = 1
  
    system_message = SystemMessagePromptTemplate.from_template(template=template_string2)
    response_container = st.container()
    # container for text box
    textcontainer = st.container()
    with textcontainer:
        user_question = st.empty()
        user_question = st.chat_input("I am here to help you!", key="input")
        if user_question:
            with st.spinner("Processing"):
                tools = [ Currentholds(), DomainInformation(),FraudInformation(),Releaseholds()]
                agent_kwargs = {
                 "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
                 "system_message": system_message
                                }
                if "memory" not in st.session_state:
                 st.session_state.memory = ConversationBufferMemory(memory_key="memory", return_messages=True)     
                agent = initialize_agent(tools , llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,
                                          agent_kwargs=agent_kwargs,memory=st.session_state.memory )
                                        #   max_iterations=3,early_stopping_method="generate"
                # agent = initialize_agent(tools , llm=llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
                #                           agent_kwargs=agent_kwargs,memory=st.session_state.memory )
                # agent = initialize_agent(tools , llm=llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
                #                           agent_kwargs=agent_kwargs,memory=st.session_state.memory )
                # agent = initialize_agent(tools , llm=llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
                #                           agent_kwargs=agent_kwargs,memory=st.session_state.memory )
                # agent = initialize_agent(tools , llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                #                           agent_kwargs=agent_kwargs,memory=st.session_state.memory )
                response = agent.run(user_question)
                st.session_state.requests.append(user_question)
                st.session_state.responses.append(response) 
    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                # for chunk in st.session_state.responses.split():
                    # full_response += chunk + " "
                    # time.sleep(0.05)
                    # mesage_placeholder.markdown(full_response+ "▌")
                    # message_placeholder.markdown(full_response)
                message(st.session_state['responses'][i],key=str(i),logo=logo)
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user', logo=logo2 )
    
    


if __name__ == '__main__':
    main()