import streamlit as st
import openai as openai
from dotenv import load_dotenv
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

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


def main():
    load_dotenv()
    # db = SQLDatabase.from_uri("postgressql+psycopg2://postgres:dhoni@localhost:5432/postgres")
    st.set_page_config(page_title="Ops Care") 
    st.header("Ops Care")
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    llm = ChatOpenAI()
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)
    
    system_msg_template = SystemMessagePromptTemplate.from_template(template=template_string3)
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
    # container for chat history
    response_container = st.container()
    # container for text box
    textcontainer = st.container()
    with textcontainer:
        user_question = st.empty()
        user_question = st.text_input("I am here to help you!", value="", key="input")
        if user_question:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                # response = conversation.predict(input=user_question)
                db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
                response = db_chain.run(user_question)
            st.session_state.requests.append(user_question)
            st.session_state.responses.append(response) 
    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
    
    st.session_state["new_todo"] = ""
  
    # if st.button("Chat"):
    #       with st.spinner("Answering"):
    #         if user_question:
    #         #st.write("Hello world")
    #         #Create conversation chain
    #         st.session_state.conversation = get_conversation_chain(template_string)
    #         # conversation_chain.run(System)
    #         # response1 = get_completion_from_messages(template_string, temperature=0)
    #         # st.write(response1)
    #         if user_question:
    #             handle_userinput(user_question)
              




if __name__ == '__main__':
    main()
