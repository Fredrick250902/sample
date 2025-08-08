from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


def validate_sql_query(query: str):
    print(f"Generated SQL Query: {query}")  # Log the query
    valid_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "WITH"]
    query = query.strip()
    if not any(query.upper().startswith(keyword) for keyword in valid_keywords):
        raise ValueError(f"Invalid SQL query generated: {query}")
    return query


def execute_sql_and_get_response(db, query):
    try:
        query = validate_sql_query(query)
        response = db.run(query)
        print(f"SQL Query Response: {response}")  # Log the response
        return response
    except Exception as e:
        print(f"SQL Execution Error: {str(e)}")  # Log the error
        return f"Error executing SQL query: {str(e)}"


def get_sql_chain(db):
    template = """
    Based on the provided table schema and the user's question, generate a valid SQL query.
Follow these rules for specific question types:
1. If the question is about the **number of columns**, use INFORMATION_SCHEMA to get the count of columns in the specified table.
2. If the question is about **column names**, use INFORMATION_SCHEMA to retrieve the column names of the specified table.
3. If the question is about the **number of records (rows)**, generate a query using COUNT(*) to count the rows in the table.
4. For general **schema-related questions**, provide queries to fetch metadata about tables, columns, or database structure.
5. For all other data-specific questions, generate a query that retrieves the required data directly from the table.
Ensure the SQL query directly answers the user's question without any additional text or explanation.
Schema: {schema}
Question: {question}
SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        groq_api_key="API_KEY_HERE"
    )

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    praising_words = ["ok", "thank you", "thanks", "great", "awesome", "nice", "well done", "cool"]
    
    if any(word in user_query.lower() for word in praising_words):
        return "You're welcome! I'm here to help. 😊"
    sql_chain = get_sql_chain(db)
    template = """
You are an exceptional assistant known for delivering accurate and precise answers. 
Based on the provided schema, the user's question, the SQL query, and the "SQL response", generate a accurate straight forward natural language answer.
kindly generate the answer that exactly reflects the SQL response. 
Schema: {schema}
Question: {question}
SQL Query: {query}
SQL Response: {response}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        groq_api_key="API_KEY_HERE"
    )

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: execute_sql_and_get_response(db, vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    print(f"User  Query: {user_query}")
    print(f"Chat History: {chat_history}")
    try:
        result = chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
        print(f"Generated Response: {result}")  # Debugging log
        return result
    except Exception as e:
        error_message = f"Error in chain invocation: {str(e)}"
        print(error_message)  # Debugging log
        return error_message


# Streamlit Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")
st.title("Chat with MySQL")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", key="Password")
    st.text_input("Database", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("Connected to database!")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        try:
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            print(f"Final Response to Display: {response}")  # Debugging log

            # Display response in Streamlit
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(error_message)  # Debugging log

            st.markdown(error_message)
            st.session_state.chat_history.append(AIMessage(content=error_message))