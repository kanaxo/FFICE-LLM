import os
# from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
import streamlit as st

# load_dotenv('.env')

# Create the WebsiteSearchTool for FF-ICE documentation
# ffice_tool = SeleniumScrapingTool(
#     website_url = "https://docs.fixm.aero/#/fixm-in-support-of-ffice/message-templates",
#     css_element = "table"
#     )

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

ffice_message_info = """
**Flight Planning Process Overview**

- Initially, Airlines can choose from:
  - **Planning** (Preliminary Flight Plan or PFP, later confirmed with Filed Flight Plan or FFP),
  - **Direct Filing** (FFP submitted directly without PFP),
  - **Trialing** (Trial Request for testing routes or timings without commitment).

---

**Service Categories and Message Types**

1. **Filing Service**
Messages related to Filing Service:
   - **Filed Flight Plan (FFP)**: Submit a confirmed flight plan. Only after FFP is submitted, then the airline can use FPU or Flight Cancellation regarding the FFP.
   - **Filing Status** (ACCEPTABLE, NOT ACCEPTABLE): Status update for FFP; Should not be used for PFP.
   - **FFP Flight Plan Update (FPU)**: Modify an existing FFP.
   - **Flight Cancellation**: Cancel an existing FFP.
  
2. **Planning Service**
Messages related to Planning Service:
   - **Preliminary Flight Plan (PFP)**: Initial, unconfirmed flight plan.
   - **Planning Status** (CONCUR, NON-CONCUR, NEGOTIATE): Status update for PFP; Should not be used for FFP.
   - **PFP Flight Plan Update (FPU)**: Modify an existing PFP. Does not replace FFP to confirm flight plan.
   - **Flight Cancellation**: Cancel an existing PFP.

3. **Trial Service**
Messages related to Trial Service:
   - **Trial Request**: Request ATC to conduct a trial on a planned route.
   - **Trial Response** (CONCUR, NON-CONCUR, NEGOTIATE): ATC's response to a Trial Request.

4. **Data Request Service**
Messages related to Data Request Service:
   - **Flight Data Request**: Request details on a specific flight (FFP, PFP, or status).
   - **Flight Data Response**: Respond to a Flight Data Request with relevant flight details.

5. **Notification Service**
A FFP MUST be filed before ATC can send a notification to the airline.
Messages related to Notification Service:
   - **Arrival Notification**: ATC notification upon flight arrival.
   - **Departure Notification**: ATC notification upon flight departure.

6. **Acknowledgment Service**
Messages related to Acknowledgment Service:
   - **Submission Response** (ACK, REJ, MAN): ANSP acknowledgment for received messages (FFP, PFP, or requests), indicating **ACK** (acknowledged), **REJ** (rejected), or **MAN** (manual processing).

7. **Re-evaluation Service**
   - ATC review of FFP or PFP due to changed conditions (e.g., weather, traffic), potentially updating the status and sending **Filing Status** or **Planning Status** to airlines.

8. **Publication Service**
   - **NOTAMs**, **AIRMETs**, **SIGMETs**, **TAFs**, **METARs**, **TFRs**: Published by ANSPs through **SWIM** to inform airlines of airspace events, restrictions, and updates. These may affect planning or filing statuses.
   - Allows airlines to subscribe to updates on airspace conditions, restrictions, and other relevant information published by ANSPs.
"""

class FFICE_Tool(BaseTool):
    name: str = "FF-ICE Message Templates and Services"
    description: str = "Provides information about FF-ICE message types and their usage."
    message_info: str 

    def __init__(self, message_info: str, **data):
        super().__init__(message_info=message_info, **data)

    def _run(self, query: str = "") -> str:
        """
        Returns the full message information for the agent to interpret.
        The query parameter is kept for compatibility but not used.
        """
        return self.message_info

# Create the tool instance
ffice_tool = FFICE_Tool(message_info=ffice_message_info)

synthesizer = Agent(
    role = "Synthesizer",
    goal= "Expand the following scenario in a FF-ICE context in an interesting manner: {topic}. You must ensure that it makes sense in current context",
    backstory= """You need to create a story that is interesting to read when FF-ICE use cases are presented to audience who have not heard of FF-ICE before (including airlines and ATC). 
    Your story will allow FF-ICE Expert to decide on what messages airline and ANSPs should send for flight planning.
    """,
    verbose = True
)

writer = Agent(
    role = "Summarizer",
    goal = "Consolidates the scenario provided by the Scenario Expander, and creates a table from the messages as mentioned by Airline and ATC agents.",
    backstory = """You are a summarizer who consolidates the scenario provided by the Scenario Expander, and creates a table from the messages as mentioned by Airline and ATC agents.""",
    verbose = True
)

ffice_expert = Agent(
    role = "FF-ICE Expert",
    goal = "Decide on what messages both ATC and airline actors send to each other during flight planning based on the scenario provided by the Scenario Expander.",
    backstory = """You are an expert in FF-ICE and there are airlines and ATC who have no clue about FF-ICE. They are currently using FPL2012 mechanism to do flight planning using ATS messages.You need to help them decide on what FF-ICE messages both actors send to each other during flight planning based on the scenario provided by the Scenario Expander. This will show them the correct FF-ICE message types to use, how to use them, and the benefits of using FF-ICE.""",
    verbose = True
)

# airline_agent = Agent(
#     role = "Airline Agent",
#     goal = "Decide on what messages to send for flight planning based on the scenario provided by the Scenario Expander.",
#     backstory = """You are an airline agent who needs to decide on what FF-ICE messages to send for flight planning based on the scenario provided by the Scenario Expander.
#     You need to consider the weather, traffic, and other conditions when deciding on what messages to send.""",
#     verbose = True
# )

# atc_agent = Agent(
#     role = "ATC Agent",
#     goal = "Decide on what messages to send for flight planning based on the scenario provided by the Scenario Expander.",
#     backstory = """You are an ATC agent who needs to decide on what FF-ICE messages to send for flight planning based on the scenario provided by the Scenario Expander.
#     If there are multiple ATC agents, specify which ANSP is sending what message back to airline
#     """,
#     verbose = True
# )   

task_expand = Task(
    description = "Expand the scenario in a FF-ICE context: {topic}",
    expected_output = "A detailed scenario in a 2-5 sentences with a random flight, and the condition as mentioned in the scenario.",
    agent = synthesizer
)

task_messages = Task(
    description = """Analyze the scenario provided by the Scenario Expander and decide on what messages both ANSPs and airline actors should send to each other during flight planning. Use the FF-ICE Tool to get information about all available FF-ICE message types and services. Based on your understanding of the scenario and the FF-ICE messages, determine the most appropriate sequence of services used, messages to send, and actions to take.

    For each step in the flight planning process:
    1. Provide a brief description of what's happening in this step.
    2. Specify the FF-ICE message type that should be used, based on your understanding of the message types and the current situation.
    3. Indicate which actor (airline or specific ANSP) is sending the message.
    4. Explain why this message type is appropriate for this step in the process.

    Remember to consider the full context of the scenario, including weather conditions, potential delays, and the need for updates or changes to the flight plan. Use your expertise to showcase how FF-ICE improves upon the traditional FPL2012 mechanism.""",
    expected_output = """A detailed, step-by-step breakdown of the FF-ICE messages exchanged during the flight planning process, including:
    - Description of each step
    - The specific FF-ICE message type used
    - The sender of each message
    - A brief explanation of why each message type was chosen
    Ensure that the sequence of messages demonstrates the benefits and capabilities of FF-ICE compared to traditional flight planning methods.""",
    agent = ffice_expert,
    tools = [ffice_tool]
)

# task_messages_airlines = Task(
#     description = """Decide on what FF-ICE messages the airline needs to send for flight planning based on the scenario provided by the Scenario Expander.
#     First, use the SeleniumScrapingTool to fetch the table of FF-ICE message types. Then, for each message you decide to send, provide:
#     1. A brief description of what's happening in this step (e.g., 'Airline sends ATC a flight plan update after confirming the flight has been delayed')
#     2. The specific FF-ICE message type to be used (e.g., 'Flight Plan Update')
#     Only use valid FF-ICE message types from the table provided by the tool.""",
#     expected_output="A list of FF-ICE messages to send for flight planning, each with a description and the specific message type.",
#     agent = airline_agent,
#     tools = [ffice_tool],
#     async_execution = False
# )

# task_messages_atc = Task(
#     description = """Decide on what FF-ICE messages ATC needs to send for flight planning based on the scenario provided by the Scenario Expander.
#     First, use the SeleniumScrapingTool to fetch the table of FF-ICE message types. Then, for each message you decide to send, provide:
#     1. A brief description of what's happening in this step (e.g., 'ATC sends a planning status to the airline')
#     2. The specific FF-ICE message type to be used (e.g., 'Planning Status')
#     3. Specify which ANSP is sending the message
#     Only use valid FF-ICE message types from the table provided by the tool.""",
#     expected_output="A list of FF-ICE messages to send for flight planning, each with a description, the specific message type, and the sending ANSP.",
#     agent=atc_agent,
#     tools = [ffice_tool],
#     async_execution = False
# )

# task_summarize = Task(
#     description = "Consolidates the scenario provided by the Scenario Expander, and creates a table from the messages as mentioned by FF-ICE Expert.",
#     expected_output = """The sentences that were provided by Scenario Expander, and a table consisting of headers: 'Description','Message','Sender','Receiver' detailing the actions and messages being sent in 
#     a sequential manner from the messages as mentioned by FF-ICE Expert.""",
#     agent = writer,
# )

task_summarize = Task(
    description = """Consolidate the scenario provided by the Scenario Expander and the FF-ICE message exchanges detailed by the FF-ICE Expert. If there are no messages detailed by the FF-ICE Expert, prompt the FF-ICE Expert again. If not, create a comprehensive summary that:

    1. Outlines the initial scenario as mentioned by Scenario Expander.
    2. Outlines the key FF-ICE messages exchanged in the scenario, explaining their purpose and timing.
    3. Compares the FF-ICE process to the traditional FPL2012 process, highlighting:
       - Benefits specific to this scenario
       - If FF-ICE doesn't show significant improvement over FPL2012, say so.
    4. Conclude with key takeaways about how FF-ICE enhances flight planning and coordination between airlines and ANSPs.

    The goal is to create a narrative that helps readers understand how flight planning and coordination would work in an FF-ICE environment, and clearly demonstrates the improvements (if any) over current flight planning processes.""",
    expected_output = """A comprehensive summary that includes:
    - A brief description of the initial scenario as mentioned by Scenario Expander.
    - An overview of the FF-ICE messages exchanged (can use a table to show the messages exchanged between which actors)
    - A balanced comparison between FF-ICE and FPL2012 processes for this scenario, including both benefits and potential limitations
    - Clear explanations of the overall impact of FF-ICE

    The summary should be written in a way that's easy for someone unfamiliar with FF-ICE to understand its potential impact on flight planning processes.""",
    agent = writer,
)

crew = Crew(
    agents = [synthesizer, ffice_expert, writer],
    tasks = [task_expand, task_messages, task_summarize],
    verbose = True
)

print("Starting the crew...")
try:
    result = crew.kickoff(inputs = {"topic": "A flight from London to Paris is delayed due to bad weather."})
    print(f"The result is: {result.raw}")
except Exception as e:
    print(f"An error occurred: {e}")
