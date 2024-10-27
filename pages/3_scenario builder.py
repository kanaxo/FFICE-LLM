import streamlit as st
from logics.scenario_builder import generate_ffice_scenario 
from utility import check_password

st.set_page_config(
    layout="centered",
    page_title="FF-ICE Scenario Builder"
)
# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
st.title("FF-ICE Scenario Builder")

st.write("""
This tool helps you build scenarios related to Flight and Flow Information for a Collaborative Environment (FF-ICE).
Please provide your scenario requirements below.
""")

st.write("""
        ### Disclaimer:
Do note that the generated scenario may have a lot of inaccuracies. For an accurate scenario, please consult the **___FF-ICE implementation guidance___** on the correct messages that should be exchanged.
""")

# Initialize the session state for scenario_input_disabled, button_disabled, and response
if "scenario_input_disabled" not in st.session_state:
    st.session_state.scenario_input_disabled = False

if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = False

if "generated_response" not in st.session_state:
    st.session_state.generated_response = None

st.write("""
### User input
""")

user_input = st.text_area(
    "### Enter your FF-ICE scenario requirements:", 
    height=50, 
    key="scenario_input", 
    disabled=st.session_state.scenario_input_disabled
)

# Disable the button if needed
generate_button = st.button("Generate Scenario", disabled=st.session_state.button_disabled)

if generate_button:
    if user_input:
        # Disable the text area and button immediately
        st.session_state.scenario_input_disabled = True
        st.session_state.button_disabled = True

        # Process the user input using the existing query handler
        with st.spinner("Generating..."):
            st.session_state.generated_response = generate_ffice_scenario(user_input)

        # Re-enable the text area and button after processing
        st.session_state.scenario_input_disabled = False
        st.session_state.button_disabled = False
    else:
        st.warning("Please enter your scenario requirements before generating.")

# Display the generated scenario if available
if st.session_state.generated_response:
    st.subheader("Generated FF-ICE Scenario:")
    st.write(st.session_state.generated_response)


