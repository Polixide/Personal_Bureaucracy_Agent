from state import State, Document  # Import the shared state and document models.

state = State(  # Create an example state; unspecified fields keep their defaults.
    user_goal="Obtain a residence permit",  # Set the user's desired outcome.
    current_procedure="Residence permit application",  # Name the active procedure.
    uploaded_documents=[  # Add the two documents supplied by the user.
        Document(document_type="passport_copy", status="uploaded"),  # Record the passport.
        Document(document_type="valid_health_insurance", status="uploaded"),  # Record insurance.
    ],  # Finish the document list.
)  # Finish creating the state.


print(state)  # Display the resulting Pydantic state.
