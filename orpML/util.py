import json

def PrintOnce(message = None):

    """Prints a message only once during GridSearchCV.
    Use message = None to reset the saved messages file.
    """

    msgfile = "/tmp/global_messages.json"

    if message == None:
        # Initialize the messages with an empty list
        messages = []
    else:
        try:
            # Load messages from the file
            with open(msgfile, 'r') as f:
                messages = json.load(f)
        except:
            # If the file is unreadable, set messages to empty list
            messages = []

    if not message == None:
        # Print the message if it's not listed in the file
        if not any(message in x for x in messages):
            messages.append(message)
            print(message)

    # Save all the used messages to the file
    with open(msgfile, 'w') as f:
        # indent=2 is not needed but makes the file human-readable if the data is nested
        json.dump(messages, f, indent=2)

