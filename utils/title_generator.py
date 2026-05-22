from llm.prompt_build import generate_chat_title
def finalize_title(session, llm):
    # Generate title only if title is missing and messages exit
    if not session.get('title') and len(session['messages']) > 0:
        title = generate_chat_title(llm, session['messages'])

        session['title'] = title
        session['title_generated'] = True